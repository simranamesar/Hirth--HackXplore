"""LangGraph nodes for the ReAct agent. Each takes the shared state dict and
returns a dict of keys to update (LangGraph merges them into state).

State lifecycle:
  reason -> [act -> observe -> reason]* -> draft -> verify -> END
"""
from __future__ import annotations

from typing import Any

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]


class AgentState(TypedDict):
    question: str
    lang: str
    scratch: list[dict[str, Any]]   # accumulated tool call results
    tool_action: str | None         # "hybrid_search" | "spec_lookup" | "answer"
    tool_args: dict[str, Any]       # args for the next tool call
    draft: str                      # composed answer text
    citations: list[dict[str, Any]] # citation objects for the API response
    grounded: bool                  # verifier result
    loops: int                      # iteration counter (guard against infinite loops)
    related: list[str]              # suggested follow-up questions


_MAX_LOOPS = 3

_TOOL_SCHEMA = """\
Available tools (respond with ONLY valid JSON, no prose):
  hybrid_search(query: str)                    — semantic search over uploaded documents
  spec_lookup(key: str, engine: str|null)      — exact spec values (RPM, torque, temp, etc.) from spreadsheets

When you have enough information to answer, use:
  {"thought": "...", "action": "answer", "args": {}}

Otherwise use a tool:
  {"thought": "...", "action": "hybrid_search", "args": {"query": "..."}}
  {"thought": "...", "action": "spec_lookup",   "args": {"key": "...", "engine": null}}
"""


def _scratch_summary(scratch: list[dict[str, Any]]) -> str:
    if not scratch:
        return ""
    parts: list[str] = []
    for i, item in enumerate(scratch, 1):
        tool = item.get("tool", "result")
        results = item.get("results", [])
        excerpts = []
        for r in results[:3]:
            if isinstance(r, dict):
                excerpts.append(r.get("content") or r.get("value") or str(r))
            else:
                excerpts.append(str(r))
        parts.append(f"[Tool {i}: {tool}]\n" + "\n".join(excerpts))
    return "\n\n".join(parts)


def reason(state: AgentState) -> dict[str, Any]:
    """LLM decides: enough info to answer, or which tool to call next."""
    from llm import chat_json

    summary = _scratch_summary(state["scratch"])
    system = (
        "You are a ReAct reasoning agent for two-stroke engine knowledge.\n"
        + _TOOL_SCHEMA
        + ("\n\nTool results so far:\n" + summary if summary else "")
    )

    decision = chat_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {state['question']}"},
        ],
        max_tokens=256,
    )

    action = str(decision.get("action", "answer"))
    args = decision.get("args") or {}
    if not isinstance(args, dict):
        args = {}

    # Guard: force answer after max loops
    if state["loops"] >= _MAX_LOOPS:
        action = "answer"

    return {
        "tool_action": action,
        "tool_args": args,
        "loops": state["loops"] + 1,
    }


def act(state: AgentState) -> dict[str, Any]:
    """Dispatch the chosen tool and append its result to scratch."""
    from agent.tools import TOOLS

    tool_name = state["tool_action"]
    args = state.get("tool_args") or {}

    if tool_name == "answer" or tool_name not in TOOLS:
        return {}

    try:
        results = TOOLS[tool_name](**args)
        if not isinstance(results, list):
            results = [results]
    except NotImplementedError:
        results = [{"content": f"Tool '{tool_name}' is not yet implemented.", "error": True}]
    except Exception as exc:
        results = [{"content": f"Tool error: {exc}", "error": True}]

    return {"scratch": state["scratch"] + [{"tool": tool_name, "args": args, "results": results}]}


def observe(state: AgentState) -> dict[str, Any]:
    """Hook for post-tool observation (e.g. summarising a long result set).

    Currently a pass-through: reason() already reads scratch directly.
    """
    return {}


def draft(state: AgentState) -> dict[str, Any]:
    """Compose a cited answer in the user's language from all accumulated evidence."""
    from llm import chat

    # Collect all evidence chunks from every tool call in scratch
    evidence: list[dict[str, Any]] = []
    for item in state["scratch"]:
        for r in item.get("results", []):
            if isinstance(r, dict) and not r.get("error"):
                evidence.append(r)

    if not evidence:
        return {
            "draft": "I don't have enough information in the knowledge base to answer that question.",
            "citations": [],
        }

    context_lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for i, c in enumerate(evidence, 1):
        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        label = ref.get("filename") or c.get("doc_id") or "unknown"
        page = ref.get("page", "")
        cite_label = f"{label} p.{page}" if page else label
        context_lines.append(f"[Source {i}] ({cite_label})\n{c.get('content', c.get('value', ''))}")
        citations.append({
            "n": i,
            "doc_id": c.get("doc_id", ""),
            "filename": label,
            "page": page,
            "snippet": str(c.get("content", c.get("value", "")))[:200],
        })

    lang = state.get("lang", "en")
    lang_note = f" Answer in {lang}." if lang not in ("en", "unknown", "") else ""
    context = "\n\n".join(context_lines)

    answer_text = chat(
        [
            {
                "role": "system",
                "content": (
                    "You are TwoStrokeGPT, an expert on two-stroke engines. "
                    "Answer ONLY using the numbered sources provided below. "
                    "Cite each fact as [Source N] immediately after the claim. "
                    "If a numeric value (RPM, temperature, timing, torque, etc.) "
                    "is NOT explicitly stated in a source, say you don't know — "
                    "never invent or estimate a value."
                    + lang_note
                    + f"\n\nSources:\n{context}"
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return {"draft": answer_text, "citations": citations}


def verify(state: AgentState) -> dict[str, Any]:
    """Grounding check: every numeric claim in draft must appear in a source.

    Also generates 2–3 related follow-up questions while we have the context loaded.
    Logs a gap if the answer is ungrounded and we've exhausted retries.
    """
    from agent import verifier as v

    context: list[dict[str, Any]] = []
    for item in state["scratch"]:
        context.extend(item.get("results", []))

    grounded = v.is_grounded(state["draft"], context)

    if not grounded and state["loops"] >= _MAX_LOOPS:
        v.log_gap(state["question"], "weak evidence: verifier could not ground all numeric claims")

    # Related questions (best-effort; never block on failure)
    related: list[str] = []
    if context:
        try:
            from llm import chat_json

            raw = chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Suggest 2–3 concise follow-up questions a two-stroke engine "
                            "technician might ask after this query. Return a JSON array of strings."
                        ),
                    },
                    {"role": "user", "content": state["question"]},
                ],
                max_tokens=150,
            )
            if isinstance(raw, list):
                related = [str(q) for q in raw[:3]]
        except Exception:
            pass

    return {"grounded": grounded, "related": related}
