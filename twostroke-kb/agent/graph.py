"""GRAPH 2 — ReAct diagnostic agent (LangGraph).

reason -> [act -> observe -> reason]* -> draft -> verify -> END

Falls back to plain retrieve->answer (Slice 1 path) if the graph misbehaves,
keeping CLAUDE.md rule 6: the demo must always have a working fallback.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes import (
    AgentState,
    act,
    draft,
    observe,
    reason,
    verify,
    _MAX_LOOPS,
)

log = logging.getLogger(__name__)

_graph = None  # compiled graph singleton — built once, reused


def _should_act(state: AgentState) -> str:
    """Route from reason: call a tool or go straight to drafting."""
    if state["tool_action"] == "answer" or state["loops"] > _MAX_LOOPS:
        return "draft"
    return "act"


def _should_end(state: AgentState) -> str:
    """Route from verify: accept answer or loop back for another round."""
    if state["grounded"] or state["loops"] > _MAX_LOOPS:
        return END
    return "reason"


def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("reason", reason)
    g.add_node("act", act)
    g.add_node("observe", observe)
    g.add_node("draft", draft)
    g.add_node("verify", verify)

    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _should_act, {"act": "act", "draft": "draft"})
    g.add_edge("act", "observe")
    g.add_edge("observe", "reason")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", _should_end, {END: END, "reason": "reason"})

    return g.compile()


def answer(question: str, session_id: str = "anon") -> dict[str, Any]:
    """Run the ReAct agent and return {answer, citations, confidence, related_questions}.

    Falls back to the plain retrieve->answer path if the graph raises.
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    from ingestion.format_router import detect_language

    initial: AgentState = {
        "question": question,
        "lang": detect_language(question),
        "scratch": [],
        "tool_action": None,
        "tool_args": {},
        "draft": "",
        "citations": [],
        "grounded": False,
        "loops": 0,
        "related": [],
    }

    try:
        result = _graph.invoke(initial)
    except Exception:
        log.exception("graph.answer: agent graph failed; falling back to plain retrieval")
        result = _plain_answer(question)

    return {
        "answer": result["draft"],
        "citations": result["citations"],
        "confidence": "high" if result.get("grounded") else "low",
        "related_questions": result.get("related", []),
    }


def _plain_answer(question: str) -> dict[str, Any]:
    """Plain retrieve -> answer fallback (CLAUDE.md rule 6)."""
    from agent.retriever_hybrid import search
    from config import get_settings
    from llm import chat

    settings = get_settings()
    chunks = search(question, k=settings.rerank_top_k)

    if not chunks:
        return {
            "draft": "I don't have enough information in the knowledge base to answer that question.",
            "citations": [],
            "grounded": True,
            "related": [],
        }

    context_lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, 1):
        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        label = ref.get("filename") or c.get("doc_id") or "unknown"
        page = ref.get("page", "")
        cite_label = f"{label} p.{page}" if page else label
        context_lines.append(f"[Source {i}] ({cite_label})\n{c['content']}")
        citations.append({
            "n": i,
            "doc_id": c.get("doc_id", ""),
            "filename": label,
            "page": page,
            "snippet": c["content"][:200],
        })

    answer_text = chat(
        [
            {
                "role": "system",
                "content": (
                    "You are TwoStrokeGPT, an expert on two-stroke engines. "
                    "Answer ONLY from the numbered sources below. "
                    "Cite each fact as [Source N]. Never invent numbers.\n\n"
                    + "\n\n".join(context_lines)
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return {"draft": answer_text, "citations": citations, "grounded": False, "related": []}
