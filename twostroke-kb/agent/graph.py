"""GRAPH 2 — ReAct diagnostic agent (LangGraph).

reason -> choose_tool -> act -> observe -> (loop) ; draft -> verify -> answer.
Keep a plain retrieve->answer fallback (CLAUDE.md rule 6).
"""
from __future__ import annotations

from typing import Any


def answer(question: str, session_id: str = "anon") -> dict[str, Any]:
    """Main entry point used by api/main.py /ask.

    Returns: {answer, citations, confidence, related_questions}.

    TODO: build the LangGraph StateGraph from agent.nodes and compile it; run it
    here. For the MVP you may start with the plain path:
        from .retriever_hybrid import search
        from .reranker import rerank
        ctx = rerank(search(question))
        draft = llm_answer(question, ctx)
        if verifier.is_grounded(draft, ctx): return draft + citations
    """
    raise NotImplementedError("TODO: compile + run ReAct graph")
