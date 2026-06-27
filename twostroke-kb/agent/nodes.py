"""LangGraph nodes for the ReAct agent. Each takes and returns the shared state dict.

State keys (suggested): question, lang, history, scratch (tool results),
draft, citations, grounded, related.
"""
from __future__ import annotations

from typing import Any

State = dict[str, Any]


def reason(state: State) -> State:
    """LLM decides: enough info to answer, or which tool to call next."""
    raise NotImplementedError


def choose_tool(state: State) -> State:
    """Pick a tool + arguments (model-driven function calling)."""
    raise NotImplementedError


def act(state: State) -> State:
    """Run the selected tool; append result to scratch."""
    raise NotImplementedError


def observe(state: State) -> State:
    """Summarize the new evidence back into the reasoning context."""
    raise NotImplementedError


def draft(state: State) -> State:
    """Compose an answer (in user's language) with inline citations + confidence."""
    raise NotImplementedError


def verify(state: State) -> State:
    """Grounding check: every claim (esp. numbers) supported by a source?
    Set state['grounded']; if False, loop back to reason. Log gaps if weak."""
    raise NotImplementedError
