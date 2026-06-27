"""Memory layers: conversation (session), user profile (long-term), feedback.

NOTE: none of this retrains the model — it changes the context the model sees and
reweights retrieval.
"""
from __future__ import annotations

from typing import Any


def get_conversation(session_id: str) -> list[dict[str, Any]]:
    """Return prior turns for follow-up questions."""
    raise NotImplementedError


def append_turn(session_id: str, role: str, content: str) -> None:
    raise NotImplementedError


def get_profile(user_id: str) -> dict[str, Any]:
    """lang_pref, expertise, engines, ..."""
    raise NotImplementedError


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    vote: int = 0,
    correction: str = "",
    expert_note: str = "",
    chunk_ids: list[int] | None = None,
) -> None:
    """Store feedback; corrections become high-priority knowledge, votes reweight retrieval."""
    raise NotImplementedError
