"""Recommendation (a challenge requirement): related questions + graph neighbours."""
from __future__ import annotations

from typing import Any


def related(question: str, used_chunks: list[dict[str, Any]]) -> list[str]:
    """Suggest 3 follow-up questions from retrieved content + graph neighbours.
    TODO: cheap version = LLM over the used chunks + graph_lookup on entities."""
    raise NotImplementedError("TODO: related questions / recommendations")
