"""Grounding verifier + gap detector.

is_grounded() gates the answer: every claim, especially NUMBERS, must be supported
by a retrieved source. Ungrounded -> loop back. Weak/no evidence -> log a gap.
"""
from __future__ import annotations

from typing import Any


def is_grounded(draft: str, context: list[dict[str, Any]]) -> bool:
    """TODO: LLM-judge or claim-matching that draft is supported by context.
    Be strict on numeric claims (reject if value not present in a source)."""
    raise NotImplementedError("TODO: grounding check")


def log_gap(question: str, reason: str) -> None:
    """Record a knowledge gap for experts. reason in
    {missing spec, missing procedure, weak evidence}."""
    raise NotImplementedError("TODO: write to gaps table")
