"""Chunk prose into overlapping passages; keep tables intact as single chunks."""
from __future__ import annotations

from typing import Any

from .types import ParsedDoc


def chunk(doc: ParsedDoc, size: int = 700, overlap: int = 100) -> list[dict[str, Any]]:
    """Return a list of chunk dicts: {content, metadata, source_refs}.
    Tables become one chunk each (don't split a table across chunks).
    TODO: implement token/char-based splitting with overlap.
    """
    raise NotImplementedError("TODO: chunking")
