"""Deduplicate near-identical chunks at ingest.

RULE (see CLAUDE.md): when cosine > threshold, KEEP one chunk but MERGE the
new source reference onto it. NEVER drop a source — conflict_check + citations
need to know every place a fact appears.
"""
from __future__ import annotations

from typing import Any

from config import get_settings


def dedup_and_merge(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare embeddings; for duplicates (cosine > threshold) merge source_refs
    into the retained chunk and skip the duplicate vector.

    TODO: compare against already-stored chunks too (not just within this batch).
    """
    threshold = get_settings().dedup_cosine_threshold
    raise NotImplementedError(f"TODO: dedup at cosine>{threshold}, merging provenance")
