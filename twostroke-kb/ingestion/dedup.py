"""Deduplicate near-identical chunks at ingest.

RULE (see CLAUDE.md): when cosine > threshold, KEEP one chunk but MERGE the
new source reference onto it. NEVER drop a source — conflict_check + citations
need to know every place a fact appears.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from config import get_settings

log = logging.getLogger(__name__)


def dedup_and_merge(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each chunk's embedding against already-stored vectors.

    If cosine similarity > DEDUP_COSINE_THRESHOLD (default 0.98):
      - Merge the new chunk's source_refs onto the existing chunk in the DB.
      - Drop the new chunk from the return list (no duplicate vector stored).

    Returns the subset of chunks that should actually be inserted.
    """
    threshold = get_settings().dedup_cosine_threshold
    kept: list[dict[str, Any]] = []

    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            kept.append(chunk)
            continue

        existing_id = _find_near_duplicate(embedding, threshold)
        if existing_id is not None:
            log.info(
                "dedup: skipping chunk (cosine > %.2f); merging provenance onto id=%d",
                threshold, existing_id,
            )
            _merge_source_refs(existing_id, chunk.get("source_refs", []))
        else:
            kept.append(chunk)

    return kept


def _find_near_duplicate(embedding: list[float], threshold: float) -> int | None:
    """Return the id of the nearest stored chunk if cosine > threshold, else None."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS cosine
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 1
            """,
            (str(embedding), str(embedding)),
        )
        row = cur.fetchone()
        if row and float(row[1]) >= threshold:
            return int(row[0])
        return None
    finally:
        conn.close()


def _merge_source_refs(chunk_id: int, new_refs: list[dict[str, Any]]) -> None:
    """Append new_refs into the existing chunk's source_refs JSONB column.

    Uses PostgreSQL || operator so existing refs are preserved.
    """
    if not new_refs:
        return

    from config import get_connection

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE chunks
                SET source_refs = source_refs || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps(new_refs), chunk_id),
            )
    finally:
        conn.close()
