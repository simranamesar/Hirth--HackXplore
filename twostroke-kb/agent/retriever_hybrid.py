"""Hybrid retrieval: dense (pgvector) for MVP; BM25 fusion added in a later slice."""
from __future__ import annotations

import json
from typing import Any


def search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Return top-k chunks by cosine similarity using pgvector.

    BM25 sparse retrieval and reciprocal-rank fusion are deferred to the next slice.
    """
    from config import get_connection, get_settings
    from ingestion.knowledge_base import _embedder

    settings = get_settings()
    top_k = k if k is not None else settings.retrieve_top_k

    model = _embedder()
    vec = model.encode([query], show_progress_bar=False)[0].tolist()

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, doc_id, content, source_refs, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (str(vec), str(vec), top_k),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        chunk_id, doc_id, content, source_refs_raw, metadata_raw, score = row
        results.append({
            "id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "source_refs": json.loads(source_refs_raw) if isinstance(source_refs_raw, str) else source_refs_raw,
            "metadata": json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw,
            "score": float(score),
        })

    return results
