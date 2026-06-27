"""Hybrid retrieval: BM25 (sparse) + pgvector (dense) fused via Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant; higher k reduces the impact of rank differences


def search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Return top-k chunks fused from BM25 and dense cosine retrieval.

    Each leg retrieves 2× top-k candidates; RRF scores are combined; final
    list is truncated to top-k and sorted by fused score descending.
    Falls back to dense-only if the BM25 index cannot be built (empty corpus).
    """
    from config import get_settings

    settings = get_settings()
    top_k = k if k is not None else settings.retrieve_top_k
    candidate_k = top_k * 2

    dense = _dense_search(query, candidate_k)

    try:
        sparse = _bm25_search(query, candidate_k)
    except Exception:
        log.warning("retriever_hybrid: BM25 failed; falling back to dense-only")
        sparse = []

    if not sparse:
        return dense[:top_k]

    return _rrf_fuse(dense, sparse, top_k)


# ---------------------------------------------------------------------------
# Dense leg
# ---------------------------------------------------------------------------

def _dense_search(query: str, top_k: int) -> list[dict[str, Any]]:
    from config import get_connection, get_settings
    from ingestion.knowledge_base import _embedder

    settings = get_settings()
    vec = _embedder().encode([query], show_progress_bar=False)[0].tolist()

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

    return [_row_to_chunk(r) for r in rows]


# ---------------------------------------------------------------------------
# BM25 leg
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer — handles both DE and EN without NLTK."""
    return text.lower().split()


def _bm25_search(query: str, top_k: int) -> list[dict[str, Any]]:
    from rank_bm25 import BM25Okapi
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, doc_id, content, source_refs, metadata FROM chunks"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    tokenized_corpus = [_tokenize(r[2]) for r in rows]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    # Pair scores with row index; take top-k by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    results: list[dict[str, Any]] = []
    for idx, score in ranked:
        if score <= 0:
            continue
        r = rows[idx]
        chunk = _row_to_chunk(r, score=score)
        results.append(chunk)

    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Combine two ranked lists via RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[int, dict[str, Any]] = {}

    for rank, chunk in enumerate(dense):
        cid = int(chunk["id"])
        fused.setdefault(cid, {**chunk, "score": 0.0})
        fused[cid]["score"] += 1.0 / (_RRF_K + rank + 1)

    for rank, chunk in enumerate(sparse):
        cid = int(chunk["id"])
        fused.setdefault(cid, {**chunk, "score": 0.0})
        fused[cid]["score"] += 1.0 / (_RRF_K + rank + 1)

    return sorted(fused.values(), key=lambda c: c["score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _row_to_chunk(row: tuple, score: float = 0.0) -> dict[str, Any]:
    chunk_id, doc_id, content = row[0], row[1], row[2]
    source_refs_raw = row[3]
    metadata_raw = row[4]
    raw_score = float(row[5]) if len(row) > 5 else score

    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "content": content,
        "source_refs": (
            json.loads(source_refs_raw)
            if isinstance(source_refs_raw, str)
            else (source_refs_raw or [])
        ),
        "metadata": (
            json.loads(metadata_raw)
            if isinstance(metadata_raw, str)
            else (metadata_raw or {})
        ),
        "score": raw_score,
    }
