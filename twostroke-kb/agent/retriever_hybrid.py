"""Hybrid retrieval: BM25 (sparse) + pgvector (dense) fused via Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant; higher k reduces the impact of rank differences

TOPIC_PRESETS = {
    "torque": "Drehmomente",
    "drehmoment": "Drehmomente",
    "fine tuning": "Feinstellung-Zweitaktmotor",
    "feinstellung": "Feinstellung-Zweitaktmotor",
    "zweitakt": "Feinstellung-Zweitaktmotor",
    "standard": "Normen DIN ISO VDI FAR ASTM LURS",
    "standards": "Normen DIN ISO VDI FAR ASTM LURS",
    "normen": "Normen DIN ISO VDI FAR ASTM LURS",
    "din": "Normen DIN ISO VDI FAR ASTM LURS",
    "iso": "Normen DIN ISO VDI FAR ASTM LURS",
    "far": "Normen DIN ISO VDI FAR ASTM LURS",
    "vibration": "Vibrationen",
    "vibrationen": "Vibrationen",
    "propeller": "Propeller",
    "combustion": "Verbrennungsmotoren",
    "engine": "Verbrennungsmotoren",
    "verbrennung": "Verbrennungsmotoren",
}


def infer_topic(query: str) -> str | None:
    """Infer a likely Hirth corpus topic from a question, if obvious."""
    folded = (query or "").casefold()
    for trigger, topic in TOPIC_PRESETS.items():
        if trigger in folded:
            return topic
    return None


def search(query: str, k: int | None = None, topic: str | None = None) -> list[dict[str, Any]]:
    """Return top-k chunks fused from BM25 and dense cosine retrieval.

    Each leg retrieves 2× top-k candidates; RRF scores are combined; final
    list is truncated to top-k and sorted by fused score descending.
    Falls back to dense-only if the BM25 index cannot be built (empty corpus).
    """
    from config import get_settings

    settings = get_settings()
    top_k = k if k is not None else settings.retrieve_top_k
    candidate_k = top_k * 2
    topic = (topic or "").strip() or None
    boost_topic = topic or infer_topic(query)

    dense = _dense_search(query, candidate_k, topic=topic)

    try:
        sparse = _bm25_search(query, candidate_k, topic=topic)
    except Exception:
        log.warning("retriever_hybrid: BM25 failed; falling back to dense-only")
        sparse = []

    if not sparse:
        results = dense[:top_k]
    else:
        results = _rrf_fuse(dense, sparse, top_k)

    if boost_topic:
        results = _apply_topic_boost(results, boost_topic)
    return _apply_feedback_boost(results)


# ---------------------------------------------------------------------------
# Dense leg
# ---------------------------------------------------------------------------

def _dense_search(query: str, top_k: int, topic: str | None = None) -> list[dict[str, Any]]:
    from config import get_connection, get_settings
    from ingestion.knowledge_base import _embedder

    settings = get_settings()
    vec = _embedder().encode([query], show_progress_bar=False)[0].tolist()

    conn = get_connection()
    try:
        cur = conn.cursor()
        if topic:
            cur.execute(
                """
                SELECT id, doc_id, content, source_refs, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE metadata->>'topic' = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (str(vec), topic, str(vec), top_k),
            )
        else:
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


def _bm25_search(query: str, top_k: int, topic: str | None = None) -> list[dict[str, Any]]:
    from rank_bm25 import BM25Okapi
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        if topic:
            cur.execute(
                "SELECT id, doc_id, content, source_refs, metadata FROM chunks WHERE metadata->>'topic' = %s",
                (topic,),
            )
        else:
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
# Feedback boost
# ---------------------------------------------------------------------------

def _apply_feedback_boost(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-score chunks using accumulated vote totals from the feedback table.

    vote_sum > 0 → small upward nudge; vote_sum < 0 → small downward nudge.
    Boost is intentionally mild (±10% max) so retrieval relevance still dominates.
    Returns list re-sorted by boosted score descending.
    """
    if not chunks:
        return chunks

    chunk_ids = [int(c["id"]) for c in chunks]
    vote_map: dict[int, int] = _fetch_vote_totals(chunk_ids)
    if not vote_map:
        return chunks

    for c in chunks:
        votes = vote_map.get(int(c["id"]), 0)
        # clamp multiplier: max ±10% adjustment
        multiplier = 1.0 + max(-0.10, min(0.10, votes * 0.02))
        c["score"] = c.get("score", 0.0) * multiplier

    return sorted(chunks, key=lambda c: c["score"], reverse=True)


def _fetch_vote_totals(chunk_ids: list[int]) -> dict[int, int]:
    """Return {chunk_id: sum(vote)} for chunks that appear in the feedback table."""
    try:
        from config import get_connection

        conn = get_connection()
        try:
            cur = conn.cursor()
            # feedback.chunk_ids is BIGINT[]; unnest to join
            cur.execute(
                """
                SELECT unnested_id, SUM(vote)
                FROM feedback, UNNEST(chunk_ids) AS unnested_id
                WHERE unnested_id = ANY(%s)
                GROUP BY unnested_id
                """,
                (chunk_ids,),
            )
            return {int(row[0]): int(row[1]) for row in cur.fetchall()}
        finally:
            conn.close()
    except Exception:
        log.debug("retriever_hybrid: could not fetch vote totals; skipping boost")
        return {}


def _apply_topic_boost(chunks: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    """Mildly boost chunks from an inferred topic without hiding other results."""
    if not chunks:
        return chunks
    for chunk in chunks:
        if _chunk_topic(chunk) == topic:
            chunk["score"] = float(chunk.get("score", 0.0)) * 1.18 + 0.01
            chunk["topic_boost"] = topic
    return sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)


def _chunk_topic(chunk: dict[str, Any]) -> str | None:
    metadata = chunk.get("metadata") or {}
    if metadata.get("topic"):
        return str(metadata["topic"])
    refs = chunk.get("source_refs") or []
    if refs and isinstance(refs[0], dict) and refs[0].get("topic"):
        return str(refs[0]["topic"])
    return None


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
