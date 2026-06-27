"""Hybrid retrieval: BM25 (sparse) + dense (pgvector) fusion.

Good for exact terms (part numbers like '3503', units) AND semantic matches.
Multilingual: a DE query can match EN chunks via the shared embedding space.
"""
from __future__ import annotations

from typing import Any


def search(query: str, k: int = 20) -> list[dict[str, Any]]:
    """Return top-k chunks by fused score.
    TODO:
        - dense: embed query, pgvector cosine search
        - sparse: rank_bm25 over the corpus (or pg full-text)
        - fuse (e.g. reciprocal rank fusion), apply feedback weights
    """
    raise NotImplementedError("TODO: hybrid BM25 + dense retrieval")
