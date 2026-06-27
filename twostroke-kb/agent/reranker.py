"""Cross-encoder re-ranker: rescore top-20 candidates, keep top-5 for the LLM."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from config import get_settings


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def rerank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score (query, chunk) pairs and return the top RERANK_TOP_K.
    TODO: model().predict([(query, c['content']) for c in candidates]); sort desc."""
    raise NotImplementedError("TODO: cross-encoder rerank")
