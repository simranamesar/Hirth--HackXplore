"""Embed chunks and persist to pgvector; persist structured_facts from tables."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from config import get_settings


@lru_cache
def _embedder():
    """Load the multilingual embedding model once."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach an `embedding` vector to each chunk. TODO: batch encode."""
    raise NotImplementedError("TODO: embed chunks with the multilingual model")


def store(chunks: list[dict[str, Any]]) -> None:
    """Insert chunks into `chunks` and any table values into `structured_facts`.
    TODO: use pgvector; ensure embedding dim == settings.embedding_dim."""
    raise NotImplementedError("TODO: write to pgvector + structured_facts")
