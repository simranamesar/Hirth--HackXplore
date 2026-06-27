"""Tools the agent can call (model-driven function calling).

Each tool returns evidence with source references attached.
"""
from __future__ import annotations

from typing import Any


def hybrid_search(query: str, k: int = 20) -> list[dict[str, Any]]:
    """BM25 + dense retrieval -> top-k candidate chunks (pre-rerank)."""
    from .retriever_hybrid import search

    return search(query, k=k)


def graph_lookup(entity: str, relation: str | None = None) -> list[dict[str, Any]]:
    """Traverse Engine->Part->Symptom->Cause->Fix from graph tables."""
    raise NotImplementedError


def spec_lookup(key: str, engine: str | None = None) -> list[dict[str, Any]]:
    """Return EXACT values from structured_facts (never computed). Always cite."""
    raise NotImplementedError


def conflict_check(claim: str) -> list[dict[str, Any]]:
    """Find sources that agree/disagree on a value (needs merged provenance)."""
    raise NotImplementedError


def unit_convert(value: float, from_unit: str, to_unit: str) -> float:
    """Deterministic unit conversion (not via the LLM)."""
    raise NotImplementedError


def diagnostic_tree(symptom: str, engine: str | None = None) -> list[dict[str, Any]]:
    """Walk symptom -> likely cause -> fix using the graph."""
    raise NotImplementedError


def source_viewer(chunk_id: int) -> dict[str, Any]:
    """Return the source passage/page (and figure image if available) for citing."""
    raise NotImplementedError


# registry exposed to the LLM as callable tool schemas
TOOLS = {
    "hybrid_search": hybrid_search,
    "graph_lookup": graph_lookup,
    "spec_lookup": spec_lookup,
    "conflict_check": conflict_check,
    "unit_convert": unit_convert,
    "diagnostic_tree": diagnostic_tree,
    "source_viewer": source_viewer,
}
