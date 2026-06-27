"""Build the knowledge graph: Engine -> Part -> Symptom -> Cause -> Fix."""
from __future__ import annotations

from .types import ParsedDoc


def extract(doc: ParsedDoc) -> None:
    """TODO: LLM-extract relationships from the doc and upsert into
    graph_nodes / graph_edges. Idempotent (UNIQUE on type+name)."""
    raise NotImplementedError("TODO: relationship extraction -> graph tables")
