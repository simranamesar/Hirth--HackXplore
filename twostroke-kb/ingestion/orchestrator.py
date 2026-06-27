"""GRAPH 1 — the ingestion pipeline as a linear LangGraph.

route -> normalize -> chunk -> enrich -> graph_build -> embed -> dedup -> store
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestResult:
    filename: str
    chunks: int
    facts: int
    skipped_duplicates: int


def run_ingestion(path: str | Path) -> IngestResult:
    """Run the full pipeline for one uploaded file. Synchronous for now.

    TODO: build a LangGraph StateGraph with one node per step below, or just call
    them in sequence for the MVP:

        from . import format_router, corpus_builder, chunker, domain_enricher,
                      graph_builder, knowledge_base, dedup
        doc   = format_router.route(path)
        clean = corpus_builder.normalize(doc)
        chunks = chunker.chunk(clean)
        chunks = domain_enricher.enrich(chunks)
        graph_builder.extract(clean)            # writes graph_nodes/edges
        chunks = knowledge_base.embed(chunks)
        kept   = dedup.dedup_and_merge(chunks)  # cosine>0.98 -> merge provenance
        knowledge_base.store(kept)              # write to pgvector + structured_facts
    """
    raise NotImplementedError("TODO: implement ingestion pipeline")
