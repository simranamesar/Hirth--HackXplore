"""GRAPH 1 — the ingestion pipeline as a linear sequence.

route -> normalize -> chunk -> embed -> store
(dedup, domain_enricher, graph_builder wired in later slices)
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
    """Run the full ingestion pipeline for one uploaded file. Synchronous.

    Returns IngestResult with counts of chunks and structured_facts written.
    """
    from . import format_router, corpus_builder, chunker, knowledge_base

    path = Path(path)

    doc = format_router.route(path)
    clean = corpus_builder.normalize(doc)
    chunks = chunker.chunk(clean)
    chunks = knowledge_base.embed(chunks)
    knowledge_base.store(chunks)

    fact_count = sum(
        1 for c in chunks if c["metadata"].get("chunk_type") == "table"
    )
    return IngestResult(
        filename=path.name,
        chunks=len(chunks),
        facts=fact_count,
        skipped_duplicates=0,
    )
