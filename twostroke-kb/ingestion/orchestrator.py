"""GRAPH 1 — the ingestion pipeline as a linear sequence.

route -> normalize -> chunk -> enrich -> embed -> dedup -> store -> graph_build

Re-uploading the same file (by name) bumps the version in the documents table
so the provenance trail remains intact without touching existing chunks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    filename: str
    chunks: int
    facts: int
    skipped_duplicates: int
    version: int = field(default=1)


def _slug(name: str) -> str:
    """Turn a filename stem into a stable doc_id slug."""
    return re.sub(r"[^\w.-]", "_", Path(name).stem.lower())


def _register_document(doc_id: str, filename: str, lang: str, storage_uri: str | None = None) -> int:
    """Upsert a new version row in the documents table; returns the new version number.

    Each re-upload of a file with the same doc_id increments the version instead
    of overwriting, so the full upload history is preserved.
    """
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(MAX(version), 0) FROM documents WHERE doc_id = %s",
            (doc_id,),
        )
        version = cur.fetchone()[0] + 1
        cur.execute(
            """
            INSERT INTO documents (doc_id, version, filename, lang, storage_uri)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (doc_id, version, filename, lang, storage_uri),
        )
        conn.commit()
        return version
    finally:
        conn.close()


def run_ingestion(path: str | Path) -> IngestResult:
    """Run the full ingestion pipeline for one uploaded file. Synchronous.

    Returns IngestResult with chunk count, fact count, duplicate count, and version.
    """
    from . import format_router, corpus_builder, chunker, knowledge_base
    from . import domain_enricher, dedup, graph_builder

    path = Path(path)

    # 1. Parse
    doc = format_router.route(path)

    # 2. Register document version (best-effort; never block ingestion)
    version = 1
    try:
        doc_id = _slug(path.name)
        lang = doc.metadata.get("lang", "unknown")
        version = _register_document(doc_id, path.name, lang, storage_uri=str(path))
        if version > 1:
            log.info("orchestrator: %s re-uploaded → version %d", path.name, version)
    except Exception:
        log.warning("orchestrator: could not register document version for %s; continuing", path.name)

    # 3. Normalize
    clean = corpus_builder.normalize(doc)

    # 4. Chunk (prose + tables)
    chunks = chunker.chunk(clean)

    # 5. Enrich with entities + tags (best-effort; LLM failure → skip silently)
    try:
        chunks = domain_enricher.enrich(chunks)
    except Exception:
        log.warning("orchestrator: domain_enricher failed for %s; continuing", path.name)

    # 6. Embed
    chunks = knowledge_base.embed(chunks)

    # 7. Dedup — compare against stored vectors; merge provenance, skip duplicates
    before = len(chunks)
    try:
        chunks = dedup.dedup_and_merge(chunks)
    except Exception:
        log.warning("orchestrator: dedup failed for %s; storing all chunks", path.name)
    skipped = before - len(chunks)

    # 8. Store chunks + structured_facts
    knowledge_base.store(chunks)

    # 9. Graph extraction (best-effort; LLM failure → skip silently)
    try:
        graph_builder.extract(clean)
    except Exception:
        log.warning("orchestrator: graph_builder failed for %s; continuing", path.name)

    fact_count = sum(
        1 for c in chunks if c["metadata"].get("chunk_type") == "table"
    )

    return IngestResult(
        filename=path.name,
        chunks=len(chunks),
        facts=fact_count,
        skipped_duplicates=skipped,
        version=version,
    )
