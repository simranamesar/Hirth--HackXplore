"""Embed chunks and persist to pgvector; persist structured_facts from tables."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from config import get_settings


@lru_cache
def _embedder():
    """Load the multilingual embedding model once."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch-encode chunk content and attach an 'embedding' list to each chunk.

    Raises ValueError if the model returns unexpected dimension.
    """
    if not chunks:
        return chunks

    settings = get_settings()
    model = _embedder()
    texts = [c["content"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False)

    for c, vec in zip(chunks, vectors):
        if len(vec) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: got {len(vec)}, expected {settings.embedding_dim}"
            )
        c["embedding"] = vec.tolist()

    return chunks


def _doc_id_slug(filename: str) -> str:
    """Lowercase filename, replace spaces and special chars with underscores."""
    return re.sub(r"[^a-z0-9_.]", "_", filename.lower())


def store(chunks: list[dict[str, Any]]) -> None:
    """Insert chunks into the `chunks` table and table rows into `structured_facts`.

    Each chunk must already have an 'embedding' key (call embed() first).
    Uses a single transaction per batch.
    """
    from config import get_connection

    if not chunks:
        return

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            for c in chunks:
                doc_id = _doc_id_slug(c["metadata"].get("filename", "unknown"))
                lang = c["metadata"].get("lang", "unknown")
                embedding = c["embedding"]
                metadata = {k: v for k, v in c["metadata"].items() if k != "embedding"}
                source_refs = json.dumps(c.get("source_refs", []))

                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, content, lang, embedding, metadata, source_refs)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s::jsonb)
                    """,
                    (
                        doc_id,
                        c["content"],
                        lang,
                        str(embedding),
                        json.dumps(metadata),
                        source_refs,
                    ),
                )

                # Write table cells to structured_facts for exact spec lookup
                if c["metadata"].get("chunk_type") == "table":
                    _store_table_facts(cur, doc_id, c)

    finally:
        conn.close()


def _store_table_facts(cur: Any, doc_id: str, chunk: dict[str, Any]) -> None:
    """Parse a table chunk and insert individual cell values into structured_facts."""
    lines = chunk["content"].splitlines()
    if not lines:
        return

    # First data row is the header (after optional [Table: name] line)
    header: list[str] = []
    data_rows: list[list[str]] = []
    for line in lines:
        if line.startswith("[Table:"):
            continue
        if line.startswith("Units:"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if not header:
            header = cells
        else:
            data_rows.append(cells)

    sheet_name = chunk["metadata"].get("table_name", "")
    source_ref = json.dumps(chunk.get("source_refs", [{}])[0])

    for row in data_rows:
        row_label = row[0] if row else ""
        for col_idx, value in enumerate(row[1:], start=1):
            col_label = header[col_idx] if col_idx < len(header) else ""
            if value.strip():
                cur.execute(
                    """
                    INSERT INTO structured_facts
                        (doc_id, sheet, row_label, col_label, key, value, source_ref)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        doc_id,
                        sheet_name,
                        row_label,
                        col_label,
                        f"{row_label}::{col_label}",
                        value.strip(),
                        source_ref,
                    ),
                )
