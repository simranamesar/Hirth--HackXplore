"""Chunk prose into overlapping passages; keep tables intact as single chunks."""
from __future__ import annotations

from typing import Any

from .types import ParsedDoc, Table


def _table_to_text(table: Table) -> str:
    """Render a Table as a compact text block for embedding."""
    lines: list[str] = []
    if table.name:
        lines.append(f"[Table: {table.name}]")
    for row in table.rows:
        lines.append(" | ".join(cell.strip() for cell in row))
    if table.units:
        lines.append("Units: " + ", ".join(f"{k}={v}" for k, v in table.units.items()))
    return "\n".join(lines)


def chunk(doc: ParsedDoc, size: int = 700, overlap: int = 100) -> list[dict[str, Any]]:
    """Return a list of chunk dicts: {content, metadata, source_refs}.

    Prose is split by sliding window (size chars, overlap chars).
    Each Table becomes one chunk — never split across chunks.
    """
    chunks: list[dict[str, Any]] = []
    idx = 0

    # Prose chunks
    text = doc.text
    step = size - overlap
    pos = 0
    while pos < len(text):
        content = text[pos : pos + size].strip()
        if content:
            chunks.append({
                "content": content,
                "metadata": {**doc.metadata, "chunk_index": idx, "chunk_type": "prose"},
                "source_refs": [doc.source_ref],
            })
            idx += 1
        pos += step
        if pos >= len(text):
            break

    # Table chunks — one per table, never split
    for table in doc.tables:
        content = _table_to_text(table)
        if content.strip():
            chunks.append({
                "content": content,
                "metadata": {**doc.metadata, "chunk_index": idx, "chunk_type": "table",
                              "table_name": table.name, "type": "table"},
                "source_refs": [doc.source_ref],
            })
            idx += 1

    return chunks
