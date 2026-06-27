"""Digital PDF → text with per-page sentinels for page-number tracking.

Each page boundary is marked with [PAGE_BREAK:N] (1-based).
The chunker strips these sentinels and uses them to record which page
each chunk came from in source_refs[0]["page"].
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from ..types import ParsedDoc

log = logging.getLogger(__name__)


def parse(path: str | Path) -> ParsedDoc:
    """Extract text per page, inserting [PAGE_BREAK:N] sentinels between pages.

    Returns ParsedDoc with text="" for scanned PDFs (caller falls back to OCR).
    The sentinel format is: \\n\\n[PAGE_BREAK:N]\\n\\n
    The chunker uses these markers to attach page numbers to each chunk.
    """
    path = Path(path)
    doc = fitz.open(path)
    pages_text: list[str] = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()

    # Join pages with sentinels so chunker can track page numbers
    # Page 1 is implicit (no leading sentinel); sentinels mark where page N begins
    if pages_text:
        parts: list[str] = [pages_text[0]]
        for i, page_text in enumerate(pages_text[1:], start=2):
            parts.append(f"\n\n[PAGE_BREAK:{i}]\n\n")
            parts.append(page_text)
        full_text = "".join(parts)
    else:
        full_text = ""

    if not full_text.strip():
        log.warning("pdf_parser: no extractable text in %s (scanned?)", path.name)

    return ParsedDoc(
        text=full_text,
        metadata={
            "filename": path.name,
            "pages": len(pages_text),
            "type": "pdf",
        },
        source_ref={"filename": path.name, "page_count": len(pages_text)},
    )
