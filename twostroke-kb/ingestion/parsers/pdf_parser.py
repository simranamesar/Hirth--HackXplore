"""Digital PDF -> text + layout, with page/section metadata. Backend: PyMuPDF."""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from ..types import ParsedDoc

log = logging.getLogger(__name__)


def parse(path: str | Path) -> ParsedDoc:
    """Extract text per page. Returns ParsedDoc with text="" for scanned PDFs.

    Caller checks doc.text.strip() == "" to decide whether to fall back to OCR.
    """
    path = Path(path)
    doc = fitz.open(path)
    pages_text: list[str] = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()

    full_text = "\n".join(pages_text)
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
