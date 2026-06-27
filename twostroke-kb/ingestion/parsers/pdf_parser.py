"""Digital PDF -> text + layout, with page/section metadata. Backend: PyMuPDF."""
from __future__ import annotations

from pathlib import Path

from ..types import ParsedDoc


def parse(path: str | Path) -> ParsedDoc:
    """Extract text per page. If text is empty, caller should fall back to ocr.parse.

    TODO: use fitz (PyMuPDF):
        import fitz
        doc = fitz.open(path)
        text = "\\n".join(page.get_text() for page in doc)
        keep page numbers in metadata; for FAR-style docs, capture clause numbers.
    """
    raise NotImplementedError("TODO: PyMuPDF text extraction")
