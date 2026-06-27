"""Modern Word (.docx) -> text + tables. Backend: python-docx."""
from __future__ import annotations

from pathlib import Path

from ..types import ParsedDoc


def parse(path: str | Path) -> ParsedDoc:
    """TODO: python-docx -> paragraphs + tables.
        from docx import Document
        d = Document(path)
        text = "\\n".join(p.text for p in d.paragraphs)
        tables -> ParsedDoc.tables
    """
    raise NotImplementedError("TODO: python-docx extraction")
