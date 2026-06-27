"""Scanned PDF / images -> OCR text (German + English). Needs system tesseract + poppler."""
from __future__ import annotations

from pathlib import Path

from ..types import ParsedDoc


def parse(path: str | Path, langs: str = "deu+eng") -> ParsedDoc:
    """OCR each page/image. Flag low-confidence pages in metadata for review.

    TODO:
        from pdf2image import convert_from_path  # PDFs
        import pytesseract
        text = pytesseract.image_to_string(img, lang=langs)
    """
    raise NotImplementedError("TODO: Tesseract OCR (deu+eng)")
