"""Detect file type + language, dispatch to the right parser.

The router is the single entry point of GRAPH 1. Add new formats here.
"""
from __future__ import annotations

from pathlib import Path

from .types import ParsedDoc

# extension -> handler name (see parsers/)
EXT_MAP: dict[str, str] = {
    ".pdf": "pdf_or_ocr",      # decide digital vs scanned inside
    ".docx": "docx",
    ".doc": "doc_legacy",
    ".xlsx": "sheet",
    ".csv": "sheet",
    ".pptx": "pptx",
    ".txt": "text",
    ".md": "text",
    ".html": "text",
    ".png": "ocr",
    ".jpg": "ocr",
    ".jpeg": "ocr",
    ".url": "link",
}


def detect_language(text: str) -> str:
    """Return ISO code ('de' | 'en' | ...)."""
    try:
        from langdetect import detect

        return detect(text) if text.strip() else "unknown"
    except Exception:
        return "unknown"


def route(path: str | Path) -> ParsedDoc:
    """Dispatch a single uploaded file to its parser and return a ParsedDoc.

    TODO: import and call the matching parser from .parsers, then set
    metadata['lang'] = detect_language(doc.text).
    """
    path = Path(path)
    handler = EXT_MAP.get(path.suffix.lower())
    if handler is None:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Example wiring (uncomment as parsers are implemented):
    # if handler == "pdf_or_ocr":
    #     from .parsers import pdf_parser, ocr
    #     doc = pdf_parser.parse(path)
    #     if not doc.text.strip():          # scanned -> fall back to OCR
    #         doc = ocr.parse(path)
    # elif handler == "sheet":
    #     from .parsers import sheet_parser; doc = sheet_parser.parse(path)
    # ... etc
    # doc.metadata["lang"] = detect_language(doc.text)
    # return doc

    raise NotImplementedError(f"TODO: wire parser for handler '{handler}'")
