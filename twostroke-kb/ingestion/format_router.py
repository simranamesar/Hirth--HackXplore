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

    Sets metadata['lang'] via langdetect after parsing.
    Unsupported formats (sheet, pptx, ocr, doc_legacy) raise NotImplementedError.
    """
    import logging
    log = logging.getLogger(__name__)

    path = Path(path)
    handler = EXT_MAP.get(path.suffix.lower())
    if handler is None:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if handler == "pdf_or_ocr":
        from .parsers import pdf_parser
        doc = pdf_parser.parse(path)
        if not doc.text.strip():
            log.warning("route: %s has no extractable text; OCR not available in this slice", path.name)
    elif handler == "docx":
        from .parsers import docx_parser
        doc = docx_parser.parse(path)
    elif handler == "text":
        doc = ParsedDoc(
            text=path.read_text(errors="replace"),
            metadata={"filename": path.name, "type": "text"},
            source_ref={"filename": path.name},
        )
    elif handler == "link":
        from .parsers import link_handler
        doc = link_handler.parse(path)
    else:
        raise NotImplementedError(f"Parser not yet implemented for handler '{handler}' ({path.suffix})")

    doc.metadata["lang"] = detect_language(doc.text)
    return doc
