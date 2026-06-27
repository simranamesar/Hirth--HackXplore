"""STRETCH / phase 2 — figure & diagram intelligence.

Extract embedded figures from PDFs, caption them with a vision-LLM (searchable text),
OCR callout numbers and link to the parts graph, store images for source_viewer.
Gated by ENABLE_FIGURE_HANDLER in .env. Not MVP.
"""
from __future__ import annotations

from pathlib import Path

from ..types import ParsedDoc


def extract_and_caption(path: str | Path) -> ParsedDoc:
    """TODO (only if ENABLE_FIGURE_HANDLER):
        - PyMuPDF: page.get_images() -> extract image bytes
        - vision-LLM: caption each figure -> text chunk
        - OCR callouts -> link numbers to parts (graph_builder)
        - store images -> figure store; source_viewer returns them
    """
    raise NotImplementedError("STRETCH: figure handler not built for MVP")
