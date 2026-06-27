"""PowerPoint (.pptx) -> slide text + speaker notes. Backend: python-pptx."""
from __future__ import annotations

from pathlib import Path

from ..types import ParsedDoc


def parse(path: str | Path) -> ParsedDoc:
    """TODO: python-pptx -> per-slide shapes text + notes_slide text."""
    raise NotImplementedError("TODO: python-pptx extraction")
