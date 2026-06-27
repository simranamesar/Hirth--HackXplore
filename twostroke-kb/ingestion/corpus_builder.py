"""Normalize a ParsedDoc into clean text + table metadata ready for chunking."""
from __future__ import annotations

from .types import ParsedDoc


def normalize(doc: ParsedDoc) -> ParsedDoc:
    """TODO: strip boilerplate, fix whitespace/hyphenation, keep tables structured,
    ensure metadata has filename/lang/type/page. Return a cleaned ParsedDoc."""
    raise NotImplementedError("TODO: normalization")
