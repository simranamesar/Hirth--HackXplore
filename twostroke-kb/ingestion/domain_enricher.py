"""Extract tags + entities (de/en) per chunk via the LLM. Feeds metadata + graph."""
from __future__ import annotations

from typing import Any


def enrich(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TODO: for each chunk, prompt the LLM to extract entities
    (engine model, part, symptom, spec name, unit) and tags; attach to metadata.
    Keep it cheap: batch, and skip for very short chunks."""
    raise NotImplementedError("TODO: entity/tag extraction")
