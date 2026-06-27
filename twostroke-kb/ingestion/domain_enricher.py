"""Extract tags + entities (de/en) per chunk via the LLM. Feeds metadata + graph."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_MIN_CONTENT_LEN = 80   # skip very short chunks — not enough context for extraction
_MAX_CONTENT_LEN = 600  # truncate to keep LLM calls cheap


def enrich(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach 'entities' and 'tags' to each chunk's metadata via LLM extraction.

    Skips chunks shorter than _MIN_CONTENT_LEN.
    LLM failures are caught per-chunk so one bad call never blocks the pipeline.
    Returns the same list with metadata updated in-place (copies made for safety).
    """
    enriched: list[dict[str, Any]] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if len(content) < _MIN_CONTENT_LEN:
            enriched.append(chunk)
            continue

        try:
            from llm import chat_json

            result = chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract named entities and topic tags from this two-stroke engine text. "
                            "Return JSON only: "
                            "{\"entities\": [{\"type\": \"engine|part|symptom|spec|unit\", \"name\": str}], "
                            "\"tags\": [str]}. "
                            "Be concise; only extract what is explicitly stated."
                        ),
                    },
                    {"role": "user", "content": content[:_MAX_CONTENT_LEN]},
                ],
                max_tokens=200,
            )

            chunk = dict(chunk)
            chunk["metadata"] = dict(chunk.get("metadata") or {})
            if isinstance(result, dict):
                chunk["metadata"]["entities"] = result.get("entities", [])
                chunk["metadata"]["tags"] = result.get("tags", [])

        except Exception:
            log.debug("domain_enricher: LLM extraction failed for chunk, skipping")

        enriched.append(chunk)

    return enriched
