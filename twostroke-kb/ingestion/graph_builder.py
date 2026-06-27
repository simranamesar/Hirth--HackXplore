"""Build the knowledge graph: Engine → Part → Symptom → Cause → Fix.

LLM extracts relationships from each ingested document and upserts them into
graph_nodes / graph_edges. When the LLM is unavailable, a regex fallback
extracts engine model names and part keywords so the graph is always populated.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .types import ParsedDoc

log = logging.getLogger(__name__)

_TEXT_SAMPLE_LEN = 1500

# Shared keyword lists (same as domain_enricher for consistency)
_ENGINE_RE   = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z]?\d{3,4}[A-Za-z]*))\b")
_PART_KEYWORDS = [
    "ignition", "carburetor", "carburettor", "cylinder", "piston", "crankshaft",
    "reed valve", "exhaust", "intake", "throttle", "spark plug", "gearbox",
    "bearing", "gasket", "flywheel", "magneto", "clutch", "connecting rod",
    "fuel pump", "oil pump", "transfer port", "main jet",
]
_PART_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _PART_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _regex_extract(text: str) -> dict[str, Any]:
    """Fallback graph extraction using regex patterns."""
    nodes: list[dict[str, str]] = []
    seen: set[str] = set()

    engine_names: list[str] = []
    for m in _ENGINE_RE.finditer(text):
        name = m.group(1).strip()
        key = name.lower()
        if key not in seen and any(c.isdigit() for c in name):
            nodes.append({"type": "engine", "name": name})
            seen.add(key)
            engine_names.append(name)

    part_names: list[str] = []
    for m in _PART_RE.finditer(text):
        name = m.group(1).lower()
        if name not in seen:
            nodes.append({"type": "part", "name": name})
            seen.add(name)
            part_names.append(name)

    # Create edges: each engine → HAS_PART → each co-occurring part
    # (co-occurrence = both found in same 200-char window)
    edges: list[dict[str, str]] = []
    for engine in engine_names:
        # Find engine position in text
        m = re.search(re.escape(engine), text)
        if not m:
            continue
        engine_pos = m.start()
        window = text[max(0, engine_pos - 100): engine_pos + 300]
        for part in part_names:
            if re.search(re.escape(part), window, re.IGNORECASE):
                edges.append({"src": engine, "dst": part, "relation": "HAS_PART"})

    return {"nodes": nodes, "edges": edges}


def extract(doc: ParsedDoc) -> None:
    """Extract Engine→Part→Symptom→Cause→Fix relationships and write to graph tables.

    Uses the first _TEXT_SAMPLE_LEN chars. Falls back to regex when LLM is down.
    Silent no-op if the text is empty or DB calls fail.
    """
    text = doc.text[:_TEXT_SAMPLE_LEN].strip()
    if not text:
        return

    result: dict[str, Any] | None = None

    # LLM extraction (best effort)
    try:
        from llm import chat_json

        result = chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract Engine→Part→Symptom→Cause→Fix relationships from this "
                        "two-stroke engine text. Return JSON: "
                        "{\"nodes\": [{\"type\": \"engine|part|symptom|cause|fix\", \"name\": str}], "
                        "\"edges\": [{\"src\": str, \"dst\": str, \"relation\": str}]}. "
                        "Only include what is explicitly stated; do not invent."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=400,
        )
        if not isinstance(result, dict):
            result = None
    except Exception:
        log.debug("graph_builder: LLM unavailable; using regex fallback")
        result = None

    # Regex fallback when LLM failed
    if result is None:
        result = _regex_extract(text)

    if not result:
        return

    _upsert(result.get("nodes", []), result.get("edges", []))


def _upsert(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Upsert nodes then edges into graph_nodes / graph_edges."""
    from config import get_connection

    if not nodes:
        return

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            node_ids: dict[str, int] = {}

            for node in nodes:
                ntype = str(node.get("type", "")).strip()
                nname = str(node.get("name", "")).strip()
                if not ntype or not nname:
                    continue
                cur.execute(
                    """
                    INSERT INTO graph_nodes (type, name)
                    VALUES (%s, %s)
                    ON CONFLICT (type, name) DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                    """,
                    (ntype, nname),
                )
                row = cur.fetchone()
                if row:
                    node_ids[nname] = int(row[0])

            for edge in edges:
                src_name = str(edge.get("src", "")).strip()
                dst_name = str(edge.get("dst", "")).strip()
                relation  = str(edge.get("relation", "")).strip()
                src_id = node_ids.get(src_name)
                dst_id = node_ids.get(dst_name)
                if not src_id or not dst_id or not relation:
                    continue
                cur.execute(
                    """
                    INSERT INTO graph_edges (src_id, dst_id, relation)
                    SELECT %s, %s, %s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM graph_edges
                        WHERE src_id = %s AND dst_id = %s AND relation = %s
                    )
                    """,
                    (src_id, dst_id, relation, src_id, dst_id, relation),
                )
    finally:
        conn.close()
