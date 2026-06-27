"""Build the knowledge graph: Engine -> Part -> Symptom -> Cause -> Fix.

LLM extracts relationships from each ingested document and upserts them into
graph_nodes / graph_edges. Operations are idempotent via ON CONFLICT on (type, name).
"""
from __future__ import annotations

import logging
from typing import Any

from .types import ParsedDoc

log = logging.getLogger(__name__)

_TEXT_SAMPLE_LEN = 1500  # chars sent to the LLM; keep cost low


def extract(doc: ParsedDoc) -> None:
    """Extract Engine→Part→Symptom→Cause→Fix relationships and write to graph tables.

    Uses the first _TEXT_SAMPLE_LEN chars of the doc's prose. Silent no-op if the
    text is empty or the LLM / DB calls fail.
    """
    text = doc.text[:_TEXT_SAMPLE_LEN].strip()
    if not text:
        return

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
    except Exception:
        log.debug("graph_builder: LLM extraction failed for %s", doc.metadata.get("filename"))
        return

    if not isinstance(result, dict):
        return

    _upsert(result.get("nodes", []), result.get("edges", []))


def _upsert(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Upsert nodes then edges into graph_nodes / graph_edges."""
    from config import get_connection

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
                relation = str(edge.get("relation", "")).strip()
                src_id = node_ids.get(src_name)
                dst_id = node_ids.get(dst_name)
                if not src_id or not dst_id or not relation:
                    continue
                # Insert only if this exact edge does not exist yet
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
