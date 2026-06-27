"""Memory layers: conversation (session), user profile (long-term), feedback.

NOTE: none of this retrains the model — it changes the context the model sees and
reweights retrieval.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def get_conversation(session_id: str) -> list[dict[str, Any]]:
    """Return prior turns for follow-up questions. Returns [] if session is new."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT turns FROM conversations WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return []
        turns = row[0]
        return json.loads(turns) if isinstance(turns, str) else (turns or [])
    finally:
        conn.close()


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the conversation; create the session row if absent."""
    from config import get_connection

    turn = json.dumps([{"role": role, "content": content}])
    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO conversations (session_id, turns)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (session_id) DO UPDATE
                SET turns      = conversations.turns || %s::jsonb,
                    updated_at = now()
                """,
                (session_id, turn, turn),
            )
    finally:
        conn.close()


def get_profile(user_id: str) -> dict[str, Any]:
    """Return the user profile, or sensible defaults for a new user."""
    from config import get_connection

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT lang_pref, expertise, engines, props FROM user_profile WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"lang_pref": None, "expertise": "expert", "engines": [], "props": {}}

    lang_pref, expertise, engines, props = row
    return {
        "lang_pref": lang_pref,
        "expertise": expertise or "expert",
        "engines": list(engines) if engines else [],
        "props": json.loads(props) if isinstance(props, str) else (props or {}),
    }


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    vote: int = 0,
    correction: str = "",
    expert_note: str = "",
    chunk_ids: list[int] | None = None,
) -> None:
    """Store feedback. Corrections become high-priority knowledge; votes reweight retrieval."""
    from config import get_connection

    conn = get_connection()
    try:
        with conn.transaction():
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback
                    (session_id, question, answer, vote, correction, expert_note, chunk_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    question,
                    answer,
                    vote,
                    correction or "",
                    expert_note or "",
                    chunk_ids or [],
                ),
            )
    finally:
        conn.close()

    # Promote non-empty corrections into the knowledge base as high-priority chunks
    if correction and correction.strip():
        _promote_correction(question, correction.strip(), session_id)


def _promote_correction(question: str, correction: str, session_id: str) -> None:
    """Embed a user correction and store it as a high-priority chunk.

    The chunk content is phrased as Q→A so the retriever can surface it when
    the same question (or a semantically similar one) is asked again.
    Silently swallows all errors so a failure never breaks the feedback path.
    """
    try:
        from ingestion.knowledge_base import embed, store

        content = f"Q: {question}\nA (correction): {correction}"
        chunk = {
            "content": content,
            "metadata": {
                "filename": f"correction:{session_id}",
                "lang": "unknown",
                "type": "correction",
                "chunk_type": "prose",
                "source": "user_feedback",
            },
            "source_refs": [{"source": "user_feedback", "session_id": session_id}],
        }
        chunks = embed([chunk])
        store(chunks)
        log.info("store: promoted correction from session %s into knowledge base", session_id)
    except Exception:
        log.warning("store: failed to promote correction from session %s", session_id)
