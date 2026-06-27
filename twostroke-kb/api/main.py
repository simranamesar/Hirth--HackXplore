"""FastAPI app: upload documents, ask questions, send feedback. Serves a minimal HTML UI."""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings

settings = get_settings()
app = FastAPI(title="TwoStrokeGPT")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """Save the upload and run the ingestion pipeline, returning chunk counts."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(await file.read())

    from ingestion.orchestrator import run_ingestion

    result = run_ingestion(dest)
    return JSONResponse({
        "filename": result.filename,
        "chunks": result.chunks,
        "facts": result.facts,
        "skipped_duplicates": result.skipped_duplicates,
        "version": result.version,
        "status": "indexed",
    })


@app.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form("anon"),
    expertise: str = Form("expert"),
) -> JSONResponse:
    """ReAct agent answer with citations, grounding check, and related questions.

    expertise: "beginner" (plain language) | "expert" (concise technical).
    Falls back to plain retrieve->answer automatically if the agent graph fails.
    """
    from agent.graph import answer
    from memory.store import append_turn

    result = answer(question, session_id=session_id, expertise=expertise)

    try:
        append_turn(session_id, "user", question)
        append_turn(session_id, "assistant", result["answer"])
    except Exception:
        pass

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Streaming endpoint (SSE)
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {_json.dumps(data)}\n\n"


def _stream_answer(
    question: str,
    session_id: str,
    expertise: str,
) -> Generator[str, None, None]:
    """Sync generator yielding SSE events for /ask/stream.

    Events:
      {"type": "thinking", "text": "..."}   — progress indicator
      {"type": "delta",    "text": "..."}   — one token of the answer
      {"type": "error",    "text": "..."}   — unrecoverable error
      {"type": "done", "citations": [...], "confidence": "high"|"low",
                        "related_questions": [...]}
    """
    from agent.retriever_hybrid import search
    from agent.reranker import rerank
    from agent.verifier import is_grounded
    from agent.recommender import related as get_related
    from ingestion.format_router import detect_language
    from llm import stream_chat

    yield _sse({"type": "thinking", "text": "Searching knowledge base…"})

    try:
        chunks = search(question, k=settings.retrieve_top_k)
    except Exception:
        chunks = []

    if chunks:
        try:
            chunks = rerank(question, chunks)
        except Exception:
            pass

    if not chunks:
        yield _sse({
            "type": "done",
            "citations": [],
            "confidence": "low",
            "related_questions": [],
        })
        return

    yield _sse({"type": "thinking", "text": f"Found {len(chunks)} relevant passages. Composing answer…"})

    # Build numbered context + citation list
    context_lines: list[str] = []
    citations: list[dict] = []
    for i, c in enumerate(chunks, 1):
        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        label = ref.get("filename") or c.get("doc_id") or "unknown"
        page = ref.get("page", "")
        cite_label = f"{label} p.{page}" if page else label
        context_lines.append(f"[Source {i}] ({cite_label})\n{c['content']}")
        citations.append({
            "n": i,
            "id": c.get("id"),
            "doc_id": c.get("doc_id", ""),
            "filename": label,
            "page": page,
            "snippet": c["content"][:200],
        })

    expertise_note = (
        " Use plain, jargon-free language. Define technical terms when you use them."
        if expertise == "beginner"
        else " Be concise and technical."
    )
    lang = detect_language(question)
    lang_note = f" Answer in {lang}." if lang not in ("en", "unknown", "") else ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are TwoStrokeGPT, an expert on two-stroke engines. "
                "Answer ONLY using the numbered sources below. "
                "Cite each fact as [Source N] immediately after the claim. "
                "Never invent or estimate numeric values."
                + expertise_note
                + lang_note
                + "\n\nSources:\n"
                + "\n\n".join(context_lines)
            ),
        },
        {"role": "user", "content": question},
    ]

    full_answer = ""
    try:
        for token in stream_chat(messages, temperature=0.1, max_tokens=1024):
            full_answer += token
            yield _sse({"type": "delta", "text": token})
    except Exception as exc:
        yield _sse({"type": "error", "text": str(exc)})
        return

    # Grounding check
    try:
        grounded = is_grounded(full_answer, chunks)
    except Exception:
        grounded = False

    # Related questions
    try:
        related_qs = get_related(question, chunks)
    except Exception:
        related_qs = []

    # Persist conversation (best-effort)
    try:
        from memory.store import append_turn

        append_turn(session_id, "user", question)
        append_turn(session_id, "assistant", full_answer)
    except Exception:
        pass

    yield _sse({
        "type": "done",
        "citations": citations,
        "confidence": "high" if grounded else "low",
        "related_questions": related_qs,
    })


@app.post("/ask/stream")
def ask_stream(
    question: str = Form(...),
    session_id: str = Form("anon"),
    expertise: str = Form("expert"),
) -> StreamingResponse:
    """Streaming version of /ask using Server-Sent Events.

    The client consumes this with fetch() + ReadableStream (see index.html).
    Falls back gracefully to an empty done event when retrieval returns nothing.
    """
    return StreamingResponse(
        _stream_answer(question, session_id=session_id, expertise=expertise),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Feedback, gaps, graph
# ---------------------------------------------------------------------------

@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    vote: int = Form(0),
    correction: str = Form(""),
) -> JSONResponse:
    """Record user feedback. Corrections become high-priority knowledge; votes reweight retrieval."""
    from memory.store import record_feedback

    try:
        record_feedback(
            session_id=session_id,
            question=question,
            answer=answer,
            vote=vote,
            correction=correction,
        )
    except Exception:
        pass

    return JSONResponse({"status": "recorded"})


@app.get("/gaps")
def get_gaps() -> JSONResponse:
    """Return unresolved knowledge gaps logged by the verifier."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, question, reason, created_at
                FROM gaps
                WHERE resolved = false
                ORDER BY created_at DESC
                LIMIT 20
                """,
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        return JSONResponse({
            "gaps": [
                {"id": r[0], "question": r[1], "reason": r[2], "created_at": str(r[3])}
                for r in rows
            ]
        })
    except Exception:
        return JSONResponse({"gaps": [], "error": "db unavailable"})


@app.post("/gaps/{gap_id}/resolve")
async def resolve_gap(gap_id: int) -> JSONResponse:
    """Mark a knowledge gap as resolved."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            with conn.transaction():
                cur = conn.cursor()
                cur.execute(
                    "UPDATE gaps SET resolved = true WHERE id = %s",
                    (gap_id,),
                )
                updated = cur.rowcount
        finally:
            conn.close()

        if updated == 0:
            return JSONResponse({"error": "gap not found"}, status_code=404)
        return JSONResponse({"status": "resolved", "id": gap_id})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/entities")
def get_entities(doc_id: str | None = None) -> JSONResponse:
    """Return aggregated entities and tags extracted by the domain enricher.

    Queries the 'entities' and 'tags' arrays stored in chunks.metadata JSONB.
    Optional doc_id filter. Returns top-50 entities and top-30 tags by frequency.
    """
    from config import get_connection
    import json as _j

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            if doc_id:
                cur.execute(
                    "SELECT metadata FROM chunks WHERE doc_id = %s AND metadata ? 'entities'",
                    (doc_id,),
                )
            else:
                cur.execute("SELECT metadata FROM chunks WHERE metadata ? 'entities'")
            rows = cur.fetchall()
        finally:
            conn.close()

        entity_counts: dict[str, dict] = {}
        tag_counts: dict[str, int] = {}

        for (meta_raw,) in rows:
            meta = meta_raw if isinstance(meta_raw, dict) else _j.loads(meta_raw or "{}")
            for ent in meta.get("entities", []):
                key = f"{ent.get('type','?')}::{ent.get('name','?')}"
                if key not in entity_counts:
                    entity_counts[key] = {"type": ent.get("type"), "name": ent.get("name"), "count": 0}
                entity_counts[key]["count"] += 1
            for tag in meta.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        entities = sorted(entity_counts.values(), key=lambda e: e["count"], reverse=True)[:50]
        tags = sorted(tag_counts.items(), key=lambda t: t[1], reverse=True)[:30]

        return JSONResponse({
            "entities": entities,
            "tags": [{"tag": t, "count": c} for t, c in tags],
        })
    except Exception as exc:
        return JSONResponse({"entities": [], "tags": [], "error": str(exc)})


@app.get("/chunks")
def list_chunks(doc_id: str | None = None, limit: int = 50) -> JSONResponse:
    """List indexed chunks, optionally filtered by doc_id."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            if doc_id:
                cur.execute(
                    """
                    SELECT id, doc_id, content, metadata, source_refs
                    FROM chunks WHERE doc_id = %s
                    ORDER BY id LIMIT %s
                    """,
                    (doc_id, limit),
                )
            else:
                cur.execute(
                    "SELECT id, doc_id, content, metadata, source_refs FROM chunks ORDER BY id LIMIT %s",
                    (limit,),
                )
            rows = cur.fetchall()
        finally:
            conn.close()

        import json as _j
        return JSONResponse({
            "chunks": [
                {
                    "id": r[0],
                    "doc_id": r[1],
                    "snippet": r[2][:200],
                    "metadata": r[3] if isinstance(r[3], dict) else _j.loads(r[3] or "{}"),
                    "source_refs": r[4] if isinstance(r[4], list) else _j.loads(r[4] or "[]"),
                }
                for r in rows
            ]
        })
    except Exception as exc:
        return JSONResponse({"chunks": [], "error": str(exc)})


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: int) -> JSONResponse:
    """Return full content + metadata for a single chunk (evidence anchor)."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, doc_id, content, metadata, source_refs FROM chunks WHERE id = %s",
                (chunk_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return JSONResponse({"error": "chunk not found"}, status_code=404)

        import json as _j
        return JSONResponse({
            "id": row[0],
            "doc_id": row[1],
            "content": row[2],
            "metadata": row[3] if isinstance(row[3], dict) else _j.loads(row[3] or "{}"),
            "source_refs": row[4] if isinstance(row[4], list) else _j.loads(row[4] or "[]"),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/graph")
def get_graph() -> JSONResponse:
    """Return graph nodes and edges for knowledge-map display."""
    from config import get_connection

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, type, name FROM graph_nodes LIMIT 200")
            nodes = [{"id": r[0], "type": r[1], "name": r[2]} for r in cur.fetchall()]
            cur.execute(
                "SELECT src_id, dst_id, relation FROM graph_edges LIMIT 400"
            )
            edges = [{"source": r[0], "target": r[1], "relation": r[2]} for r in cur.fetchall()]
        finally:
            conn.close()

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception:
        return JSONResponse({"nodes": [], "edges": [], "error": "db unavailable"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.app_host, port=settings.app_port, reload=True)
