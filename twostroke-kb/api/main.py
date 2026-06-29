"""FastAPI app: upload documents, ask questions, send feedback. Serves a minimal HTML UI."""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings

settings = get_settings()
app = FastAPI(title="TwoStrokeGPT")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def landing() -> FileResponse:
    """Serve the landing page."""
    return FileResponse(STATIC_DIR / "landing.html")


@app.get("/app")
def app_page() -> FileResponse:
    """Serve the main application UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/inventory/scan")
async def inventory_scan(
    root_path: str = Form(...),
    max_files: int = Form(50000),
) -> JSONResponse:
    """Metadata-only scan of a local corpus folder.

    This does not parse, chunk, embed, or run KG extraction. It only catalogs
    file metadata so large corpora can be filtered before selective ingestion.
    """
    import asyncio

    try:
        from ingestion.inventory import scan_inventory

        result = await asyncio.to_thread(scan_inventory, root_path, max_files)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/inventory")
def inventory_list(
    batch_id: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> JSONResponse:
    """List inventory rows from the metadata catalog."""
    try:
        from ingestion.inventory import list_inventory

        return JSONResponse({
            "items": list_inventory(batch_id=batch_id, limit=limit, offset=offset),
            "batch_id": batch_id,
            "limit": limit,
            "offset": offset,
        })
    except Exception as exc:
        return JSONResponse({"items": [], "error": str(exc)})


@app.get("/inventory/summary")
def inventory_get_summary(batch_id: str | None = None) -> JSONResponse:
    """Return inventory rollups by topic, category, extension, and status."""
    try:
        from ingestion.inventory import inventory_summary

        return JSONResponse(inventory_summary(batch_id=batch_id))
    except Exception as exc:
        return JSONResponse({
            "batch_id": batch_id,
            "total_files": 0,
            "total_size_bytes": 0,
            "by_topic": [],
            "by_category": [],
            "by_extension": [],
            "by_status": [],
            "error": str(exc),
        })


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_ids(value: str | None) -> list[int]:
    ids: list[int] = []
    for part in _split_csv(value):
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


@app.post("/inventory/ingest-selected")
async def inventory_ingest_selected(
    topic: str = Form(""),
    extensions: str = Form(""),
    inventory_ids: str = Form(""),
    max_files: int = Form(25),
    max_file_size_mb: int = Form(50),
    skip_existing: bool = Form(True),
    kg_enabled: bool = Form(False),
    kg_max_chunks_per_doc: int = Form(20),
    dry_run: bool = Form(True),
) -> JSONResponse:
    """Dry-run or ingest a controlled selection from the metadata inventory."""
    import asyncio

    try:
        from ingestion.inventory import dry_run_selected, ingest_selected

        kwargs = {
            "topic": topic.strip() or None,
            "extensions": _split_csv(extensions),
            "inventory_ids": _split_ids(inventory_ids),
            "max_files": max(1, min(max_files, 500)),
            "max_file_size_mb": max(1, min(max_file_size_mb, 2048)),
            "skip_existing": skip_existing,
        }
        if dry_run:
            result = await asyncio.to_thread(dry_run_selected, **kwargs)
        else:
            result = await asyncio.to_thread(
                ingest_selected,
                **kwargs,
                kg_enabled=kg_enabled,
                kg_max_chunks_per_doc=max(0, min(kg_max_chunks_per_doc, 100)),
            )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/inventory/jobs/{job_id}")
def inventory_job_status(job_id: str) -> JSONResponse:
    """Return persisted progress for a selective inventory ingestion job."""
    try:
        from ingestion.inventory import get_ingestion_job

        return JSONResponse(get_ingestion_job(job_id))
    except Exception as exc:
        return JSONResponse({"job_id": job_id, "status": "error", "error": str(exc)}, status_code=500)


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """Save the upload and run the ingestion pipeline (non-streaming fallback)."""
    import asyncio
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(await file.read())

    from ingestion.orchestrator import run_ingestion

    # Run blocking ingestion in a thread so we don't block the event loop
    result = await asyncio.to_thread(run_ingestion, dest)
    return JSONResponse({
        "filename": result.filename,
        "chunks": result.chunks,
        "facts": result.facts,
        "skipped_duplicates": result.skipped_duplicates,
        "version": result.version,
        "status": "indexed",
    })


@app.post("/upload/stream")
async def upload_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a file and stream ingestion progress as SSE events.

    Events:
      {"type": "stage",  "text": "Parsing…",  "pct": 10}
      {"type": "stage",  "text": "Chunking…", "pct": 40}
      {"type": "done",   "filename": …, "chunks": …, "facts": …, …}
      {"type": "error",  "text": "…"}
    """
    import asyncio

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(await file.read())

    async def _generate():
        import queue, threading

        q: queue.Queue = queue.Queue()

        def _run():
            try:
                # Monkey-patch orchestrator to send progress via queue
                from ingestion import format_router, corpus_builder, chunker as chunker_mod
                from ingestion import domain_enricher, dedup as dedup_mod, knowledge_base, graph_builder
                from ingestion.orchestrator import _slug, _register_document, IngestResult
                from pathlib import Path as _Path
                import re as _re, logging as _log

                p = _Path(dest)
                q.put({"type": "stage", "text": "Parsing document…", "pct": 10})
                doc = format_router.route(p)

                version = 1
                try:
                    doc_id = _slug(p.name)
                    lang = doc.metadata.get("lang", "unknown")
                    version = _register_document(doc_id, p.name, lang, storage_uri=str(p))
                except Exception:
                    pass

                q.put({"type": "stage", "text": "Normalising text…", "pct": 22})
                clean = corpus_builder.normalize(doc)

                q.put({"type": "stage", "text": "Chunking…", "pct": 35})
                chunks = chunker_mod.chunk(clean)

                q.put({"type": "stage", "text": f"Enriching {len(chunks)} chunks…", "pct": 50})
                try:
                    chunks = domain_enricher.enrich(chunks)
                except Exception:
                    pass

                q.put({"type": "stage", "text": "Embedding…", "pct": 65})
                chunks = knowledge_base.embed(chunks)

                q.put({"type": "stage", "text": "Deduplicating…", "pct": 78})
                before = len(chunks)
                try:
                    chunks = dedup_mod.dedup_and_merge(chunks)
                except Exception:
                    pass
                skipped = before - len(chunks)

                q.put({"type": "stage", "text": f"Storing {len(chunks)} chunks…", "pct": 88})
                knowledge_base.store(chunks)

                q.put({"type": "stage", "text": "Building knowledge graph…", "pct": 95})
                try:
                    graph_builder.extract(clean, chunks=chunks)
                except Exception:
                    pass

                fact_count = sum(1 for c in chunks if c["metadata"].get("chunk_type") == "table")
                q.put({"type": "done", "filename": p.name, "chunks": len(chunks),
                       "facts": fact_count, "skipped_duplicates": skipped,
                       "version": version, "status": "indexed"})
            except Exception as exc:
                q.put({"type": "error", "text": str(exc)})

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=300)
                )
                yield f"data: {_json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except Exception as exc:
                yield f"data: {_json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
                break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form("anon"),
) -> JSONResponse:
    """ReAct agent answer with citations, grounding check, and related questions.

    Falls back to plain retrieve->answer automatically if the agent graph fails.
    """
    from agent.graph import answer
    from memory.store import append_turn

    result = answer(question, session_id=session_id)

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
    topic: str | None = None,
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
    from agent.kg_retrieval import retrieve_kg_context
    from ingestion.format_router import detect_language
    from llm import stream_chat

    yield _sse({"type": "thinking", "text": "Searching knowledge base…"})

    # Load prior conversation for context-aware answers
    history_note = ""
    try:
        from memory.store import get_conversation
        turns = get_conversation(session_id)
        if turns:
            recent = turns[-4:]
            history_parts = [
                f"{t.get('role','').capitalize()}: {str(t.get('content',''))[:300]}"
                for t in recent
            ]
            history_note = "\n\nPrior conversation:\n" + "\n".join(history_parts)
    except Exception:
        pass

    # Use question + history for retrieval so follow-ups find relevant chunks
    search_query = question + (" " + history_note if history_note else "")

    try:
        kg_result = retrieve_kg_context(question)
    except Exception:
        kg_result = {"intent": {"intent": "general_question"}, "paths": [], "graph_evidence": [], "context": ""}

    try:
        chunks = search(search_query, k=settings.retrieve_top_k, topic=(topic or None))
    except Exception:
        chunks = []

    if chunks:
        try:
            chunks = rerank(question, chunks)
        except Exception:
            pass

    if not chunks:
        yield _sse({"type": "delta", "text": "I cannot find information about this in the uploaded documents. Please upload relevant documents first, or rephrase your question."})
        yield _sse({
            "type": "done",
            "citations": [],
            "confidence": "low",
            "related_questions": [],
            "kg_paths": kg_result.get("paths", []),
            "graph_evidence": kg_result.get("graph_evidence", []),
            "intent": kg_result.get("intent", {}),
        })
        return

    yield _sse({"type": "thinking", "text": f"Found {len(chunks)} relevant passages. Composing answer…"})

    # Build numbered context + citation list
    context_lines: list[str] = []
    citations: list[dict] = []
    for i, c in enumerate(chunks, 1):
        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        metadata = c.get("metadata") or {}
        label = ref.get("filename") or c.get("doc_id") or "unknown"
        page = ref.get("page", "")
        slide = ref.get("slide") or metadata.get("slide")
        sheet = ref.get("sheet") or metadata.get("sheet") or metadata.get("table_name")
        source_topic = metadata.get("topic") or ref.get("topic")
        location = f"p.{page}" if page else f"slide {slide}" if slide else f"sheet {sheet}" if sheet else ""
        cite_label = f"{label} {location}".strip()
        if source_topic:
            cite_label = f"{source_topic} / {cite_label}"
        context_lines.append(f"[Source {i}] ({cite_label})\n{c['content']}")
        citations.append({
            "n": i,
            "id": c.get("id"),
            "doc_id": c.get("doc_id", ""),
            "filename": label,
            "page": page,
            "slide": slide,
            "sheet": sheet,
            "topic": source_topic,
            "relative_path": metadata.get("relative_path") or ref.get("relative_path"),
            "source_title": metadata.get("source_title") or ref.get("source_title") or label,
            "snippet": c["content"][:200],
        })

    expertise_note = " Be concise and technical."
    lang = detect_language(question)
    lang_note = f" Answer in {lang}." if lang not in ("en", "unknown", "") else ""
    kg_context = kg_result.get("context", "")
    kg_note = (
        "\n\nUse this Knowledge Graph evidence to structure diagnostic reasoning when relevant. "
        "Do not cite KG paths as [Source N]; cite numbered document sources for factual claims. "
        "Mention graph-backed paths only when evidence is shown.\n\n"
        + kg_context
        if kg_context
        else ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are TwoStrokeGPT, a document-grounded assistant for two-stroke engine manuals.\n\n"
                "STRICT RULES — follow them exactly:\n"
                "1. Answer ONLY using the numbered [Source N] documents provided below. "
                "DO NOT use your training knowledge under any circumstances.\n"
                "2. Every factual claim MUST be followed by its [Source N] citation immediately.\n"
                "3. If the answer to the question is not explicitly stated in any source, "
                "respond with: 'I cannot find information about this in the uploaded documents.' "
                "Do NOT guess, infer, or fill in from memory.\n"
                "4. NEVER invent, estimate, or approximate any numeric value "
                "(RPM, temperature, torque, timing, pressure, gap, ratio, etc.). "
                "Only state numbers that appear word-for-word in a source.\n"
                "5. Write a complete, well-structured answer. Never end mid-sentence."
                + expertise_note
                + lang_note
                + ("\n\n" + history_note.strip() if history_note else "")
                + "\n\nSOURCES:\n"
                + "\n\n".join(context_lines)
                + kg_note
            ),
        },
        {"role": "user", "content": question},
    ]

    full_answer = ""
    try:
        for token in stream_chat(messages, temperature=0.1):
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

    # Related questions — pass full question with history for better suggestions
    try:
        related_qs = get_related(question + (history_note or ""), chunks)
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
        "kg_paths": kg_result.get("paths", []),
        "graph_evidence": kg_result.get("graph_evidence", []),
        "intent": kg_result.get("intent", {}),
    })


@app.post("/ask/stream")
def ask_stream(
    question: str = Form(...),
    session_id: str = Form("anon"),
    topic: str = Form(""),
) -> StreamingResponse:
    """Streaming version of /ask using Server-Sent Events.

    The client consumes this with fetch() + ReadableStream (see index.html).
    Falls back gracefully to an empty done event when retrieval returns nothing.
    """
    return StreamingResponse(
        _stream_answer(question, session_id=session_id, topic=topic.strip() or None),
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


@app.get("/chunks/{chunk_id}/view")
def view_chunk_html(chunk_id: int):
    """Render a standalone HTML page for a single chunk (evidence anchor, opens in new tab)."""
    from fastapi.responses import HTMLResponse
    from config import get_connection
    import json as _j

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
            return HTMLResponse("<h2>Chunk not found</h2>", status_code=404)

        meta = row[3] if isinstance(row[3], dict) else _j.loads(row[3] or "{}")
        refs = row[4] if isinstance(row[4], list) else _j.loads(row[4] or "[]")
        ref_str = ", ".join(r.get("filename") or r.get("source") or str(r) for r in refs) or "—"
        content_escaped = row[2].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>Chunk {row[0]} — {row[1]}</title>
<style>
  body{{font-family:ui-sans-serif,system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 24px;color:#1c1917;background:#fafaf9}}
  h2{{font-size:18px;margin-bottom:4px}}
  .meta{{font-size:12px;color:#78716c;margin-bottom:20px}}
  pre{{background:#f5f5f4;border-radius:8px;padding:16px;white-space:pre-wrap;word-break:break-word;line-height:1.6;font-size:14px}}
  a,button.close{{color:#b45309;cursor:pointer;background:none;border:none;font-size:14px;padding:0;text-decoration:underline}}
  .notice{{font-size:12px;color:#78716c;margin-top:4px}}
</style></head><body>
<h2>Chunk #{row[0]} — <code>{row[1]}</code></h2>
<div class="meta">
  type: {meta.get("chunk_type", meta.get("type","?"))} &nbsp;|&nbsp;
  lang: {meta.get("lang","?")} &nbsp;|&nbsp;
  page: {meta.get("page", "?")} &nbsp;|&nbsp;
  sources: {ref_str}
</div>
<pre>{content_escaped}</pre>
<p>
  <button class="close" onclick="window.close()">Close this tab</button>
  <span class="notice">&nbsp;— opened by TwoStrokeGPT</span>
</p>
</body></html>"""
        return HTMLResponse(html)
    except Exception as exc:
        return HTMLResponse(f"<h2>Error: {exc}</h2>", status_code=500)


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


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _listify(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def serialize_graph_node(node_id: int, ntype: str, name: str, props_raw: Any) -> dict[str, Any] | None:
    """Serialize a graph node with backward-compatible fields plus rich metadata."""
    from ingestion.kg_normalizer import normalize_entity

    normalized = normalize_entity(name, ntype)
    if not normalized["is_valid"]:
        return None
    props = _json_obj(props_raw)
    aliases = props.get("aliases") or normalized.get("aliases") or []
    doc_ids = _listify(props.get("doc_ids")) or _listify(props.get("doc_id"))
    confidence = _confidence(props.get("confidence"))
    label = props.get("display_name") or normalized["display_name"] or normalized["canonical_name"]
    return {
        "id": node_id,
        "label": label,
        "name": label,  # Backward compatibility for current frontend.
        "type": normalized["type"],
        "canonical_name": normalized["canonical_name"],
        "aliases": aliases,
        "confidence": confidence,
        "source_count": int(props.get("source_count") or len(set(map(str, doc_ids))) or 0),
        "doc_ids": doc_ids,
        "props": props,
    }


def serialize_graph_edge(edge_id: int, src_id: int, dst_id: int, relation: str, props_raw: Any) -> dict[str, Any]:
    """Serialize a graph edge with provenance defaults for old rows."""
    props = _json_obj(props_raw)
    confidence = _confidence(props.get("confidence"))
    extraction_method = props.get("extraction_method") or "unknown"
    evidence = str(props.get("evidence") or "")
    chunk_id = props.get("source_chunk_id", props.get("chunk_id"))
    topic = props.get("topic")
    relative_path = props.get("relative_path")
    file_type = props.get("file_type")
    return {
        "id": edge_id,
        "source": src_id,
        "target": dst_id,
        "relation": relation,  # Backward compatibility for current frontend.
        "type": relation,
        "confidence": confidence,
        "evidence": evidence,
        "extraction_method": extraction_method,
        "doc_id": props.get("doc_id"),
        "source_chunk_id": chunk_id,
        "chunk_id": chunk_id,
        "page": props.get("page"),
        "source_title": props.get("source_title"),
        "topic": topic,
        "relative_path": relative_path,
        "file_type": file_type,
        "props": {
            "doc_id": props.get("doc_id"),
            "source_chunk_id": chunk_id,
            "chunk_id": chunk_id,
            "page": props.get("page"),
            "evidence": evidence,
            "confidence": confidence,
            "extraction_method": extraction_method,
            "source_title": props.get("source_title"),
            "topic": topic,
            "relative_path": relative_path,
            "file_type": file_type,
            **props,
        },
    }


@app.get("/graph")
def get_graph(
    node_type: str | None = None,
    edge_type: str | None = None,
    doc_id: str | None = None,
    topic: str | None = None,
    min_confidence: float = 0.4,
    extraction_method: str | None = None,
    include_seed: bool = True,
    limit: int = 600,
) -> JSONResponse:
    """Return graph nodes and edges for knowledge-map display."""
    from config import get_connection

    try:
        limit = max(1, min(int(limit), 2000))
        min_confidence = max(0.0, min(1.0, float(min_confidence)))
        topic = (topic or "").strip() or None
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, type, name, props FROM graph_nodes LIMIT %s", (limit,))
            nodes = []
            valid_ids = set()
            all_node_ids = set()
            for row in cur.fetchall():
                node = serialize_graph_node(*row)
                if not node:
                    continue
                all_node_ids.add(node["id"])
                if node_type and node["type"] != node_type:
                    continue
                if doc_id and doc_id not in {str(d) for d in node["doc_ids"]}:
                    continue
                nodes.append(node)
                valid_ids.add(node["id"])
            cur.execute(
                "SELECT id, src_id, dst_id, relation, props FROM graph_edges LIMIT %s",
                (limit,),
            )
            edges = []
            edge_node_ids = set()
            for row in cur.fetchall():
                edge = serialize_graph_edge(*row)
                if edge["source"] not in all_node_ids or edge["target"] not in all_node_ids:
                    continue
                if edge_type and edge["relation"] != edge_type:
                    continue
                if doc_id and edge.get("doc_id") != doc_id:
                    continue
                if topic and edge.get("topic") != topic:
                    continue
                if extraction_method and edge["extraction_method"] != extraction_method:
                    continue
                if not include_seed and edge["extraction_method"] == "manual_seed":
                    continue
                if (
                    edge["confidence"] is not None
                    and edge["confidence"] < min_confidence
                    and edge["extraction_method"] != "manual_seed"
                ):
                    continue
                edge_node_ids.update([edge["source"], edge["target"]])
                edges.append(edge)
            if node_type or doc_id:
                edges = [e for e in edges if e["source"] in valid_ids and e["target"] in valid_ids]
                edge_node_ids = {nid for e in edges for nid in (e["source"], e["target"])}
            if topic:
                valid_ids = edge_node_ids
            else:
                valid_ids = edge_node_ids or valid_ids
            if topic:
                nodes = [n for n in nodes if n["id"] in valid_ids]
            else:
                nodes = [n for n in nodes if n["id"] in valid_ids or not edges]
        finally:
            conn.close()

        return JSONResponse({
            "nodes": nodes,
            "edges": edges,
            "filters": {
                "node_type": node_type,
                "edge_type": edge_type,
                "doc_id": doc_id,
                "topic": topic,
                "min_confidence": min_confidence,
                "extraction_method": extraction_method,
                "include_seed": include_seed,
                "limit": limit,
            },
        })
    except Exception:
        return JSONResponse({"nodes": [], "edges": [], "error": "db unavailable"})


@app.get("/graph/diagnostic-paths")
def graph_diagnostic_paths(query: str, limit: int = 5) -> JSONResponse:
    """Return KG diagnostic paths for a query."""
    try:
        from agent.kg_retrieval import retrieve_kg_context

        result = retrieve_kg_context(query, limit=max(1, min(limit, 20)))
        return JSONResponse({
            "query": query,
            "intent": result.get("intent", {}),
            "paths": result.get("paths", []),
            "context": result.get("context", ""),
        })
    except Exception as exc:
        return JSONResponse({"query": query, "paths": [], "error": str(exc)}, status_code=500)


@app.get("/graph/quality")
def graph_quality() -> JSONResponse:
    """Return KG quality metrics: node/edge counts, confidence, evidence coverage."""
    from config import get_connection
    import json as _j

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM graph_nodes")
            total_nodes = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM graph_edges")
            total_edges = cur.fetchone()[0]

            cur.execute("SELECT type, COUNT(*) FROM graph_nodes GROUP BY type ORDER BY COUNT(*) DESC")
            nodes_by_type = {r[0]: r[1] for r in cur.fetchall()}

            cur.execute("SELECT relation, COUNT(*) FROM graph_edges GROUP BY relation ORDER BY COUNT(*) DESC")
            edges_by_relation = {r[0]: r[1] for r in cur.fetchall()}

            # Props-based stats — safe against NULL props
            cur.execute("SELECT props FROM graph_edges WHERE props IS NOT NULL")
            edge_props_rows = cur.fetchall()

            method_counts: dict = {}
            confidences: list[float] = []
            with_evidence = 0
            with_doc_id = 0
            with_chunk_id = 0

            for (props_raw,) in edge_props_rows:
                props = props_raw if isinstance(props_raw, dict) else _j.loads(props_raw or "{}")
                method = str(props.get("extraction_method") or "unknown")
                method_counts[method] = method_counts.get(method, 0) + 1
                conf = props.get("confidence")
                try:
                    confidences.append(float(conf))
                except (TypeError, ValueError):
                    pass
                if props.get("evidence"):
                    with_evidence += 1
                if props.get("doc_id"):
                    with_doc_id += 1
                if props.get("source_chunk_id") or props.get("chunk_id"):
                    with_chunk_id += 1

            avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
            edge_total = len(edge_props_rows)
            evidence_pct  = round(with_evidence / edge_total * 100, 1) if edge_total else 0
            doc_id_pct    = round(with_doc_id   / edge_total * 100, 1) if edge_total else 0
            chunk_id_pct  = round(with_chunk_id / edge_total * 100, 1) if edge_total else 0

            # Top connected nodes
            cur.execute("""
                SELECT n.name, n.type, COUNT(*) AS degree
                FROM graph_nodes n
                JOIN graph_edges e ON e.src_id = n.id OR e.dst_id = n.id
                GROUP BY n.id, n.name, n.type
                ORDER BY degree DESC LIMIT 10
            """)
            top_connected = [{"name": r[0], "type": r[1], "degree": r[2]} for r in cur.fetchall()]

            # Unknown / noisy nodes
            cur.execute("SELECT name FROM graph_nodes WHERE type = 'unknown' LIMIT 10")
            noisy_unknown = [r[0] for r in cur.fetchall()]

        finally:
            conn.close()

        return JSONResponse({
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "nodes_by_type": nodes_by_type,
            "edges_by_relation": edges_by_relation,
            "edges_by_extraction_method": method_counts,
            "avg_confidence": avg_confidence,
            "evidence_coverage_pct": evidence_pct,
            "doc_id_coverage_pct": doc_id_pct,
            "chunk_id_coverage_pct": chunk_id_pct,
            "top_connected_nodes": top_connected,
            "noisy_unknown_nodes": noisy_unknown,
        })
    except Exception as exc:
        return JSONResponse({
            "total_nodes": 0, "total_edges": 0,
            "nodes_by_type": {}, "edges_by_relation": {},
            "edges_by_extraction_method": {}, "avg_confidence": None,
            "evidence_coverage_pct": 0, "doc_id_coverage_pct": 0,
            "chunk_id_coverage_pct": 0, "top_connected_nodes": [],
            "noisy_unknown_nodes": [], "error": str(exc),
        })


@app.get("/topics")
def topics() -> JSONResponse:
    """Return known corpus topics from indexed chunks, inventory, and demo presets."""
    from config import get_connection

    preset_topics = [
        "CAD",
        "Verbrennungsmotoren",
        "Oberflächenbehandlung",
        "Elektrotechnik",
        "Aluminiumguss",
        "Propeller",
        "Konstruktionslehre",
        "Werkstoffkunde",
        "Luftfahrt",
        "Sonst. Stoffe",
        "Normen DIN ISO VDI FAR ASTM LURS",
        "Relevante Hirth-Information _ alt",
        "Vorlagen Testprotokolle",
        "Bauteilsicherheit und -zuverlaellisgkeit",
        "Schulungen",
        "Bachelor_Master_Diplom_Doktorarbeiten",
        "Vibrationen",
        "Drehmomente",
        "Feinstellung-Zweitaktmotor",
    ]
    topic_map = {name: {"topic": name, "chunks": 0, "inventory_files": 0, "size_bytes": 0} for name in preset_topics}

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT metadata->>'topic' AS topic, COUNT(*)
                FROM chunks
                WHERE metadata ? 'topic' AND COALESCE(metadata->>'topic', '') <> ''
                GROUP BY topic
                """
            )
            for topic_name, count in cur.fetchall():
                row = topic_map.setdefault(topic_name, {"topic": topic_name, "chunks": 0, "inventory_files": 0, "size_bytes": 0})
                row["chunks"] = int(count or 0)

            cur.execute(
                """
                SELECT topic, COUNT(*), COALESCE(SUM(size_bytes), 0)
                FROM file_inventory
                WHERE COALESCE(topic, '') <> ''
                GROUP BY topic
                """
            )
            for topic_name, count, size_bytes in cur.fetchall():
                row = topic_map.setdefault(topic_name, {"topic": topic_name, "chunks": 0, "inventory_files": 0, "size_bytes": 0})
                row["inventory_files"] = int(count or 0)
                row["size_bytes"] = int(size_bytes or 0)
        finally:
            conn.close()
    except Exception:
        pass

    presets = [
        {"label": "Engine troubleshooting", "topic": "Verbrennungsmotoren", "question": "What are the likely causes and fixes for engine misfire?"},
        {"label": "Torque specs", "topic": "Drehmomente", "question": "What torque specifications are available?"},
        {"label": "Vibration", "topic": "Vibrationen", "question": "What vibration issues and checks are documented?"},
        {"label": "Two-stroke fine tuning", "topic": "Feinstellung-Zweitaktmotor", "question": "What should I check for two-stroke fine tuning?"},
        {"label": "Standards/certification", "topic": "Normen DIN ISO VDI FAR ASTM LURS", "question": "Which standards or certification requirements are referenced?"},
        {"label": "Propeller", "topic": "Propeller", "question": "What propeller-related guidance is available?"},
    ]
    rows = sorted(topic_map.values(), key=lambda r: r["topic"].casefold())
    return JSONResponse({"topics": rows, "presets": presets})


@app.get("/search")
def semantic_search(q: str, limit: int = 10, topic: str | None = None) -> JSONResponse:
    """Hybrid BM25 + dense semantic search over indexed chunks."""
    import json as _j
    from config import get_connection

    if not q or not q.strip():
        return JSONResponse({"results": [], "error": "empty query"})

    topic = (topic or "").strip() or None
    try:
        from agent.retriever_hybrid import search as hybrid_search

        hits = hybrid_search(q.strip(), k=limit, topic=topic)
        results = []
        for h in hits:
            metadata = h.get("metadata", {}) or {}
            refs = h.get("source_refs") or [{}]
            ref = refs[0] if refs else {}
            results.append({
                "id":       h.get("id"),
                "doc_id":   h.get("doc_id", ""),
                "score":    round(float(h.get("score", 0)), 4),
                "snippet":  (h.get("content") or h.get("snippet") or "")[:300],
                "metadata": metadata,
                "filename": h.get("filename") or ref.get("filename", ""),
                "topic": metadata.get("topic") or ref.get("topic"),
                "source_title": metadata.get("source_title") or ref.get("source_title") or ref.get("filename", ""),
                "page": ref.get("page") or metadata.get("page"),
                "slide": ref.get("slide") or metadata.get("slide"),
                "sheet": ref.get("sheet") or metadata.get("sheet") or metadata.get("table_name"),
            })
        return JSONResponse({"results": results, "topic": topic})
    except Exception as exc:
        # Fallback: plain SQL full-text search
        try:
            conn = get_connection()
            try:
                cur = conn.cursor()
                where_topic = "AND metadata->>'topic' = %s" if topic else ""
                params = [q, q]
                if topic:
                    params.append(topic)
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT id, doc_id, content, metadata, source_refs,
                           ts_rank_cd(to_tsvector('simple', content),
                                      plainto_tsquery('simple', %s)) AS rank
                    FROM chunks
                    WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
                    {where_topic}
                    ORDER BY rank DESC LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
            results = []
            for r in rows:
                meta = r[3] if isinstance(r[3], dict) else _j.loads(r[3] or "{}")
                refs = r[4] if isinstance(r[4], list) else _j.loads(r[4] or "[]")
                ref = refs[0] if refs else {}
                filename = (ref.get("filename") or ref.get("source") or "") if refs else ""
                results.append({
                    "id":       r[0],
                    "doc_id":   r[1],
                    "score":    round(float(r[5]), 4),
                    "snippet":  r[2][:300],
                    "metadata": meta,
                    "filename": filename,
                    "topic": meta.get("topic") or ref.get("topic"),
                    "source_title": meta.get("source_title") or ref.get("source_title") or filename,
                    "page": ref.get("page") or meta.get("page"),
                    "slide": ref.get("slide") or meta.get("slide"),
                    "sheet": ref.get("sheet") or meta.get("sheet") or meta.get("table_name"),
                })
            return JSONResponse({"results": results, "mode": "fulltext_fallback", "topic": topic})
        except Exception as exc2:
            return JSONResponse({"results": [], "error": str(exc2)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.app_host, port=settings.app_port, reload=True)
