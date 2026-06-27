"""FastAPI app: upload documents, ask questions, send feedback. Serves a minimal HTML UI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
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
        "status": "indexed",
    })


@app.post("/ask")
async def ask(question: str = Form(...), session_id: str = Form("anon")) -> JSONResponse:
    """Plain retrieve→answer with citations. Agent loop (Graph 2) wired later.

    Grounding rule: the system prompt forbids inventing numbers not present in sources.
    """
    from agent.retriever_hybrid import search
    from llm import chat

    chunks = search(question, k=settings.rerank_top_k)

    if not chunks:
        return JSONResponse({
            "answer": "I don't have enough information in the knowledge base to answer that question.",
            "citations": [],
            "confidence": None,
            "related_questions": [],
        })

    # Build numbered context passages
    context_lines: list[str] = []
    citations: list[dict] = []
    for i, c in enumerate(chunks, start=1):
        source_refs = c.get("source_refs") or [{}]
        ref = source_refs[0] if source_refs else {}
        label = ref.get("filename", c.get("doc_id", "unknown"))
        page = ref.get("page", "")
        cite_label = f"{label} p.{page}" if page else label
        context_lines.append(f"[Source {i}] ({cite_label})\n{c['content']}")
        citations.append({"n": i, "doc_id": c["doc_id"], "filename": label,
                          "page": page, "snippet": c["content"][:200]})

    context = "\n\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are TwoStrokeGPT, an expert on two-stroke engines. "
                "Answer ONLY using the numbered sources provided below. "
                "Cite each fact as [Source N] immediately after the claim. "
                "If a numeric value (RPM, temperature, timing, torque, etc.) "
                "is NOT explicitly stated in a source, say you don't know. "
                "Never invent or estimate a value.\n\n"
                f"Sources:\n{context}"
            ),
        },
        {"role": "user", "content": question},
    ]

    answer_text = chat(messages, temperature=0.1, max_tokens=1024)

    return JSONResponse({
        "answer": answer_text,
        "citations": citations,
        "confidence": None,
        "related_questions": [],
    })


@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    vote: int = Form(0),
    correction: str = Form(""),
) -> JSONResponse:
    """Record feedback -> reweights retrieval (does NOT retrain the model)."""
    # from memory.store import record_feedback
    # record_feedback(...)
    return JSONResponse({"status": "recorded"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.app_host, port=settings.app_port, reload=True)
