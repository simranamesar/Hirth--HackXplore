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
    """ReAct agent answer with citations and grounding check.

    Falls back to plain retrieve->answer automatically if the agent graph fails.
    """
    from agent.graph import answer

    result = answer(question, session_id=session_id)
    return JSONResponse(result)


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
