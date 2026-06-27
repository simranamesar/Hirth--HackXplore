# TwoStrokeGPT

AI-powered knowledge platform for two-stroke engines (Hirth Engines challenge). Upload technical documents in any format; ask questions in English or German and get **cited, self-checked** answers.

> For development conventions and architecture rules, see **CLAUDE.md**.

## Quick start

```bash
# 1. Python env + deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. System deps (NOT pip) — needed for some parsers
#    macOS (brew):   brew install tesseract tesseract-lang poppler libreoffice
#    Ubuntu (apt):   sudo apt install tesseract-ocr tesseract-ocr-deu poppler-utils libreoffice

# 3. Config
cp .env.example .env        # then fill in DATABASE_URL + an LLM key

# 4. Database (Postgres + pgvector via Docker)
docker compose up -d        # schema auto-applies on first run

# 5. Run
uvicorn api.main:app --reload
# open http://localhost:8000
```

Or use the shortcuts: `make setup`, `make db`, `make run`, `make test`.

## What works in the skeleton

- App boots, serves the upload + chat UI, `/health` passes.
- `format_router` extension map + `.url` link handler are implemented (with tests).
- Everything else is a typed stub with a `TODO` describing exactly what to build.

## Build order (see CLAUDE.md "MVP cut line")

1. Ingestion: route → parse (PDF/docx/txt) → chunk → embed → pgvector; plain cited answers.
2. OCR + spreadsheets (table-aware) + legacy `.doc`.
3. ReAct agent + grounding verifier (self-correction).
4. Re-ranker, then recommendations.
5. Feedback loop + dedup (merge provenance) + gap detector.
6. Graph view, beginner/expert mode.
7. Stretch: figures/diagrams, versioning, streaming.

## Architecture

Two LangGraph graphs — a format-aware ingestion pipeline and a ReAct diagnostic agent — over a single Postgres+pgvector store. Full design in `docs/ARCHITECTURE.md` (diagram) and `docs/PLAN.md` (rationale, X-factor, build order). All LLM calls go through `llm.py`.

## Non-negotiables

- **Never invent numbers.** Cite the source cell/clause; the verifier rejects unsupported numeric claims.
- **Dedup merges provenance**, never deletes a source.
- **Upload-only**: `.url` files are stored as links, not fetched.
