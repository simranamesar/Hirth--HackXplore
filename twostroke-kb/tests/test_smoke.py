"""Smoke tests — confirm the app imports and basic routing works before features land."""
from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_config_loads():
    from config import get_settings

    s = get_settings()
    assert s.embedding_dim > 0
    assert s.rerank_top_k <= s.retrieve_top_k


def test_format_router_known_ext():
    from ingestion.format_router import EXT_MAP

    assert EXT_MAP[".pdf"] == "pdf_or_ocr"
    assert EXT_MAP[".xlsx"] == "sheet"
    assert EXT_MAP[".url"] == "link"


def test_link_handler_extracts_url(tmp_path):
    from ingestion.parsers import link_handler

    f = tmp_path / "HYDAC International.url"
    f.write_text("[InternetShortcut]\nURL=https://www.hydac.com\n")
    doc = link_handler.parse(f)
    assert doc.metadata["url"] == "https://www.hydac.com"
    assert doc.metadata["fetched"] is False


def test_app_health():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"ok": True}


# --- Phase A: parse layer ---------------------------------------------------

def test_pdf_parser_extracts_text():
    from ingestion.parsers import pdf_parser

    doc = pdf_parser.parse(FIXTURES / "sample.pdf")
    assert "3503" in doc.text or "Hirth" in doc.text
    assert doc.metadata["type"] == "pdf"
    assert doc.metadata["pages"] >= 1
    assert doc.source_ref["filename"] == "sample.pdf"


def test_docx_parser_extracts_text_and_tables():
    from ingestion.parsers import docx_parser

    doc = docx_parser.parse(FIXTURES / "sample.docx")
    assert "Hirth" in doc.text or "manual" in doc.text.lower()
    assert doc.metadata["type"] == "docx"
    assert len(doc.tables) >= 1
    # first table should have a header row
    assert doc.tables[0].rows[0][0] != ""


def test_route_txt_returns_parseddoc(tmp_path):
    from ingestion.format_router import route

    f = tmp_path / "notes.txt"
    f.write_text("Hirth 3503 ignition timing spec.")
    doc = route(f)
    assert "Hirth" in doc.text
    assert doc.metadata["type"] == "text"
    assert "lang" in doc.metadata


def test_route_pdf_dispatches_to_pdf_parser():
    from ingestion.format_router import route

    doc = route(FIXTURES / "sample.pdf")
    assert doc.metadata["type"] == "pdf"
    assert "lang" in doc.metadata


def test_route_unsupported_raises():
    import pytest
    from ingestion.format_router import route

    with pytest.raises(ValueError):
        route(Path("file.xyz"))


# --- Phase B: normalize + chunk --------------------------------------------

def test_normalize_strips_boilerplate():
    from ingestion.corpus_builder import normalize
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(
        text="Real content here.\n--------\nMore real content.\n\n\n\nEnd.",
        metadata={"filename": "test.txt", "type": "text"},
        source_ref={"filename": "test.txt"},
    )
    result = normalize(doc)
    assert "--------" not in result.text
    assert result.text.count("\n\n\n") == 0
    assert result.metadata["filename"] == "test.txt"


def test_normalize_fills_missing_metadata():
    from ingestion.corpus_builder import normalize
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(text="Some text.", metadata={}, source_ref={})
    result = normalize(doc)
    assert result.metadata["filename"] == "unknown"
    assert result.metadata["type"] == "unknown"


def test_chunker_splits_prose():
    from ingestion.chunker import chunk
    from ingestion.types import ParsedDoc

    long_text = "word " * 500  # ~2500 chars
    doc = ParsedDoc(
        text=long_text,
        metadata={"filename": "test.txt", "type": "text", "lang": "en"},
        source_ref={"filename": "test.txt"},
    )
    chunks = chunk(doc, size=700, overlap=100)
    prose_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "prose"]
    assert len(prose_chunks) >= 3
    for c in prose_chunks:
        assert len(c["content"]) <= 700
        assert c["source_refs"] == [{"filename": "test.txt"}]


def test_chunker_preserves_tables():
    from ingestion.chunker import chunk
    from ingestion.types import ParsedDoc, Table

    doc = ParsedDoc(
        text="Short prose.",
        tables=[Table(name="specs", rows=[["Param", "Value"], ["RPM", "6500"]])],
        metadata={"filename": "manual.pdf", "type": "pdf", "lang": "en"},
        source_ref={"filename": "manual.pdf"},
    )
    chunks = chunk(doc)
    table_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
    assert len(table_chunks) == 1
    assert "RPM" in table_chunks[0]["content"]
    assert "6500" in table_chunks[0]["content"]


# --- Phase C: embed --------------------------------------------------------

def test_embed_returns_correct_dim():
    from ingestion.knowledge_base import embed
    from config import get_settings

    chunks = [
        {"content": "Hirth 3503 ignition timing.", "metadata": {}, "source_refs": []},
        {"content": "Kraftstoff-Übersicht für Zweitaktmotoren.", "metadata": {}, "source_refs": []},
    ]
    result = embed(chunks)
    dim = get_settings().embedding_dim
    assert all("embedding" in c for c in result)
    assert all(len(c["embedding"]) == dim for c in result)


def test_embed_empty_list():
    from ingestion.knowledge_base import embed

    assert embed([]) == []

