"""Smoke tests — confirm the app imports and basic routing works before features land."""
from __future__ import annotations

from pathlib import Path

import pytest

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


# --- Slice 2: sheet_parser, ocr, doc_convert, spec_lookup -------------------

def test_sheet_parser_csv(tmp_path):
    from ingestion.parsers import sheet_parser

    f = tmp_path / "specs.csv"
    f.write_text("Parameter,Value [rpm],Torque [Nm]\nIdle,800,5.2\nMax,6500,12.8\n")
    doc = sheet_parser.parse(f)
    assert doc.metadata["type"] == "sheet"
    assert doc.metadata["filename"] == "specs.csv"
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.rows[0] == ["Parameter", "Value [rpm]", "Torque [Nm]"]
    assert table.units == {"Value [rpm]": "rpm", "Torque [Nm]": "Nm"}
    assert "800" in table.rows[1]
    assert "Parameter" in doc.text  # column names appear in prose summary


def test_sheet_parser_xlsx():
    from ingestion.parsers import sheet_parser

    doc = sheet_parser.parse(FIXTURES / "sample.xlsx")
    assert doc.metadata["type"] == "sheet"
    assert doc.metadata["filename"] == "sample.xlsx"
    assert len(doc.tables) >= 1
    table = doc.tables[0]
    assert len(table.rows) >= 2        # header + at least one data row
    assert table.units                 # bracket-notation units detected
    assert "6500" in str(table.rows)   # Max Power row present


def test_sheet_parser_units_flow_into_chunker():
    """Units stored in Table.units must appear in the chunk's Units: line."""
    from ingestion.parsers import sheet_parser
    from ingestion.chunker import chunk

    doc = sheet_parser.parse(FIXTURES / "sample.xlsx")
    chunks = chunk(doc)
    table_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
    assert table_chunks, "expected at least one table chunk"
    assert "Units:" in table_chunks[0]["content"]


def test_ocr_png():
    import shutil
    import pytest

    if not shutil.which("tesseract"):
        pytest.skip("tesseract not installed")

    from ingestion.parsers import ocr

    doc = ocr.parse(FIXTURES / "sample.png")
    assert doc.metadata["type"] == "ocr"
    assert doc.metadata["pages"] == 1
    assert isinstance(doc.metadata["low_confidence_pages"], list)
    assert doc.source_ref["filename"] == "sample.png"
    assert doc.source_ref["page_count"] == 1


def test_doc_convert_no_system_dep(tmp_path):
    """When neither libreoffice nor antiword is on PATH, raise RuntimeError."""
    import shutil
    import pytest
    from unittest.mock import patch

    if shutil.which("libreoffice") or shutil.which("antiword"):
        pytest.skip("system dep present; skipping no-dep error path")

    stub = tmp_path / "legacy.doc"
    stub.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")  # OLE2 magic bytes

    from ingestion.parsers import doc_convert

    with pytest.raises(RuntimeError, match="libreoffice"):
        doc_convert.parse(stub)


# --- Slice 3: reranker, verifier, graph -------------------------------------

def test_reranker_sorts_and_truncates():
    """Cross-encoder scores must sort results descending, capped at RERANK_TOP_K."""
    from unittest.mock import patch, MagicMock
    from config import get_settings

    candidates = [
        {"content": "Low relevance text", "doc_id": "a"},
        {"content": "Highly relevant RPM spec", "doc_id": "b"},
        {"content": "Medium relevance content", "doc_id": "c"},
        {"content": "Another low relevance doc", "doc_id": "d"},
        {"content": "Top match for the query", "doc_id": "e"},
    ]
    mock_scores = [0.1, 0.85, 0.5, 0.05, 0.95]
    mock_ce = MagicMock()
    mock_ce.predict.return_value = mock_scores

    with patch("agent.reranker._model", return_value=mock_ce):
        from agent.reranker import rerank
        results = rerank("RPM specification", candidates)

    top_k = get_settings().rerank_top_k
    assert len(results) <= top_k
    assert results[0]["doc_id"] == "e"   # score 0.95
    assert results[1]["doc_id"] == "b"   # score 0.85
    assert "rerank_score" in results[0]
    assert results[0]["rerank_score"] == pytest.approx(0.95, abs=1e-6)


def test_reranker_empty_input():
    from agent.reranker import rerank
    assert rerank("anything", []) == []


def test_verifier_grounded_no_numbers():
    """A draft with no numeric claims is always grounded."""
    from agent.verifier import is_grounded
    assert is_grounded("I don't have enough information to answer.", [])


def test_verifier_grounded_true():
    from agent.verifier import is_grounded
    ctx = [{"content": "The Hirth 3503 operates at a maximum of 6500 rpm."}]
    assert is_grounded("The engine reaches 6500 rpm at full power [Source 1].", ctx)


def test_verifier_grounded_false():
    from agent.verifier import is_grounded
    ctx = [{"content": "The engine produces 12.8 Nm of torque."}]
    # 7000 does not appear anywhere in the source
    assert not is_grounded("The engine produces 7000 rpm [Source 1].", ctx)


def test_verifier_grounded_decimal():
    from agent.verifier import is_grounded
    ctx = [{"content": "Torque at peak: 12.8 Nm"}]
    assert is_grounded("Peak torque is 12.8 Nm [Source 1].", ctx)


def test_graph_answer_returns_expected_shape():
    """Full graph round-trip with mocked LLM: shape and types must match API contract."""
    import importlib
    from unittest.mock import patch

    # chat_json drives reason() (returns "answer" immediately) then verify()'s
    # related-questions call
    chat_json_calls = iter([
        {"thought": "Answering directly", "action": "answer", "args": {}},
        ["What is the idle RPM?", "How to check ignition timing?"],
    ])

    def fake_chat_json(messages, **kw):
        return next(chat_json_calls)

    def fake_chat(messages, **kw):
        return "I don't have enough information in the knowledge base to answer that question."

    import agent.graph as ag
    ag._graph = None  # reset singleton so tests don't bleed

    with patch("llm.chat_json", side_effect=fake_chat_json), \
         patch("llm.chat", side_effect=fake_chat):
        result = ag.answer("What is the ignition timing for the Hirth 3503?")

    assert isinstance(result["answer"], str)
    assert isinstance(result["citations"], list)
    assert result["confidence"] in ("high", "low")
    assert isinstance(result["related_questions"], list)


def test_spec_lookup_queries_db():
    """spec_lookup issues the right SQL and maps rows to dicts with source_ref."""
    from unittest.mock import MagicMock, patch

    mock_row = (
        "fuel_data", "Specs", "Benzin", "Value [rpm]",
        "Benzin::Value [rpm]", "6500", "rpm", {"filename": "fuel_data.xlsx"},
    )
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchall.return_value = [mock_row]

    with patch("config.get_connection", return_value=mock_conn):
        from agent import tools
        # reload to pick up patched get_connection
        import importlib; importlib.reload(tools)
        results = tools.spec_lookup("Value [rpm]")

    assert len(results) == 1
    r = results[0]
    assert r["value"] == "6500"
    assert r["unit"] == "rpm"
    assert r["key"] == "Benzin::Value [rpm]"
    assert r["source_ref"] == {"filename": "fuel_data.xlsx"}

