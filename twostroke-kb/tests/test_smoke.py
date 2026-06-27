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


# --- Slice 4: recommender + reranker wired into act() -----------------------

def test_recommender_llm_path():
    """When the graph DB is unavailable, recommender falls back to LLM suggestions."""
    from unittest.mock import patch

    chunks = [{"content": "Hirth 3503 ignition timing is 28 degrees BTDC."}]

    def fake_chat_json(messages, **kw):
        return ["What is the idle RPM?", "How to adjust ignition timing?"]

    with patch("llm.chat_json", side_effect=fake_chat_json), \
         patch("config.get_connection", side_effect=RuntimeError("no db")):
        from agent.recommender import related
        result = related("What is the ignition timing?", chunks)

    assert isinstance(result, list)
    assert 1 <= len(result) <= 3
    assert all(isinstance(q, str) for q in result)


def test_recommender_empty_chunks():
    """No chunks and no graph → empty list, no error."""
    from unittest.mock import patch

    with patch("config.get_connection", side_effect=RuntimeError("no db")):
        from agent.recommender import related
        result = related("What is the RPM?", [])

    assert result == []


def test_act_reranks_hybrid_search_results():
    """act() must apply the reranker after hybrid_search and return results in reranked order."""
    from unittest.mock import patch, MagicMock
    from agent.nodes import act, AgentState

    raw = [
        {"content": "Low relevance text", "doc_id": "a", "source_refs": [], "metadata": {}, "score": 0.5},
        {"content": "High relevance text", "doc_id": "b", "source_refs": [], "metadata": {}, "score": 0.4},
    ]
    reranked = [
        {**raw[1], "rerank_score": 0.92},
        {**raw[0], "rerank_score": 0.11},
    ]
    mock_rerank = MagicMock(return_value=reranked)

    state: AgentState = {
        "question": "What is the max RPM?",
        "lang": "en",
        "expertise": "expert",
        "scratch": [],
        "tool_action": "hybrid_search",
        "tool_args": {"query": "max RPM"},
        "draft": "",
        "citations": [],
        "grounded": False,
        "loops": 1,
        "related": [],
    }

    with patch("agent.retriever_hybrid.search", return_value=raw), \
         patch("agent.reranker.rerank", mock_rerank):
        result = act(state)

    assert result["scratch"][0]["results"][0]["doc_id"] == "b"  # highest rerank score first
    mock_rerank.assert_called_once_with("max RPM", raw)


# --- Slice 5: memory, dedup, enricher, graph_builder, tools -----------------

def test_memory_record_feedback():
    """record_feedback executes an INSERT on the feedback table."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from memory.store import record_feedback
        record_feedback("sess1", "What is RPM?", "It is 6500.", vote=1)

    mock_cur.execute.assert_called_once()
    sql = mock_cur.execute.call_args[0][0]
    assert "INSERT INTO feedback" in sql


def test_memory_get_conversation_empty():
    """get_conversation returns [] for an unknown session."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value  = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from memory.store import get_conversation
        turns = get_conversation("unknown-session")

    assert turns == []


def test_dedup_skips_near_duplicate():
    """dedup_and_merge drops a chunk when a near-identical vector is already stored."""
    from unittest.mock import MagicMock, patch

    chunk = {
        "content": "The max RPM is 6500.",
        "embedding": [0.1] * 384,
        "metadata": {},
        "source_refs": [{"filename": "manual.pdf"}],
    }

    # _find_near_duplicate returns id=42 → chunk is a duplicate
    with patch("ingestion.dedup._find_near_duplicate", return_value=42), \
         patch("ingestion.dedup._merge_source_refs") as mock_merge:
        from ingestion.dedup import dedup_and_merge
        result = dedup_and_merge([chunk])

    assert result == []
    mock_merge.assert_called_once_with(42, [{"filename": "manual.pdf"}])


def test_dedup_keeps_unique_chunk():
    """dedup_and_merge keeps a chunk when no near-duplicate exists."""
    from unittest.mock import patch

    chunk = {
        "content": "Unique spec text.",
        "embedding": [0.1] * 384,
        "metadata": {},
        "source_refs": [],
    }

    with patch("ingestion.dedup._find_near_duplicate", return_value=None):
        from ingestion.dedup import dedup_and_merge
        result = dedup_and_merge([chunk])

    assert len(result) == 1


def test_domain_enricher_attaches_entities():
    """enrich() attaches entities and tags to chunk metadata on LLM success."""
    from unittest.mock import patch

    chunks = [{"content": "A" * 100, "metadata": {"filename": "x.pdf"}, "source_refs": []}]

    def fake_chat_json(messages, **kw):
        return {"entities": [{"type": "engine", "name": "Hirth 3503"}], "tags": ["ignition"]}

    with patch("llm.chat_json", side_effect=fake_chat_json):
        from ingestion.domain_enricher import enrich
        result = enrich(chunks)

    assert result[0]["metadata"]["entities"] == [{"type": "engine", "name": "Hirth 3503"}]
    assert "ignition" in result[0]["metadata"]["tags"]


def test_domain_enricher_skips_short_chunks():
    """Chunks shorter than the minimum threshold are returned unchanged."""
    from ingestion.domain_enricher import enrich

    chunk = {"content": "Short.", "metadata": {}, "source_refs": []}
    result = enrich([chunk])
    assert "entities" not in result[0].get("metadata", {})


def test_unit_convert_basic():
    from agent.tools import unit_convert
    assert unit_convert(1.0, "kW", "hp") == pytest.approx(1.34102, rel=1e-3)
    assert unit_convert(100.0, "°C", "°F") == pytest.approx(212.0, rel=1e-3)
    assert unit_convert(0.0, "°C", "°F") == pytest.approx(32.0, rel=1e-3)
    assert unit_convert(6500.0, "rpm", "rpm") == pytest.approx(6500.0)


def test_unit_convert_unknown_raises():
    from agent.tools import unit_convert
    with pytest.raises(ValueError, match="no conversion"):
        unit_convert(1.0, "furlong", "fortnight")


def test_conflict_check_flags_divergence():
    """conflict_check marks rows as conflict=True when multiple values exist for a key."""
    from unittest.mock import MagicMock, patch

    rows = [
        ("max_rpm::Value [rpm]", "6500", "doc_a", {}),
        ("max_rpm::Value [rpm]", "7000", "doc_b", {}),
    ]
    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_conn.cursor.return_value  = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from agent.tools import conflict_check
        result = conflict_check("max_rpm")

    assert len(result) == 2
    assert all(r["conflict"] for r in result)


def test_source_viewer_not_found():
    """source_viewer returns an error dict when chunk_id does not exist."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value  = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from agent.tools import source_viewer
        result = source_viewer(9999)

    assert "error" in result


# --- Slice 6: graph_builder, graph_lookup, diagnostic_tree, expertise mode --

def test_graph_builder_calls_llm_and_upserts():
    """graph_builder.extract() calls the LLM and attempts DB upsert."""
    from unittest.mock import MagicMock, patch
    from ingestion.types import ParsedDoc

    doc = ParsedDoc(
        text="The Hirth 3503 engine overheats when the carburettor jets are clogged.",
        metadata={"filename": "manual.pdf", "type": "pdf"},
        source_ref={"filename": "manual.pdf"},
    )
    llm_result = {
        "nodes": [
            {"type": "engine",  "name": "Hirth 3503"},
            {"type": "symptom", "name": "overheating"},
            {"type": "cause",   "name": "clogged jets"},
        ],
        "edges": [
            {"src": "Hirth 3503", "dst": "overheating", "relation": "has_symptom"},
        ],
    }

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = (1,)
    mock_conn.cursor.return_value  = mock_cur

    with patch("llm.chat_json", return_value=llm_result), \
         patch("config.get_connection", return_value=mock_conn):
        from ingestion import graph_builder
        graph_builder.extract(doc)

    # Should have called execute for node inserts + edge insert
    assert mock_cur.execute.call_count >= 2


def test_graph_answer_beginner_mode():
    """graph.answer() with expertise='beginner' produces a result with the right shape."""
    import importlib
    from unittest.mock import patch

    chat_json_calls = iter([
        {"thought": "Answering directly", "action": "answer", "args": {}},
        ["Follow-up question?"],
    ])

    def fake_chat_json(messages, **kw):
        return next(chat_json_calls)

    def fake_chat(messages, **kw):
        # Verify the beginner style note is present in the system prompt
        system = messages[0]["content"]
        assert "plain" in system.lower() or "beginner" in system.lower() or "jargon" in system.lower()
        return "I don't have enough information to answer that."

    import agent.graph as ag
    ag._graph = None

    with patch("llm.chat_json", side_effect=fake_chat_json), \
         patch("llm.chat", side_effect=fake_chat):
        result = ag.answer("What is ignition timing?", expertise="beginner")

    assert isinstance(result["answer"], str)
    assert result["confidence"] in ("high", "low")


def test_api_health_and_gaps_endpoints():
    """Health endpoint returns ok; /gaps returns the expected shape when DB is unavailable."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"ok": True}

    # /gaps without a DB returns the graceful error shape
    resp = client.get("/gaps")
    data = resp.json()
    assert "gaps" in data


def test_api_graph_endpoint():
    """/graph returns nodes and edges keys."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    data = client.get("/graph").json()
    assert "nodes" in data
    assert "edges" in data


# --- Slice 7: pptx_parser, BM25/RRF, figure_handler, stream_chat, versioning --

def test_pptx_parser_extracts_slides():
    """pptx_parser returns slide text from every shape and speaker notes."""
    from ingestion.parsers import pptx_parser

    doc = pptx_parser.parse(FIXTURES / "sample.pptx")
    assert doc.metadata["type"] == "pptx"
    assert doc.metadata["slides"] == 2
    assert "Hirth 3503" in doc.text
    assert "6500" in doc.text
    assert "Speaker notes" in doc.text or "2-stroke oil" in doc.text
    assert "--- Slide 1 ---" in doc.text
    assert "--- Slide 2 ---" in doc.text


def test_pptx_parser_source_ref():
    """source_ref must carry filename and slide_count."""
    from ingestion.parsers import pptx_parser

    doc = pptx_parser.parse(FIXTURES / "sample.pptx")
    assert doc.source_ref["filename"] == "sample.pptx"
    assert doc.source_ref["slide_count"] == 2


def test_format_router_routes_pptx():
    """format_router.route() dispatches .pptx to pptx_parser without error."""
    from ingestion.format_router import route

    doc = route(FIXTURES / "sample.pptx")
    assert doc.metadata["type"] == "pptx"
    assert "lang" in doc.metadata


def test_rrf_fuse_combines_lists():
    """_rrf_fuse gives a chunk appearing in both lists a higher score than one in only one."""
    from agent.retriever_hybrid import _rrf_fuse

    chunk_both   = {"id": 1, "content": "in both",   "score": 0.9}
    chunk_dense  = {"id": 2, "content": "dense only", "score": 0.8}
    chunk_sparse = {"id": 3, "content": "sparse only","score": 0.7}

    dense  = [chunk_both, chunk_dense]
    sparse = [chunk_both, chunk_sparse]

    result = _rrf_fuse(dense, sparse, top_k=3)

    ids = [c["id"] for c in result]
    assert ids[0] == 1, "chunk in both lists should rank first"
    assert len(result) == 3


def test_rrf_fuse_top_k_limits_output():
    """_rrf_fuse must honour top_k even when inputs are larger."""
    from agent.retriever_hybrid import _rrf_fuse

    dense  = [{"id": i, "content": f"d{i}", "score": 1/(i+1)} for i in range(10)]
    sparse = [{"id": i+5, "content": f"s{i}", "score": 1/(i+1)} for i in range(10)]

    result = _rrf_fuse(dense, sparse, top_k=4)
    assert len(result) == 4


def test_bm25_tokenize():
    """_tokenize lowercases and splits on whitespace — DE+EN corpus, no NLTK needed."""
    from agent.retriever_hybrid import _tokenize

    tokens = _tokenize("Hirth 3503 Zündung BTDC")
    assert tokens == ["hirth", "3503", "zündung", "btdc"]


def test_figure_handler_disabled_returns_empty():
    """With ENABLE_FIGURE_HANDLER=false, extract_and_caption returns an empty ParsedDoc."""
    from unittest.mock import patch
    from ingestion.parsers import figure_handler

    with patch("config.get_settings") as mock_settings:
        mock_settings.return_value.enable_figure_handler = False
        doc = figure_handler.extract_and_caption(FIXTURES / "sample.pdf")

    assert doc.text == ""
    assert doc.metadata["type"] == "figure"
    assert doc.images == []


def test_stream_chat_yields_tokens():
    """stream_chat must yield individual string tokens from the provider."""
    from unittest.mock import patch, MagicMock

    # Simulate OpenAI streaming: a list of chunk objects with delta.content
    def _make_chunk(text):
        c = MagicMock()
        c.choices[0].delta.content = text
        return c

    fake_stream = [_make_chunk("The "), _make_chunk("max "), _make_chunk("RPM "), _make_chunk("is 6500.")]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(fake_stream)

    with patch("config.get_settings") as mock_cfg, \
         patch("openai.OpenAI", return_value=mock_client):
        mock_cfg.return_value.llm_provider = "openai"
        mock_cfg.return_value.llm_model    = "gpt-4o-mini"
        mock_cfg.return_value.openai_api_key = "test"

        from llm import stream_chat
        tokens = list(stream_chat([{"role": "user", "content": "What is max RPM?"}]))

    assert tokens == ["The ", "max ", "RPM ", "is 6500."]
    assert "".join(tokens) == "The max RPM is 6500."


def test_ask_stream_endpoint_sse_format():
    """/ask/stream emits well-formed SSE events ending with a 'done' event."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from api.main import app

    fake_tokens = iter(["The ", "answer ", "is 6500."])

    def fake_search(q, k=None): return []  # no chunks → straight to done event

    with patch("agent.retriever_hybrid.search", side_effect=fake_search):
        client = TestClient(app)
        resp = client.post(
            "/ask/stream",
            data={"question": "What is the max RPM?", "session_id": "test", "expertise": "expert"},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n\n")
    events = []
    for line in lines:
        if line.startswith("data: "):
            import json as _json
            events.append(_json.loads(line[6:]))

    types = [e["type"] for e in events]
    assert "done" in types
    done = next(e for e in events if e["type"] == "done")
    assert "citations" in done
    assert "confidence" in done
    assert "related_questions" in done


def test_document_versioning_increments():
    """_register_document inserts version=MAX+1 and returns the new version number."""
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchone.return_value = (2,)   # MAX(version) = 2 → next = 3
    mock_conn.cursor.return_value = mock_cur

    with patch("config.get_connection", return_value=mock_conn):
        from ingestion.orchestrator import _register_document
        version = _register_document("hirth_3503_manual", "manual.pdf", "de")

    assert version == 3
    sql = mock_cur.execute.call_args_list[1][0][0]
    assert "INSERT INTO documents" in sql


def test_ingest_result_has_version_field():
    """IngestResult dataclass must expose a version field."""
    from ingestion.orchestrator import IngestResult

    r = IngestResult(filename="a.pdf", chunks=5, facts=2, skipped_duplicates=1, version=3)
    assert r.version == 3

