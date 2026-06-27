"""Smoke tests — confirm the app imports and basic routing works before features land."""
from __future__ import annotations


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
