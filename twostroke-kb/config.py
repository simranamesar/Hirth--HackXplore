"""Central settings + DB helper. Read from .env via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/twostroke"

    # llm — "local" uses an OpenAI-compatible endpoint (Ollama /v1, vLLM, LM Studio, etc.)
    llm_provider: str = "local"             # local | openai | anthropic
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen3:8b"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # embeddings / rerank
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieve_top_k: int = 20
    rerank_top_k: int = 5

    # ingestion
    dedup_cosine_threshold: float = 0.98
    ocr_langs: str = "deu+eng"
    upload_dir: str = "./data/uploads"
    enable_figure_handler: bool = False
    llm_vision_model: str = "llama3.2:3b"  # model used by describe_image(); pulled separately

    # app
    app_host: str = "0.0.0.0"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_connection():
    """Return a psycopg connection. TODO: pool for production."""
    import psycopg  # local import so the module loads even before deps installed

    return psycopg.connect(get_settings().database_url)
