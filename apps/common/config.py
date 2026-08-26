"""Shared application settings (pydantic-settings).

Loads config from, in order of precedence:
  1. process environment variables
  2. `.env` file at the repo root (never committed; see `.env.example`)

Every app imports `settings` from here — never reads `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration shared across all apps.

    Paths default to repo-relative; override with env vars or `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data (App 1 cleansing) --------------------------------------------
    data_dir: Path = Field(
        default=REPO_ROOT / "_assignment" / "synthetic_customer_data",
        description="Directory containing the 5 raw CSVs.",
    )
    db_path: Path = Field(
        default=REPO_ROOT / "data" / "intelligence.duckdb",
        description="Shared DuckDB database file.",
    )

    # --- LLM / OpenAI-compatible API (App 2 agent, App 4 mcp) ---------------
    # The code talks to the OpenAI SDK; the endpoint is any OpenAI-compatible
    # API (default: OpenRouter). Env vars are OPENAI_* so any compatible
    # provider can be plugged in via .env.
    openai_api_key: str | None = Field(
        default=None,
        description="API key for the OpenAI-compatible endpoint. Never commit. The demo runs online — set this in .env before any LLM/embedding step.",
    )
    openai_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible endpoint base URL (default OpenRouter).",
    )
    openai_model: str = Field(
        default="deepseek/deepseek-v4-flash-0731",
        description="Chat model id for the agent (design doc app3-agent.md; env OPENAI_MODEL).",
    )
    openai_embedding_model: str = Field(
        default="voyageai/voyage-4-lite",
        description="Embedding model id (verified on OpenRouter /embeddings).",
    )
    openai_rerank_model: str = Field(
        default="voyageai/rerank-2.5-lite",
        description="Reranking model id (verified on OpenRouter /rerank).",
    )
    openai_max_tokens: int = Field(default=1024, ge=1)
    openai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- Embedding & enrichment (App 2) --------------------------------------
    embedding_dim: int = Field(
        default=1024,
        description="Embedding vector dimension (voyage-4-lite is Matryoshka; pick one and stay in sync via vector.embedding_meta).",
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        description="Embedding requests are batched; this is the max inputs per call.",
    )
    rerank_enabled: bool = Field(
        default=True,
        description="Toggle the rerank step in retrieval (env RERANK_ENABLED).",
    )
    top_k: int = Field(default=5, ge=1, description="Default retrieval top-k.")

    # --- Guardrails (App 5) ---------------------------------------------------
    moderation_enabled: bool = Field(
        default=True,
        description="Toggle the LLM moderation pre-check (env MODERATION_ENABLED).",
    )
    moderation_model: str = Field(
        default="meta-llama/llama-guard-4-12b",
        description="Moderation pre-check model via OpenRouter (togglable; fail-open if unavailable).",
    )
    max_input_chars: int = Field(
        default=4000,
        ge=1,
        description="Length cap on user input (guardrail).",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (re-reads .env only on process restart)."""
    return Settings()


settings = get_settings()

# Module-level convenience aliases (apps may import these directly).
DATA_DIR = settings.data_dir
DB_PATH = settings.db_path
