"""Shared pydantic-settings config tests.

Checks defaults, .env loading, and env-var precedence. Uses a temp .env via
env_file override so the tests don't depend on the developer's real .env.
"""

from __future__ import annotations

from pathlib import Path

from apps.common.config import Settings

TMP_ENV = """\
OPENAI_BASE_URL=https://override.example/v1
OPENAI_MODEL=file-model
"""


def test_defaults_without_env(monkeypatch, tmp_path: Path) -> None:
    for var in (
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_RERANK_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    # _env_file=None + env vars cleared -> pure defaults (repo .env not read)
    s = Settings(_env_file=None)
    assert s.openai_base_url == "https://openrouter.ai/api/v1"
    assert s.openai_model == "deepseek/deepseek-v4-flash-0731"
    assert s.openai_api_key is None
    assert s.openai_rerank_model == "voyageai/rerank-2.5-lite"
    assert s.embedding_dim == 1024


def test_env_file_loaded(monkeypatch, tmp_path: Path) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(TMP_ENV)

    s = Settings(_env_file=env_file)
    assert s.openai_base_url == "https://override.example/v1"
    assert s.openai_model == "file-model"


def test_env_var_overrides_env_file(tmp_path: Path, monkeypatch) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(TMP_ENV)
    monkeypatch.setenv("OPENAI_MODEL", "env-var-model")

    s = Settings(_env_file=env_file)
    assert s.openai_model == "env-var-model"  # process env wins
    assert s.openai_base_url == "https://override.example/v1"  # from file


def test_paths_are_repo_relative() -> None:
    s = Settings(_env_file=None)
    root = Path(__file__).resolve().parents[3]
    assert s.db_path == root / "data" / "intelligence.duckdb"
    assert s.data_dir == root / "_assignment" / "synthetic_customer_data"
