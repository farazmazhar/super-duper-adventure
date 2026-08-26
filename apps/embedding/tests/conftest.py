"""Shared fixtures for App 2 tests: tiny fact tables + a fake embedding client.

The App 2 pipeline reads `main.fact_feedback` / `main.fact_ticket` and writes
`main.aggregate_theme` (seeded by App 1). These fixtures build just those
tables so tests are independent of App 1's full pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pytest

APP_DIR = Path(__file__).resolve().parents[1]  # apps/embedding

DDL = """
CREATE SCHEMA IF NOT EXISTS main;
CREATE TABLE IF NOT EXISTS main.fact_feedback (
    feedback_id VARCHAR, customer_id VARCHAR, created_at TIMESTAMP,
    feedback_text VARCHAR, feedback_source VARCHAR, rating BIGINT
);
CREATE TABLE IF NOT EXISTS main.fact_ticket (
    ticket_id VARCHAR, customer_id VARCHAR, created_at TIMESTAMP,
    subject VARCHAR, message VARCHAR, category VARCHAR, priority VARCHAR,
    resolution_time_hours DOUBLE, status VARCHAR, satisfaction_score DOUBLE
);
CREATE TABLE IF NOT EXISTS main.aggregate_theme (
    feedback_id VARCHAR, customer_id VARCHAR, created_at TIMESTAMP, text VARCHAR,
    theme VARCHAR, sentiment VARCHAR, source VARCHAR
);
"""

FEEDBACK = [
    ("FDB-0001", "CUST-0001", "2026-05-10 09:00", "Love the search feature", "support_chat", 5),
    ("FDB-0002", "CUST-0002", "2026-05-11 09:00", "The invoice was wrong and the bill is broken", "email", 2),
    ("FDB-0003", "CUST-0003", "2026-05-12 09:00", None, "app_store_review", 3),
    ("FDB-0004", "CUST-0001", "2026-05-13 09:00", "Great API docs but slow support", "nps_survey", 4),
]

TICKETS = [
    ("TCK-0001", "CUST-0001", "2026-05-01 10:00", "Bug", "It crashes on export", "bug", "high", 5.0, "resolved", 4.0),
    ("TCK-0002", "CUST-0002", "2026-05-02 11:00", "Question", None, "general_question", "low", None, "open", None),
    ("TCK-0003", "CUST-0003", "2026-05-03 12:00", "Invoice request", "Can we get an itemized invoice?", "billing", "medium", -5.0, "resolved", 3.0),
]


class FakeEmbeddings:
    """Mimics openai.resources.embeddings.Embeddings.create()."""

    def create(self, model: str, input: list[str]) -> "FakeEmbeddingResponse":
        return FakeEmbeddingResponse([FakeEmbeddingItem(self._vec(text)) for text in input])

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:8]]


class FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, data: list[FakeEmbeddingItem]) -> None:
        self.data = data


class FakeRerankResponse:
    def __init__(self, indices: list[int]) -> None:
        self._indices = indices

    def json(self) -> dict:
        return {"results": [{"index": i} for i in self._indices]}


class FakeOpenAIClient:
    """Drop-in OpenAI client with .embeddings.create and .post(path, body)."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()
        self.rerank_indices: list[int] = []
        self.last_rerank_body: dict | None = None

    def post(self, path: str, body: dict, cast_to=None) -> FakeRerankResponse:
        assert path == "/rerank"
        self.last_rerank_body = body
        return FakeRerankResponse(self.rerank_indices)


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(DDL)
    con.executemany("INSERT INTO main.fact_feedback VALUES (?,?,?,?,?,?)", FEEDBACK)
    con.executemany("INSERT INTO main.fact_ticket VALUES (?,?,?,?,?,?,?,?,?,?)", TICKETS)
    yield con
    con.close()
