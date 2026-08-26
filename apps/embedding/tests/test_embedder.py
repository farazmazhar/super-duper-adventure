"""Embedder tests: sync writes, cache skip, retrieval + rerank (all mocked, no network)."""

from __future__ import annotations

import duckdb
import pytest

from apps.embedding.chunker import build_chunks
from apps.embedding.embedder import EmbeddingClient, sync_embeddings, retrieve
from apps.embedding.run import enrich_themes
from apps.embedding.tests.conftest import FakeOpenAIClient


def make_client(fake: FakeOpenAIClient) -> EmbeddingClient:
    return EmbeddingClient(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        batch_size=2,
        client_factory=lambda: fake,
    )


def test_sync_embeddings_writes_vector_schema(con: duckdb.DuckDBPyConnection) -> None:
    fake = FakeOpenAIClient()
    client = make_client(fake)
    chunks = build_chunks(con)
    counts = sync_embeddings(con, chunks, client)

    assert counts == {"embedded": len(chunks), "skipped": 0, "failed": 0}
    n = con.execute("SELECT count(*) FROM vector.embeddings").fetchone()[0]
    assert n == len(chunks)
    # one row per chunk in embedding_meta with a hash
    assert con.execute("SELECT count(*) FROM vector.embedding_meta").fetchone()[0] == len(chunks)
    # embeddings stored as FLOAT[] with expected dimension (8 here: fake vectors)
    dim = con.execute("SELECT len(embedding) FROM vector.embeddings LIMIT 1").fetchone()[0]
    assert dim == 8
    # source/model recorded
    assert con.execute("SELECT DISTINCT source FROM vector.embeddings").fetchall() == [("voyage",)]
    assert con.execute("SELECT DISTINCT model FROM vector.embeddings").fetchall() == [("voyageai/voyage-4-lite",)]


def test_sync_embeddings_is_incremental(con: duckdb.DuckDBPyConnection) -> None:
    fake = FakeOpenAIClient()
    client = make_client(fake)
    chunks = build_chunks(con)

    first = sync_embeddings(con, chunks, client)
    assert first["embedded"] == len(chunks)

    # Second run: all cached -> all skipped, nothing re-embedded
    second = sync_embeddings(con, chunks, client)
    assert second == {"embedded": 0, "skipped": len(chunks), "failed": 0}
    assert con.execute("SELECT count(*) FROM vector.embeddings").fetchone()[0] == len(chunks)

    # A changed record -> re-embedded (hash differs)
    con.execute(
        "UPDATE main.fact_feedback SET feedback_text = 'The invoice was changed' WHERE feedback_id = 'FDB-0002'"
    )
    third = sync_embeddings(con, build_chunks(con), client)
    assert third["embedded"] == 1
    assert third["skipped"] == len(chunks) - 1


def test_sync_embeddings_force(con: duckdb.DuckDBPyConnection) -> None:
    fake = FakeOpenAIClient()
    client = make_client(fake)
    chunks = build_chunks(con)
    sync_embeddings(con, chunks, client)
    counts = sync_embeddings(con, chunks, client, force=True)
    assert counts["embedded"] == len(chunks)
    assert counts["skipped"] == 0


def test_retrieve_returns_top_k(con: duckdb.DuckDBPyConnection) -> None:
    fake = FakeOpenAIClient()
    client = make_client(fake)
    sync_embeddings(con, build_chunks(con), client)

    fake.rerank_indices = [2, 0, 1, 3, 4]
    results = retrieve(con, "invoice billing", client, top_k=5, rerank_top_n=3)
    assert len(results) == 5  # no rerank call when not enabled? RERANK_ENABLED is on by default
    assert all("text" in r and "score" in r for r in results)


def test_retrieve_filters(con: duckdb.DuckDBPyConnection) -> None:
    fake = FakeOpenAIClient()
    client = make_client(fake)
    sync_embeddings(con, build_chunks(con), client)
    fake.rerank_indices = [0, 1, 2, 3, 4, 5]

    results = retrieve(
        con, "invoice", client, top_k=50,
        filters={"record_type": "ticket", "customer_id": "CUST-0003"},
    )
    assert len(results) >= 1
    assert all(r["record_type"] == "ticket" for r in results)
    assert all(r["customer_id"] == "CUST-0003" for r in results)


def test_retrieve_rerank_disabled(con: duckdb.DuckDBPyConnection, monkeypatch) -> None:
    monkeypatch.setattr("apps.embedding.embedder.RERANK_ENABLED", False)
    fake = FakeOpenAIClient()
    client = make_client(fake)
    sync_embeddings(con, build_chunks(con), client)

    results = retrieve(con, "invoice", client, top_k=3)
    assert len(results) == 3


def test_embedder_failure_counts_as_failed(con: duckdb.DuckDBPyConnection) -> None:
    """A total OpenRouter failure raises EmbeddingError -> sync counts failed (no fallback)."""
    class ExplodingClient:
        def __init__(self) -> None:
            self.embeddings = ExplodingEmbeddings()

    class ExplodingEmbeddings:
        def create(self, model: str, input: list[str]) -> None:
            raise RuntimeError("connection refused")

    client = EmbeddingClient(
        api_key="test-key", base_url="x", batch_size=2,
        client_factory=lambda: ExplodingClient(),
    )
    chunks = build_chunks(con)
    counts = sync_embeddings(con, chunks, client)
    assert counts["embedded"] == 0
    assert counts["failed"] == len(chunks)
    assert con.execute("SELECT count(*) FROM vector.embeddings").fetchone()[0] == 0


def test_enrich_themes_is_idempotent(con: duckdb.DuckDBPyConnection) -> None:
    """enrich_themes refreshes rule-seeded rows instead of duplicating them."""
    first = enrich_themes(con)
    assert first == 4  # 4 feedback rows in the fixture
    assert con.execute("SELECT count(*) FROM main.aggregate_theme WHERE source = 'rule'").fetchone()[0] == 4

    second = enrich_themes(con)
    assert second == 4
    # still exactly 4 rule rows after a re-run — no duplicates
    assert con.execute("SELECT count(*) FROM main.aggregate_theme WHERE source = 'rule'").fetchone()[0] == 4
    # LLM overlays are untouched
    con.execute("INSERT INTO main.aggregate_theme VALUES ('FDB-0001', 'CUST-0001', NULL, 'x', 'billing', 'positive', 'llm')")
    enrich_themes(con)
    assert con.execute("SELECT count(*) FROM main.aggregate_theme WHERE source = 'llm'").fetchone()[0] == 1
