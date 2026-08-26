"""Retrieval tests: embed -> cosine -> optional rerank (all mocked, no network)."""

from __future__ import annotations

from apps.mcp.retrieval import RetrievalClient, retrieve_sources


class FakeClient:
    """Fake OpenAI-compatible client: deterministic embeddings + rerank indices."""

    def __init__(self, query_vec: list[float], rerank_indices: list[int] | None = None) -> None:
        self.query_vec = query_vec
        self.rerank_indices = rerank_indices or [0, 1, 2]
        self.embed_calls = 0
        self.rerank_calls = 0
        self.embeddings = _Embeddings(self)  # attribute, matching the real OpenAI client

    def post(self, path: str, body: dict, cast_to=None):
        assert path == "/rerank"
        self.rerank_calls += 1
        return _RerankResponse(self.rerank_indices)


class _Embeddings:
    def __init__(self, parent: FakeClient) -> None:
        self._parent = parent

    def create(self, model: str, input: list[str]) -> "_EmbedResult":
        self._parent.embed_calls += 1
        return _EmbedResult([_EmbeddingItem(self._parent.query_vec) for _ in input])


class _EmbeddingItem:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _EmbedResult:
    def __init__(self, data: list) -> None:
        self.data = data


class _RerankResponse:
    def __init__(self, indices: list[int]) -> None:
        # `indices` is in RERANK-PRIORITY order (best first), matching the real
        # /rerank endpoint which returns results sorted by relevance.
        self._indices = indices

    def json(self) -> dict:
        return {"results": [{"index": i} for i in self._indices]}


def make_client(
    query_vec: list[float] | None = None, rerank_indices: list[int] | None = None,
) -> tuple[RetrievalClient, FakeClient]:
    fake = FakeClient(query_vec or [1.0, 0.0, 0.0], rerank_indices)
    client = RetrievalClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        rerank_enabled=False,  # default off for deterministic cosine tests
        client_factory=lambda: fake,
    )
    return client, fake


def test_retrieve_returns_cosine_top_k(mcp_db) -> None:
    client, fake = make_client(query_vec=[1.0, 0.0, 0.0])
    res = retrieve_sources("search", k=3, client=client)
    # query vector [1,0,0] matches FDB-0001 (embedding [1,0,0]) best
    assert len(res["data"]) == 3
    assert res["data"][0]["record_id"] == "FDB-0001"
    assert res["data"][0]["score"] > res["data"][1]["score"]
    assert res["source_refs"] == ["feedback:FDB-0001", "ticket:TCK-0002", "feedback:FDB-0002"]
    assert fake.embed_calls == 1


def test_retrieve_filters(mcp_db) -> None:
    client, _ = make_client(query_vec=[0.0, 1.0, 0.0])
    res = retrieve_sources(
        "outage", k=10, filters={"record_type": "ticket", "customer_id": "CUST-0002"}, client=client,
    )
    assert len(res["data"]) == 1
    assert res["data"][0]["record_id"] == "TCK-0002"


def test_retrieve_rerank_reorders(mcp_db) -> None:
    client, fake = make_client(query_vec=[1.0, 0.0, 0.0], rerank_indices=[2, 0, 1])
    client.rerank_enabled = True
    res = retrieve_sources("search", k=3, client=client)
    # reranked order: FDB-0002, FDB-0001, TCK-0002
    assert [d["record_id"] for d in res["data"]] == ["FDB-0002", "FDB-0001", "TCK-0002"]
    assert fake.rerank_calls == 1


def test_retrieve_rerank_disabled(mcp_db) -> None:
    client, fake = make_client(query_vec=[1.0, 0.0, 0.0], rerank_indices=[2, 0, 1])
    assert client.rerank_enabled is False
    res = retrieve_sources("search", k=3, client=client)
    assert [d["record_id"] for d in res["data"]] == ["FDB-0001", "TCK-0002", "FDB-0002"]
    assert fake.rerank_calls == 0


def test_retrieve_rerank_error_keeps_cosine_order(mcp_db) -> None:
    class ExplodingRerank(FakeClient):
        def post(self, path: str, body: dict, cast_to=None):
            raise RuntimeError("provider down")

    fake = ExplodingRerank([1.0, 0.0, 0.0], [2, 0, 1])
    client = RetrievalClient(
        api_key="test-key", base_url="https://example.test/v1",
        rerank_enabled=True, client_factory=lambda: fake,
    )
    res = retrieve_sources("search", k=3, client=client)
    # cosine order preserved, warning surfaced
    assert [d["record_id"] for d in res["data"]] == ["FDB-0001", "TCK-0002", "FDB-0002"]
    assert any("rerank skipped" in w for w in res["warnings"])


def test_retrieve_empty_table(mcp_db, tmp_path, monkeypatch) -> None:
    """Point DB_PATH at a fresh empty DB -> vector.embeddings missing/empty warning."""
    import duckdb
    from pathlib import Path
    from apps.mcp import retrieval as retrieval_mod
    from apps.common import config as common_config

    empty_db = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(empty_db))
    con.execute("CREATE SCHEMA vector")
    con.execute("CREATE TABLE vector.embeddings (record_type VARCHAR)")
    con.close()
    monkeypatch.setattr(common_config, "DB_PATH", empty_db)

    client, _ = make_client()
    res = retrieval_mod.retrieve_sources("anything", client=client)
    assert res["data"] == []
    assert any("empty" in w for w in res["warnings"])
