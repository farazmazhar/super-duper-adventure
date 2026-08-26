"""App 4 — runtime retrieval (RAG via MCP).

Owns the `EmbeddingClient` + rerank config at runtime. Per the architecture
decision (docs/internal/app4-mcp.md), App 4 is the only runtime owner of the
embedding client — the agent reaches retrieval only through this MCP tool, and
no cross-app import of `apps.embedding` happens at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.common import config as common_config  # noqa: E402
from apps.common.config import settings  # noqa: E402


class RetrievalClient:
    """Thin wrapper around the OpenAI-compatible /embeddings + /rerank endpoints.

    Replicates the embed + cosine + optional-rerank flow that App 2's build-time
    `retrieve()` implements, but lives in App 4 so the runtime path has no
    dependency on the batch embedding app. `client_factory` is injectable for
    tests (a fake OpenAI-compatible client).
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str | None = None,
        dimension: int | None = None,
        rerank_model: str | None = None,
        rerank_enabled: bool | None = None,
        client_factory: Any = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or settings.openai_embedding_model
        self.dimension = dimension or settings.embedding_dim
        self.rerank_model = rerank_model or settings.openai_rerank_model
        self.rerank_enabled = settings.rerank_enabled if rerank_enabled is None else rerank_enabled
        self._client = None
        self._client_factory = client_factory

    @property
    def client(self) -> Any:
        """Lazily build the OpenAI client (avoids import cost when unused)."""
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key or "missing-key",
                    base_url=self.base_url,
                )
        return self._client

    def close(self) -> None:
        self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        """Return document indices best-first; identity order if rerank disabled.

        The /rerank endpoint returns results in relevance order; we take the
        index of each result in that order (no re-sorting).
        """
        if not self.rerank_enabled:
            return list(range(len(documents)))
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        resp = self.client.post(path="/rerank", body=payload, cast_to=__import__("httpx").Response)
        return [r["index"] for r in resp.json()["results"]]


def retrieve_sources(
    query: str,
    k: int = 20,
    filters: dict[str, Any] | None = None,
    client: RetrievalClient | None = None,
    rerank_enabled: bool | None = None,
) -> dict[str, Any]:
    """Embedding search over vector.embeddings with optional rerank.

    Returns {data, source_refs, warnings} — the standard tool contract.
    Requires the embedding app to have run (vector.embeddings populated) and an
    API key to be set (online-only demo). `rerank_enabled` overrides the env
    default per call (the FE chat toggle uses this).
    """
    if settings.openai_api_key is None and client is None:
        return {
            "data": [],
            "source_refs": [],
            "warnings": ["No OPENAI_API_KEY — retrieval needs embeddings. Set it in .env first."],
        }

    own_client = client is None
    if client is None:
        client = RetrievalClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            rerank_enabled=rerank_enabled,
        )
    elif rerank_enabled is not None:
        client.rerank_enabled = rerank_enabled

    with duckdb.connect(str(common_config.DB_PATH), read_only=True) as con:
        n = con.execute("SELECT count(*) FROM vector.embeddings").fetchone()[0]
        if n == 0:
            return {
                "data": [],
                "source_refs": [],
                "warnings": ["vector.embeddings is empty — run the embedding app (App 2) first."],
            }

        query_vec = client.embed([query])[0]

        where: list[str] = []
        params: list[Any] = [query_vec]
        if filters:
            if filters.get("customer_id"):
                where.append("customer_id = ?")
                params.append(filters["customer_id"])
            if filters.get("record_type"):
                where.append("record_type = ?")
                params.append(filters["record_type"])
            if filters.get("created_from"):
                where.append("created_at >= ?::TIMESTAMP")
                params.append(filters["created_from"])
            if filters.get("created_to"):
                where.append("created_at <= ?::TIMESTAMP")
                params.append(filters["created_to"])
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        rows = con.execute(
            f"""
            SELECT record_type, record_id, customer_id, created_at, text, metadata,
                   list_cosine_similarity(embedding, ?) AS score
            FROM vector.embeddings
            {where_sql}
            ORDER BY score DESC
            LIMIT ?
            """,
            params + [k],
        ).fetchall()

    docs = [
        {
            "record_type": r[0],
            "record_id": r[1],
            "customer_id": r[2],
            "created_at": r[3],
            "text": r[4],
            "metadata": r[5],
            "score": float(r[6]),
        }
        for r in rows
    ]

    warnings: list[str] = []
    if docs and client.rerank_enabled:
        try:
            indices = client.rerank(query, [d["text"] for d in docs], top_n=min(k, 20))
            docs = [docs[i] for i in indices if i < len(docs)]
        except Exception:  # noqa: BLE001 — online-only: surface, keep cosine order
            warnings.append("rerank skipped (provider error) — results in cosine order.")

    if own_client:
        client.close()

    return {
        "data": docs,
        "source_refs": [f"{d['record_type']}:{d['record_id']}" for d in docs],
        "warnings": warnings,
    }
