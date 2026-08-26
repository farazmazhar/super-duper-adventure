"""Embedding + retrieval — the only network touchpoint in App 2.

Path: `voyageai/voyage-4-lite` (embedding, fixed dim 1024 via Matryoshka) and
`voyageai/rerank-2.5-lite` (rerank) through OpenRouter's OpenAI-compatible
API. On runtime failure (rate limit / overload) we retry with backoff; a call
that still fails raises `EmbeddingError` so the orchestrator can count it as
failed (the demo is online-only — there is no offline fallback model).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import duckdb
import httpx

from apps.embedding.chunker import Chunk
from apps.embedding.config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    RERANK_ENABLED,
    TOP_K,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


class EmbeddingError(RuntimeError):
    """Raised when an embedding call fails after retries (no offline fallback)."""


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    source: str  # 'voyage'
    model: str
    dimension: int


class EmbeddingClient:
    """Thin wrapper around the OpenAI SDK pointed at OpenRouter.

    `client_factory` is injectable for tests (a fake OpenAI-compatible client).
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
        batch_size: int = BATCH_SIZE,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self._client = None
        self._client_factory = client_factory

    # -- client lifecycle -----------------------------------------------------
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

    # -- embed -----------------------------------------------------------------
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a list of texts via OpenRouter (retry with backoff)."""
        try:
            return self._embed_voyage(list(texts))
        except Exception as exc:  # noqa: BLE001 - surface as EmbeddingError
            raise EmbeddingError(
                f"OpenRouter embedding failed after retries: {type(exc).__name__}: {exc}"
            ) from exc

    def _embed_voyage(self, texts: list[str]) -> EmbeddingResult:
        resp = self._call_with_retry(lambda: self.client.embeddings.create(model=self.model, input=texts))
        vectors = [item.embedding for item in resp.data]
        return EmbeddingResult(vectors=vectors, source="voyage", model=self.model, dimension=self.dimension)

    def _call_with_retry(self, call: Callable[[], Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - inspect status if present
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                    wait = BACKOFF_SECONDS * (2**attempt)
                    logger.warning("embedding call failed (status=%s, attempt %d); retrying in %.1fs", status, attempt + 1, wait)
                    time.sleep(wait)
                    continue
                raise
        raise last_exc  # type: ignore[misc]  # pragma: no cover - unreachable

    # -- rerank (used by retrieval; togglable) --------------------------------
    def rerank(self, query: str, documents: Sequence[str], top_n: int = 5) -> list[int]:
        """Call OpenRouter /rerank with the rerank model; return document indices (best first)."""
        if not RERANK_ENABLED:
            return list(range(len(documents)))
        from apps.embedding.config import settings

        payload = {
            "model": settings.openai_rerank_model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
        }
        resp = self._call_with_retry(
            lambda: self.client.post(path="/rerank", body=payload, cast_to=httpx.Response)
        )
        results = sorted(resp.json()["results"], key=lambda r: r["index"])
        return [r["index"] for r in results]


def _hash_chunk(chunk: Chunk) -> str:
    return chunk.source_hash()


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the vector schema + tables if missing (idempotent)."""
    con.execute("CREATE SCHEMA IF NOT EXISTS vector")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS vector.embeddings (
            record_type VARCHAR,
            record_id   VARCHAR,
            customer_id VARCHAR,
            created_at  TIMESTAMP,
            text        VARCHAR,
            metadata    JSON,
            embedding   FLOAT[],
            source      VARCHAR,
            model       VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS vector.embedding_meta (
            model        VARCHAR,
            dimension    INT,
            embedded_at  TIMESTAMP,
            source_hash  VARCHAR
        )
        """
    )


def sync_embeddings(
    con: duckdb.DuckDBPyConnection,
    chunks: list[Chunk],
    client: EmbeddingClient,
    force: bool = False,
) -> dict[str, int]:
    """Incremental embed: skip unchanged (hash match), embed only new/changed.

    Returns {"embedded": n, "skipped": n, "failed": n}.
    """
    _ensure_schema(con)

    cached_hashes: set[str] = set()
    if not force:
        rows = con.execute("SELECT source_hash FROM vector.embedding_meta").fetchall()
        cached_hashes = {r[0] for r in rows}

    to_embed = [c for c in chunks if force or _hash_chunk(c) not in cached_hashes]
    skipped = len(chunks) - len(to_embed)

    embedded = 0
    failed = 0
    for i in range(0, len(to_embed), client.batch_size):
        batch = to_embed[i : i + client.batch_size]
        try:
            result = client.embed_texts([c.text for c in batch])
        except EmbeddingError:
            failed += len(batch)
            continue
        rows_to_insert = [
            (
                c.record_type,
                c.record_id,
                c.customer_id,
                c.created_at,
                c.text,
                c.metadata,
                vec,
                result.source,
                result.model,
            )
            for c, vec in zip(batch, result.vectors)
        ]
        # Embeddings + their cache-meta rows are committed atomically per batch,
        # so an interrupted run can never leave rows that a later run re-embeds.
        con.execute("BEGIN")
        try:
            con.executemany(
                """
                INSERT INTO vector.embeddings
                    (record_type, record_id, customer_id, created_at, text, metadata, embedding, source, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            con.executemany(
                "INSERT INTO vector.embedding_meta (model, dimension, embedded_at, source_hash) VALUES (?, ?, now(), ?)",
                [(result.model, result.dimension, _hash_chunk(c)) for c in batch],
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        embedded += len(batch)

    return {"embedded": embedded, "skipped": skipped, "failed": failed}


def retrieve(
    con: duckdb.DuckDBPyConnection,
    query: str,
    client: EmbeddingClient,
    top_k: int = TOP_K,
    rerank_top_n: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieval: embed query (same model/dim) -> cosine over vector.embeddings -> optional rerank.

    `filters` may contain customer_id, record_type, created_from/created_to
    (ISO datetime strings). Returns top-k rows with a similarity score, reranked
    to rerank_top_n when enabled.
    """
    result = client.embed_texts([query])
    query_vec = result.vectors[0]

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
        SELECT record_type, record_id, customer_id, created_at, text, metadata, embedding,
               list_cosine_similarity(embedding, ?) AS score
        FROM vector.embeddings
        {where_sql}
        ORDER BY score DESC
        LIMIT ?
        """,
        params + [top_k],
    ).fetchall()

    docs = [
        {
            "record_type": r[0],
            "record_id": r[1],
            "customer_id": r[2],
            "created_at": r[3],
            "text": r[4],
            "metadata": r[5],
            "score": float(r[7]),
        }
        for r in rows
    ]
    if not docs or not RERANK_ENABLED:
        return docs

    indices = client.rerank(query, [d["text"] for d in docs], top_n=rerank_top_n)
    ordered = [docs[i] for i in indices if i < len(docs)]
    return ordered
