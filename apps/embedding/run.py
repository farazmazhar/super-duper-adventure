"""App 2 — Embedding & Enrichment orchestrator.

chunk -> enrich (rule-based themes) -> cache-check -> embed -> store.
Run after App 1 (needs main.fact_*) and before App 3 (reads vector.*).

    python -m apps.embedding.run
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb

# Make the repo root importable (repo-root `apps/` package) regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from apps.embedding.chunker import build_chunks  # noqa: E402
from apps.embedding.config import DB_PATH, settings  # noqa: E402
from apps.embedding.embedder import EmbeddingClient, sync_embeddings  # noqa: E402
from apps.embedding.themes import enrich_feedback_row  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def check_inputs(con: duckdb.DuckDBPyConnection) -> None:
    """Fail fast if App 1 hasn't produced the fact tables yet."""
    required = ("main.fact_feedback", "main.fact_ticket", "main.aggregate_theme")
    missing = [
        t
        for t in required
        if not con.execute(
            f"SELECT count(*) FROM information_schema.tables WHERE table_name = ? AND table_schema = ?",
            [t.split(".")[1], t.split(".")[0]],
        ).fetchone()[0]
    ]
    if missing:
        sys.exit(
            f"Missing input table(s) {', '.join(missing)} — run App 1 cleansing first."
        )


def enrich_themes(con: duckdb.DuckDBPyConnection) -> int:
    """Rule-based theme + sentiment baseline into main.aggregate_theme (source='rule').

    Idempotent: App 1 already seeds this table (source='rule'); re-running App 2
    must not duplicate rows, so we refresh the rule-seeded rows in place. LLM
    overlays (source='llm') are left untouched.
    """
    rows = con.execute(
        "SELECT feedback_id, customer_id, created_at, feedback_text, rating FROM main.fact_feedback"
    ).fetchall()
    enriched = [
        (*enrich_feedback_row(feedback_id, customer_id, created_at, text, rating), text)
        for feedback_id, customer_id, created_at, text, rating in rows
    ]
    # Remove the previous rule-based baseline so re-runs are idempotent.
    con.execute("DELETE FROM main.aggregate_theme WHERE source = 'rule'")
    con.executemany(
        """
        INSERT INTO main.aggregate_theme (feedback_id, customer_id, created_at, theme, sentiment, text, source)
        VALUES (?, ?, ?, ?, ?, ?, 'rule')
        """,
        enriched,
    )
    return len(enriched)


def print_batch_log(counts: dict[str, int]) -> None:
    print("=" * 60)
    print("APP 2 — EMBEDDING & ENRICHMENT")
    print("=" * 60)
    print(f"  embedded : {counts['embedded']}")
    print(f"  skipped  : {counts['skipped']}  (cached, hash unchanged)")
    print(f"  failed   : {counts['failed']}")
    print("=" * 60)


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        check_inputs(con)
        chunks = build_chunks(con)
        n_enriched = enrich_themes(con)

        client = EmbeddingClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        try:
            counts = sync_embeddings(con, chunks, client)
        finally:
            client.close()
        print_batch_log(counts)
        print(f"  enriched : {n_enriched} feedback rows (rule-based theme + sentiment)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
