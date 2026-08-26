"""App 7 — read-only DuckDB access for the FE.

The FE must NEVER open a read-write connection (spec §DuckDB access). Every
read goes through `get_connection()` which opens `read_only=True`. The
connection is cached per process (plain module cache — no Streamlit).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.common.config import DB_PATH  # noqa: E402

_CACHED_CON: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Read-only connection, cached per process."""
    global _CACHED_CON
    if _CACHED_CON is None:
        _CACHED_CON = duckdb.connect(str(DB_PATH), read_only=True)
    return _CACHED_CON


def close() -> None:
    global _CACHED_CON
    if _CACHED_CON is not None:
        try:
            _CACHED_CON.close()
        except Exception:
            pass
        _CACHED_CON = None


def table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    return (
        con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
            [schema, table],
        ).fetchone()[0]
        > 0
    )


def query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Run a read-only SELECT and return rows as dicts."""
    con = get_connection()
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
