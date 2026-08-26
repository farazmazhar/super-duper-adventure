"""App 1 — Data Cleansing orchestrator.

Thin wrapper: connect to DuckDB, render and execute pipeline.sql, print the
quality report. All logic lives in pipeline.sql (SQL-first, auditable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# Make the repo root importable (repo-root `apps/` package) regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import DATA_DIR, DB_PATH, RAW_FILES  # noqa: E402

PIPELINE_SQL = Path(__file__).resolve().parent / "pipeline.sql"


def check_inputs() -> None:
    """Fail fast with a clear message if any raw CSV is missing."""
    missing = [name for name in RAW_FILES.values() if not (DATA_DIR / name).is_file()]
    if missing:
        sys.exit(f"Missing raw input file(s) in {DATA_DIR}: {', '.join(missing)}")


def run_pipeline(con: duckdb.DuckDBPyConnection) -> None:
    """Execute the full pipeline (staging -> clean -> features -> quality report)."""
    sql = PIPELINE_SQL.read_text().replace("{{data_dir}}", str(DATA_DIR))
    con.execute(sql)


def print_quality_report(con: duckdb.DuckDBPyConnection) -> None:
    """Print a readable, grouped summary of quality_report."""
    rows = con.execute(
        "SELECT rule, table_name, description, count FROM quality_report ORDER BY table_name, rule"
    ).fetchall()
    if not rows:
        print("(quality_report is empty)")
        return

    print("=" * 78)
    print("QUALITY REPORT — App 1 Data Cleansing")
    print("=" * 78)
    current_table = None
    for rule, table_name, description, count in rows:
        if table_name != current_table:
            print(f"\n[{table_name}]")
            current_table = table_name
        print(f"  - {rule:<28} count={count:<6} {description}")
    print("=" * 78)


def main() -> None:
    check_inputs()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        run_pipeline(con)
        print_quality_report(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
