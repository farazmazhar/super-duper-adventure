"""Quality-report tests: one row per rule, with the expected affected counts."""

from __future__ import annotations

import duckdb


def test_quality_report_has_rows(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute("SELECT count(*) FROM quality_report").fetchone()[0] >= 15


def test_customer_rules_logged(con: duckdb.DuckDBPyConnection) -> None:
    rules = dict(
        con.execute(
            "SELECT rule, count FROM quality_report WHERE table_name = 'customers'"
        ).fetchall()
    )
    assert rules["deduplicate_customers"] == 1
    assert rules["normalize_account_status"] == 2  # ACTIVE + CANCELED
    assert rules["null_revenue_kept"] == 1


def test_ticket_rules_logged(con: duckdb.DuckDBPyConnection) -> None:
    rules = dict(
        con.execute(
            "SELECT rule, count FROM quality_report WHERE table_name = 'support_tickets'"
        ).fetchall()
    )
    assert rules["deduplicate_tickets"] == 1
    assert rules["normalize_category"] == 3  # Bug, High-category rows, General_question (all differ from lowercase)
    assert rules["clip_negative_resolution"] == 1


def test_feedback_rules_logged(con: duckdb.DuckDBPyConnection) -> None:
    rules = dict(
        con.execute(
            "SELECT rule, count FROM quality_report WHERE table_name = 'customer_feedback'"
        ).fetchall()
    )
    assert rules["normalize_rating_scale"] == 1
    assert rules["null_feedback_text_kept"] == 1


def test_usage_rules_logged(con: duckdb.DuckDBPyConnection) -> None:
    rules = dict(
        con.execute(
            "SELECT rule, count FROM quality_report WHERE table_name = 'product_usage'"
        ).fetchall()
    )
    assert rules["null_session_duration_kept"] == 1


def test_sub_events_passthrough_logged(con: duckdb.DuckDBPyConnection) -> None:
    rules = dict(
        con.execute(
            "SELECT rule, count FROM quality_report WHERE table_name = 'subscription_events'"
        ).fetchall()
    )
    assert rules["pass_through"] == 2


def test_every_rule_has_description(con: duckdb.DuckDBPyConnection) -> None:
    empty = con.execute(
        "SELECT count(*) FROM quality_report WHERE description IS NULL OR description = ''"
    ).fetchone()[0]
    assert empty == 0
