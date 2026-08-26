"""End-to-end cleansing tests on a tiny fixture.

Covers: dedup (customers keep-last, tickets exact), case normalization,
NULL preservation, rating clipping, negative resolution -> NULL, date casts,
usage-trend windows, and aggregate row counts.
"""

from __future__ import annotations

import duckdb


def test_customer_dedup_keeps_last(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT customer_id, monthly_revenue FROM dimension_customer ORDER BY customer_id"
    ).fetchall()
    # CUST-0001 deduped: keep LAST (1100.00), CUST-0002 revenue NULL kept
    assert rows == [
        ("CUST-0001", 1100.0),
        ("CUST-0002", None),
        ("CUST-0003", 2500.0),
    ]


def test_customer_status_normalized(con: duckdb.DuckDBPyConnection) -> None:
    statuses = dict(
        con.execute("SELECT customer_id, account_status FROM dimension_customer").fetchall()
    )
    assert statuses == {
        "CUST-0001": "active",
        "CUST-0002": "canceled",
        "CUST-0003": "paused",
    }


def test_null_revenue_preserved(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute(
        "SELECT count(*) FROM dimension_customer WHERE monthly_revenue IS NULL"
    ).fetchone()[0] == 1
    assert con.execute(
        "SELECT count(DISTINCT revenue_imputed) FROM dimension_customer"
    ).fetchone()[0] == 1  # all False


def test_ticket_exact_dup_dropped(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute(
        "SELECT count(*) FROM fact_ticket WHERE ticket_id = 'TCK-0001'"
    ).fetchone()[0] == 1


def test_ticket_category_normalized(con: duckdb.DuckDBPyConnection) -> None:
    cats = con.execute(
        "SELECT DISTINCT category FROM fact_ticket ORDER BY category"
    ).fetchall()
    assert cats == [("billing",), ("bug",), ("general_question",)]


def test_negative_resolution_to_null(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute(
        "SELECT count(*) FROM fact_ticket WHERE resolution_time_hours < 0"
    ).fetchone()[0] == 0
    # TCK-0003 was -5.0 -> NULL
    assert con.execute(
        "SELECT resolution_time_hours FROM fact_ticket WHERE ticket_id = 'TCK-0003'"
    ).fetchone()[0] is None


def test_null_resolution_and_satisfaction_kept(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute(
        "SELECT count(*) FROM fact_ticket WHERE resolution_time_hours IS NULL"
    ).fetchone()[0] == 2  # TCK-0002 (unresolved) + TCK-0003 (clipped)
    assert con.execute(
        "SELECT count(*) FROM fact_ticket WHERE satisfaction_score IS NULL"
    ).fetchone()[0] == 1  # only the unresolved ticket


def test_rating_out_of_range_normalized(con: duckdb.DuckDBPyConnection) -> None:
    # FDB-0002 was rating 8 (10-point scale) -> normalized to 4.0
    assert con.execute(
        "SELECT rating FROM fact_feedback WHERE feedback_id = 'FDB-0002'"
    ).fetchone()[0] == 4.0
    # in-range ratings pass through unchanged
    assert con.execute(
        "SELECT rating FROM fact_feedback WHERE feedback_id = 'FDB-0001'"
    ).fetchone()[0] == 5
    assert con.execute(
        "SELECT count(*) FROM fact_feedback WHERE rating > 5"
    ).fetchone()[0] == 0


def test_null_feedback_text_kept(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute(
        "SELECT count(*) FROM fact_feedback WHERE feedback_text IS NULL"
    ).fetchone()[0] == 1


def test_date_casts(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute(
        "SELECT typeof(account_created_at) FROM dimension_customer LIMIT 1"
    ).fetchone()[0] == "DATE"
    assert con.execute(
        "SELECT typeof(created_at) FROM fact_ticket LIMIT 1"
    ).fetchone()[0] == "TIMESTAMP"
    assert con.execute(
        "SELECT typeof(date) FROM fact_usage LIMIT 1"
    ).fetchone()[0] == "DATE"
    assert con.execute(
        "SELECT typeof(event_date) FROM fact_subscription_event LIMIT 1"
    ).fetchone()[0] == "DATE"


def test_usage_trend_windows(con: duckdb.DuckDBPyConnection) -> None:
    # CUST-0001: sessions last_4_weeks (>= 2026-06-17) = 150 + 200 = 350
    #            previous_4_weeks (2026-05-20..06-17) = 100
    row = con.execute(
        """SELECT sessions_last_4_weeks, sessions_previous_4_weeks, sessions_change_percent
           FROM aggregate_customer_features WHERE customer_id = 'CUST-0001'"""
    ).fetchone()
    assert row[0] == 350
    assert row[1] == 100
    assert row[2] == 250.0  # (350-100)/100*100


def test_null_session_duration_kept(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute(
        "SELECT count(*) FROM fact_usage WHERE average_session_duration IS NULL"
    ).fetchone()[0] == 1


def test_feature_table_row_counts(con: duckdb.DuckDBPyConnection) -> None:
    assert con.execute("SELECT count(*) FROM dimension_customer").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM fact_ticket").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM fact_feedback").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM fact_usage").fetchone()[0] == 4
    assert con.execute("SELECT count(*) FROM fact_subscription_event").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM aggregate_customer_features").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM aggregate_theme").fetchone()[0] == 3


def test_aggregate_customer_features_churn_proxy(con: duckdb.DuckDBPyConnection) -> None:
    rows = {
        cid: (churn, canc)
        for cid, churn, canc in con.execute(
            "SELECT customer_id, churn_proxy, cancellations FROM aggregate_customer_features"
        ).fetchall()
    }
    assert rows["CUST-0001"] == (False, 0)
    assert rows["CUST-0002"] == (True, 1)  # canceled + cancellation event


def test_aggregate_theme_seeded(con: duckdb.DuckDBPyConnection) -> None:
    # "Love the search feature" -> search theme, positive sentiment, source=rule
    row = con.execute(
        """SELECT theme, sentiment, source FROM aggregate_theme WHERE feedback_id = 'FDB-0001'"""
    ).fetchone()
    assert row == ("search", "positive", "rule")
