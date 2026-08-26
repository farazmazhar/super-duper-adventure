"""Shared fixture: a tiny DuckDB mirroring the App 1 schema for tool tests.

The tools read `DB_PATH` from apps.common.config; we monkeypatch it per-test to
point at this fixture so tests run without touching the real intelligence.duckdb.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from apps.common import config as common_config

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _build_fixture_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    # dimension_customer
    con.execute(
        """
        CREATE TABLE main.dimension_customer (
            customer_id VARCHAR, customer_name VARCHAR, customer_segment VARCHAR,
            country VARCHAR, subscription_plan VARCHAR, monthly_revenue DOUBLE,
            account_created_at DATE, account_status VARCHAR, revenue_imputed BOOLEAN
        )
        """
    )
    con.executemany(
        "INSERT INTO main.dimension_customer VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("CUST-0001", "Alpha", "Mid-Market", "France", "Business", 1000.0, "2023-01-01", "active", False),
            ("CUST-0002", "Beta", "SMB", "Germany", "Team", 500.0, "2023-02-01", "canceled", False),
            ("CUST-0003", "Gamma", "Enterprise", "USA", "Enterprise", 2500.0, "2023-03-01", "paused", False),
            ("CUST-0004", "Delta", "SMB", "USA", "Free", None, "2023-04-01", "active", False),
        ],
    )
    # aggregate_customer_features (subset of columns used by the tools)
    con.execute(
        """
        CREATE TABLE main.aggregate_customer_features (
            customer_id VARCHAR, account_status VARCHAR, customer_segment VARCHAR,
            subscription_plan VARCHAR, monthly_revenue DOUBLE, churn_proxy BOOLEAN,
            ticket_count INT, tickets_open INT, tickets_urgent INT,
            tickets_billing INT, tickets_bug INT, average_resolution_time_hours DOUBLE,
            average_satisfaction_score DOUBLE, sessions_last_4_weeks DOUBLE,
            sessions_previous_4_weeks DOUBLE, sessions_change_percent DOUBLE,
            active_users_last_4_weeks DOUBLE, active_users_previous_4_weeks DOUBLE,
            active_users_change_percent DOUBLE, errors_total INT,
            average_session_duration DOUBLE, feedback_count INT, average_rating DOUBLE,
            rating_last_half DOUBLE, rating_prior_half DOUBLE, cancellations INT,
            downgrades INT, upgrades INT, renewals INT, revenue_change_sum DOUBLE,
            last_event_date DATE, last_event_type VARCHAR
        )
        """
    )
    con.executemany(
        """
        INSERT INTO main.aggregate_customer_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            # CUST-0001: active, slight usage decline, no open tickets
            ("CUST-0001", "active", "Mid-Market", "Business", 1000.0, False, 3, 0, 0, 1, 0, 20.0, 4.2, 100.0, 150.0, -33.33, 10.0, 12.0, -16.67, 5, 8.0, 2, 4.0, 4.5, 3.5, 0, 0, 1, 1, 100.0, "2026-07-01", "upgrade"),
            # CUST-0002: canceled, no usage last 4wk, urgent tickets
            ("CUST-0002", "canceled", "SMB", "Team", 500.0, True, 5, 2, 1, 0, 1, 30.0, 2.0, None, 80.0, None, None, 10.0, None, 12, 6.0, 1, 2.0, None, None, 1, 0, 0, 0, -200.0, "2026-06-15", "cancellation"),
            # CUST-0003: paused, moderate usage decline
            ("CUST-0003", "paused", "Enterprise", "Enterprise", 2500.0, True, 8, 1, 2, 3, 2, 15.0, 3.0, 200.0, 100.0, 100.0, 20.0, 12.0, 66.67, 3, 7.0, 3, 3.0, 2.5, 3.5, 0, 1, 0, 0, -500.0, "2026-07-05", "downgrade"),
            # CUST-0004: active, no tickets, no usage, NULL revenue
            ("CUST-0004", "active", "SMB", "Free", None, False, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None, 0, None, 0, None, None, None, 0, 0, 0, 0, None, None, None),
        ],
    )
    # fact_ticket / fact_feedback / aggregate_theme / fact_subscription_event
    con.execute(
        """
        CREATE TABLE main.fact_ticket (
            ticket_id VARCHAR, customer_id VARCHAR, created_at TIMESTAMP, subject VARCHAR,
            message VARCHAR, category VARCHAR, priority VARCHAR,
            resolution_time_hours DOUBLE, status VARCHAR, satisfaction_score DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO main.fact_ticket VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("TCK-0001", "CUST-0001", "2026-05-01 10:00:00", "Bug", "It crashes", "bug", "high", 5.0, "resolved", 4.0),
            ("TCK-0002", "CUST-0002", "2026-05-02 11:00:00", "Urgent", "Outage", "bug", "urgent", None, "open", None),
            ("TCK-0003", "CUST-0002", "2026-05-03 12:00:00", "Question", "How to", "general_question", "low", None, "open", None),
            ("TCK-0004", "CUST-0003", "2026-05-04 13:00:00", "Billing", "Invoice", "billing", "urgent", 10.0, "resolved", 2.0),
        ],
    )
    con.execute(
        """
        CREATE TABLE main.fact_feedback (
            feedback_id VARCHAR, customer_id VARCHAR, created_at TIMESTAMP,
            feedback_text VARCHAR, feedback_source VARCHAR, rating DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO main.fact_feedback VALUES (?,?,?,?,?,?)",
        [
            ("FDB-0001", "CUST-0001", "2026-05-10 09:00:00", "Love search", "support_chat", 5.0),
            ("FDB-0002", "CUST-0002", "2026-05-11 09:00:00", "Bad api", "email", 2.0),
            ("FDB-0003", "CUST-0003", "2026-05-12 09:00:00", "Ok", "app_store_review", 3.0),
        ],
    )
    con.execute(
        """
        CREATE TABLE main.aggregate_theme (
            feedback_id VARCHAR, customer_id VARCHAR, created_at TIMESTAMP,
            text VARCHAR, theme VARCHAR, sentiment VARCHAR, source VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO main.aggregate_theme VALUES (?,?,?,?,?,?,?)",
        [
            ("FDB-0001", "CUST-0001", "2026-05-10 09:00:00", "Love search", "search", "positive", "rule"),
            ("FDB-0002", "CUST-0002", "2026-05-11 09:00:00", "Bad api", "integrations", "negative", "rule"),
            ("FDB-0003", "CUST-0003", "2026-05-12 09:00:00", "Ok", "other", "neutral", "rule"),
        ],
    )
    con.execute(
        """
        CREATE TABLE main.fact_subscription_event (
            customer_id VARCHAR, event_date DATE, event_type VARCHAR,
            previous_plan VARCHAR, new_plan VARCHAR, revenue_change DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO main.fact_subscription_event VALUES (?,?,?,?,?,?)",
        [
            ("CUST-0001", "2026-07-01", "upgrade", "Business", "Enterprise", 500.0),
            ("CUST-0002", "2026-06-15", "cancellation", "Team", "Free", -200.0),
            ("CUST-0003", "2026-07-05", "downgrade", "Enterprise", "Business", -500.0),
        ],
    )
    con.execute(
        """
        CREATE TABLE main.fact_usage (
            customer_id VARCHAR, date DATE, active_users INT, sessions INT,
            feature_usage VARCHAR, errors INT, average_session_duration DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO main.fact_usage VALUES (?,?,?,?,?,?,?)",
        [
            ("CUST-0001", "2026-06-20", 10, 50, "search", 1, 8.0),
            ("CUST-0001", "2026-06-25", 12, 50, "search", 2, 8.5),
            ("CUST-0003", "2026-06-20", 20, 100, "api", 0, 7.0),
            ("CUST-0003", "2026-06-25", 22, 100, "api", 1, 7.5),
        ],
    )
    con.execute(
        """
        CREATE TABLE main.aggregate_segment_metrics (
            segment VARCHAR, plan VARCHAR, customers INT, revenue DOUBLE,
            revenue_customers INT, customers_with_tickets INT, ticket_count INT,
            complaint_rate DOUBLE, average_resolution_time_hours DOUBLE,
            average_satisfaction_score DOUBLE, cancel_rate DOUBLE,
            global_customers INT, global_revenue DOUBLE, global_average_resolution DOUBLE,
            global_average_satisfaction DOUBLE, global_churn INT, global_cancel_rate DOUBLE
        )
        """
    )
    con.executemany(
        """
        INSERT INTO main.aggregate_segment_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            ("SMB", None, 2, 500.0, 1, 1, 2, 0.5, 30.0, 2.0, 0.5, 4, 4000.0, 20.0, 3.2, 2, 0.5),
            ("Mid-Market", None, 1, 1000.0, 1, 1, 3, 0.0, 20.0, 4.2, 0.0, 4, 4000.0, 20.0, 3.2, 2, 0.5),
            ("Enterprise", None, 1, 2500.0, 1, 1, 8, 1.0, 15.0, 3.0, 1.0, 4, 4000.0, 20.0, 3.2, 2, 0.5),
            (None, "Business", 1, 1000.0, 1, 1, 3, 0.0, 20.0, 4.2, 0.0, 4, 4000.0, 20.0, 3.2, 2, 0.5),
        ],
    )
    con.execute("CREATE SCHEMA IF NOT EXISTS agent")
    con.execute(
        "CREATE TABLE IF NOT EXISTS agent.agent_memory (key VARCHAR PRIMARY KEY, value VARCHAR, updated_at TIMESTAMP DEFAULT now())"
    )
    con.execute("INSERT INTO agent.agent_memory VALUES ('user_pref', 'wants dollar impact', now())")

    # vector schema (App 2 output) — a few embedding rows for retrieval tests
    con.execute("CREATE SCHEMA IF NOT EXISTS vector")
    con.execute(
        """
        CREATE TABLE vector.embeddings (
            record_type VARCHAR, record_id VARCHAR, customer_id VARCHAR,
            created_at TIMESTAMP, text VARCHAR, metadata JSON,
            embedding FLOAT[], source VARCHAR, model VARCHAR
        )
        """
    )
    con.executemany(
        """
        INSERT INTO vector.embeddings VALUES (?,?,?,?,?,?,?,?,?)
        """,
        [
            ("feedback", "FDB-0001", "CUST-0001", "2026-05-10 09:00:00", "Love the search feature", '{}', [1.0, 0.0, 0.0], "voyage", "voyageai/voyage-4-lite"),
            ("ticket", "TCK-0002", "CUST-0002", "2026-05-02 11:00:00", "Outage in production", '{}', [0.0, 1.0, 0.0], "voyage", "voyageai/voyage-4-lite"),
            ("feedback", "FDB-0002", "CUST-0002", "2026-05-11 09:00:00", "Bad api experience", '{}', [0.0, 0.0, 1.0], "voyage", "voyageai/voyage-4-lite"),
        ],
    )
    con.close()


@pytest.fixture(scope="session")
def mcp_db(tmp_path_factory) -> Path:
    """Build the fixture DB and point apps.common.config.DB_PATH at it."""
    db_path = tmp_path_factory.mktemp("db") / "intelligence.duckdb"
    _build_fixture_db(db_path)
    common_config.DB_PATH = db_path
    return db_path
