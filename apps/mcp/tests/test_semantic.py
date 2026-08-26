"""Semantic-layer tests: catalog validation, SQL translation, read-only execution.

Runs against the shared fixture DB (apps/mcp/tests/conftest.py) which mirrors
App 1's schema, so execution is deterministic and offline.
"""

from __future__ import annotations

import pytest

from apps.mcp.semantic import (
    ENTITIES,
    METRICS,
    SemanticQueryError,
    execute_semantic_query,
    translate_semantic_query,
    validate_semantic_query,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_valid_query_passes() -> None:
    q = {"metric": "count", "of": "ticket", "dimensions": ["category"]}
    validate_semantic_query(q)  # no raise


def test_unknown_metric_rejected() -> None:
    res = execute_semantic_query({"metric": "drop_table", "of": "ticket"})
    assert res["data"] == []
    assert any("Unknown metric" in w for w in res["warnings"])


def test_unknown_entity_rejected() -> None:
    res = execute_semantic_query({"metric": "count", "of": "customers2"})
    assert any("Unknown entity" in w for w in res["warnings"])


def test_unknown_dimension_rejected() -> None:
    res = execute_semantic_query(
        {"metric": "count", "of": "ticket", "dimensions": ["category; drop"]}
    )
    assert any("Unknown dimension" in w for w in res["warnings"])


def test_unknown_filter_rejected() -> None:
    res = execute_semantic_query(
        {"metric": "count", "of": "ticket", "filters": {"evil_col": "x"}}
    )
    assert any("Unknown filter" in w for w in res["warnings"])


def test_sum_requires_dimension() -> None:
    res = execute_semantic_query({"metric": "sum", "of": "ticket"})
    assert any("requires 'of_dimension'" in w for w in res["warnings"])


def test_sum_invalid_dimension_rejected() -> None:
    res = execute_semantic_query(
        {"metric": "sum", "of": "ticket", "of_dimension": "customer_segment"}
    )
    assert any("not numeric/valid" in w for w in res["warnings"])


def test_bad_limit_rejected() -> None:
    res = execute_semantic_query({"metric": "count", "of": "ticket", "limit": 99999})
    assert any("limit" in w for w in res["warnings"])


def test_bad_time_range_rejected() -> None:
    res = execute_semantic_query(
        {"metric": "count", "of": "ticket", "time_range": {"bogus": "x"}}
    )
    assert any("time_range" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def test_translate_count_by_category() -> None:
    sql, params = translate_semantic_query(
        {"metric": "count", "of": "ticket", "dimensions": ["category"]}
    )
    assert "main.fact_ticket" in sql
    assert "GROUP BY category" in sql
    # LIMIT is limit+1 so execute can detect truncation (fetch limit+1, trim)
    assert "LIMIT 101" in sql
    assert params == []


def test_translate_sum_with_join_and_filter() -> None:
    q = {
        "metric": "sum",
        "of": "ticket",
        "of_dimension": "satisfaction_score",
        "filters": {"customer_segment": "Enterprise"},
        "limit": 10,
    }
    sql, params = translate_semantic_query(q)
    assert "JOIN main.dimension_customer" in sql
    assert "customer_segment = ?" in sql
    assert params == ["Enterprise"]


def test_translate_time_range() -> None:
    q = {
        "metric": "count",
        "of": "ticket",
        "time_range": {"from": "2026-05-01", "to": "2026-06-30"},
    }
    sql, params = translate_semantic_query(q)
    assert "created_at >= ?" in sql
    assert "created_at <= ?" in sql
    assert params == ["2026-05-01", "2026-06-30"]


def test_translate_satisfaction_avg() -> None:
    sql, _ = translate_semantic_query({"metric": "satisfaction_avg", "of": "ticket"})
    assert "avg(satisfaction_score)" in sql


# ---------------------------------------------------------------------------
# Execution (against the fixture DB)
# ---------------------------------------------------------------------------
def test_execute_count_tickets(mcp_db) -> None:
    res = execute_semantic_query({"metric": "count", "of": "ticket"})
    assert res["data"][0]["value"] == 4  # 4 tickets in the fixture
    assert "value" in res["columns"]


def test_execute_count_by_category(mcp_db) -> None:
    res = execute_semantic_query(
        {"metric": "count", "of": "ticket", "dimensions": ["category"]}
    )
    by_cat = {r["category"]: r["value"] for r in res["data"]}
    assert by_cat == {"bug": 2, "general_question": 1, "billing": 1}


def test_execute_sum_revenue_by_segment(mcp_db) -> None:
    res = execute_semantic_query(
        {
            "metric": "sum",
            "of": "customer",
            "of_dimension": "monthly_revenue",
            "dimensions": ["customer_segment"],
        }
    )
    by_seg = {r["customer_segment"]: r["value"] for r in res["data"]}
    # fixture: SMB 500 (CUST-0002) + Free null (CUST-0004 excluded), Mid-Market 1000, Enterprise 2500
    assert by_seg["Mid-Market"] == 1000.0
    assert by_seg["Enterprise"] == 2500.0


def test_execute_filter_segment(mcp_db) -> None:
    res = execute_semantic_query(
        {"metric": "count", "of": "customer", "filters": {"customer_segment": "SMB"}}
    )
    assert res["data"][0]["value"] == 2


def test_execute_time_range(mcp_db) -> None:
    res = execute_semantic_query(
        {
            "metric": "count",
            "of": "feedback",
            "time_range": {"from": "2026-05-11", "to": "2026-05-12"},
        }
    )
    assert res["data"][0]["value"] == 1  # only FDB-0002/0003 on those dates? FDB-0002 = 05-11


def test_execute_revenue_at_risk_metric(mcp_db) -> None:
    res = execute_semantic_query({"metric": "revenue_at_risk", "of": "customer_features"})
    # fixture: canceled CUST-0002 excluded (already lost); paused CUST-0003 = 2500
    assert res["data"][0]["value"] == 2500.0


def test_no_rows_warning(mcp_db) -> None:
    # count with a non-matching filter -> 0 (aggregate), not an error
    res = execute_semantic_query(
        {"metric": "count", "of": "ticket", "filters": {"customer_segment": "Nope"}}
    )
    assert res["data"] == [{"value": 0}]
    # a grouped query with no matches -> empty data + warning
    res2 = execute_semantic_query(
        {
            "metric": "count", "of": "ticket", "dimensions": ["category"],
            "filters": {"customer_segment": "Nope"},
        }
    )
    assert res2["data"] == []
    assert any("no rows" in w for w in res2["warnings"])


def test_catalog_sanity() -> None:
    # every entity has a table; every metric is defined
    assert "ticket" in ENTITIES
    assert "count" in METRICS
    assert "revenue_at_risk" in METRICS
