"""Tool-level tests: profile, risk rank, revenue-at-risk, themes, etc."""

from __future__ import annotations

import pytest

from apps.mcp import tools


def test_get_customer_profile(mcp_db) -> None:
    res = tools.get_customer_profile("CUST-0001")
    assert res["data"]["customer_name"] == "Alpha"
    assert res["data"]["customer_segment"] == "Mid-Market"
    assert res["data"]["monthly_revenue"] == 1000.0
    assert res["data"]["sessions_change_percent"] == -33.33
    assert res["source_refs"] == ["CUST-0001"]


def test_get_customer_profile_missing(mcp_db) -> None:
    res = tools.get_customer_profile("CUST-9999")
    assert res["data"] is None
    assert "not found" in res["warnings"][0]


def test_rank_customer_risk(mcp_db) -> None:
    res = tools.rank_customer_risk(limit=10)
    # Canceled customers (CUST-0002) are already lost -> excluded from at-risk.
    # CUST-0003 (paused + churn signals) ranks highest among remaining.
    assert res["data"][0]["customer_id"] == "CUST-0003"
    assert res["data"][0]["risk_score"] >= res["data"][1]["risk_score"]
    assert len(res["data"][0]["risk_drivers"]) >= 2
    assert len(res["data"]) == 3  # 4 customers minus the canceled one


def test_rank_customer_risk_segment_filter(mcp_db) -> None:
    res = tools.rank_customer_risk(segment="SMB")
    # CUST-0002 is canceled -> excluded; only CUST-0004 remains in SMB
    assert {r["customer_id"] for r in res["data"]} == {"CUST-0004"}


def test_rank_customer_risk_status_filter(mcp_db) -> None:
    res = tools.rank_customer_risk(status="canceled")
    assert {r["customer_id"] for r in res["data"]} == {"CUST-0002"}


def test_calculate_revenue_at_risk(mcp_db) -> None:
    res = tools.calculate_revenue_at_risk()
    d = res["data"]
    # Canceled (CUST-0002) excluded; only paused CUST-0003 (2500) at risk
    assert d["revenue_at_risk"] == 2500.0
    assert d["at_risk_customers"] == 1
    assert d["at_risk_with_unknown_revenue"] == 0


def test_calculate_revenue_at_risk_segment(mcp_db) -> None:
    res = tools.calculate_revenue_at_risk(segment="SMB")
    # SMB has CUST-0002 (canceled, excluded) + CUST-0004 (active, no signals)
    assert res["data"]["at_risk_customers"] == 0
    assert res["data"]["revenue_at_risk"] == 0.0


def test_get_customer_tickets(mcp_db) -> None:
    res = tools.get_customer_tickets("CUST-0002")
    ids = [r["ticket_id"] for r in res["data"]]
    assert ids == ["TCK-0003", "TCK-0002"]  # ordered by created_at desc
    assert res["source_refs"] == ids


def test_get_customer_tickets_filters(mcp_db) -> None:
    res = tools.get_customer_tickets("CUST-0002", category="bug")
    assert [r["ticket_id"] for r in res["data"]] == ["TCK-0002"]


def test_get_customer_feedback(mcp_db) -> None:
    res = tools.get_customer_feedback("CUST-0001")
    assert res["data"][0]["feedback_id"] == "FDB-0001"
    assert res["data"][0]["theme"] == "search"
    assert res["data"][0]["sentiment"] == "positive"


def test_get_feedback_themes(mcp_db) -> None:
    res = tools.get_feedback_themes(min_count=1)
    themes = {r["theme"]: r["feedback_count"] for r in res["data"]}
    assert themes == {"search": 1, "integrations": 1, "other": 1}


def test_get_feedback_themes_segment(mcp_db) -> None:
    res = tools.get_feedback_themes(segment="SMB")
    assert {r["theme"] for r in res["data"]} == {"integrations"}


def test_get_ticket_breakdown(mcp_db) -> None:
    res = tools.get_ticket_breakdown()
    assert len(res["data"]) >= 3
    by_id = {(r["category"], r["priority"], r["status"]): r["ticket_count"] for r in res["data"]}
    assert by_id[("bug", "urgent", "open")] == 1


def test_get_usage_change_single(mcp_db) -> None:
    res = tools.get_usage_change(customer_id="CUST-0001")
    assert res["data"][0]["sessions_change_percent"] == -33.33


def test_get_usage_change_aggregate(mcp_db) -> None:
    res = tools.get_usage_change()
    d = res["data"]
    assert d["customers"] == 4
    # CUST-0001 (100) + CUST-0003 (200) have last_4wk; CUST-0002/4 NULL -> 0
    assert d["sessions_last_4_weeks"] == 300.0
    assert d["customers_with_sessions_decline"] == 1  # CUST-0001 (-33%); CUST-0002 NULL change not counted


def test_get_usage_trend(mcp_db) -> None:
    res = tools.get_usage_trend("CUST-0001", weeks=8)
    assert len(res["data"]) == 2
    assert res["data"][0]["date"].isoformat() == "2026-06-20"


def test_get_subscription_events(mcp_db) -> None:
    res = tools.get_subscription_events("CUST-0003")
    assert res["data"][0]["event_type"] == "downgrade"
    assert res["data"][0]["revenue_change"] == -500.0


def test_calculate_segment_metrics(mcp_db) -> None:
    res = tools.calculate_segment_metrics()
    assert len(res["data"]) == 4
    smb = [r for r in res["data"] if r["segment"] == "SMB"][0]
    assert smb["cancel_rate"] == 0.5
    assert smb["global_cancel_rate"] == 0.5


def test_list_customers(mcp_db) -> None:
    res = tools.list_customers(segment="SMB", limit=10)
    assert {r["customer_id"] for r in res["data"]} == {"CUST-0002", "CUST-0004"}


def test_list_customers_search(mcp_db) -> None:
    res = tools.list_customers(search="alpha")
    assert [r["customer_id"] for r in res["data"]] == ["CUST-0001"]


def test_memory_roundtrip(mcp_db) -> None:
    tools.write_memory("test_key", "test_value")
    res = tools.read_memory("test_key")
    assert res["data"]["value"] == "test_value"
    assert len(tools.list_memory()["data"]) >= 1
