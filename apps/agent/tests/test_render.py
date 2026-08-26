"""Render-hint builder + CLI renderer tests (pure functions, no network)."""

from __future__ import annotations

from apps.agent.cli_render import render_answer_text, render_chart, render_table
from apps.agent.render import hint_from_results


def _results(**data):
    return {k: {"data": v, "source_refs": [], "warnings": []} for k, v in data.items()}


# ---------------------------------------------------------------------------
# hint_from_results
# ---------------------------------------------------------------------------
def test_trend_hint_is_chart() -> None:
    hint = hint_from_results(
        "trend",
        _results(
            get_usage_trend=[
                {"date": "2026-06-20", "sessions": 50, "active_users": 10},
                {"date": "2026-06-21", "sessions": 55, "active_users": 11},
            ]
        ),
        "how has usage changed?",
    )
    assert hint["kind"] == "chart"
    assert hint["spec"]["type"] == "line"
    assert len(hint["data"]) == 2


def test_analytics_hint_is_table_with_cards() -> None:
    hint = hint_from_results(
        "analytics",
        _results(
            rank_customer_risk=[
                {"customer_id": "CUST-0009", "risk_score": 70.0, "account_status": "canceled", "risk_drivers": ["churn"]},
            ],
            calculate_revenue_at_risk={"revenue_at_risk": 50000.0, "at_risk_customers": 29},
        ),
        "which customers are at risk?",
    )
    assert hint["kind"] == "table"
    assert hint["data"][0]["customer_id"] == "CUST-0009"


def test_customer_hint_is_cards() -> None:
    hint = hint_from_results(
        "customer",
        _results(
            get_customer_profile={
                "customer_id": "CUST-0001", "customer_name": "Alpha", "account_status": "active",
                "subscription_plan": "Business", "monthly_revenue": 1000.0, "sessions_change_percent": -20.0,
            },
            get_customer_tickets=[{"ticket_id": "TCK-1", "created_at": "2026-05-01", "category": "bug", "priority": "high", "status": "open"}],
        ),
        "who is CUST-0001?",
    )
    assert hint["kind"] == "table"  # tickets present -> table
    assert hint["data"][0]["ticket_id"] == "TCK-1"


def test_themes_hint_is_chart() -> None:
    hint = hint_from_results(
        "themes",
        _results(get_feedback_themes=[{"theme": "billing", "feedback_count": 10}]),
        "top themes?",
    )
    assert hint["kind"] == "chart"
    assert hint["spec"]["type"] == "bar"


# ---------------------------------------------------------------------------
# CLI renderers
# ---------------------------------------------------------------------------
def test_render_table_aligns() -> None:
    out = render_table([{"a": 1, "b": "x"}, {"a": 222, "b": "yy"}], title="T")
    assert "T" in out
    assert "a" in out and "b" in out
    assert "222" in out


def test_render_chart_bar() -> None:
    out = render_chart({"kind": "chart", "data": [{"theme": "billing", "count": 10}], "spec": {"type": "bar", "x": "theme", "y": "count"}})
    assert "█" in out
    assert "billing" in out


def test_render_chart_empty() -> None:
    out = render_chart({"kind": "chart", "data": [], "spec": {}})
    assert "no chart data" in out


def test_render_answer_text_sections() -> None:
    answer = {
        "summary": "Top theme is billing",
        "confidence": "high",
        "facts": [{"heading": "Facts", "content": "billing has 10 mentions\nreporting has 5", "citations": ["theme:billing"]}],
        "interpretation": [{"heading": "Interpretation", "content": "billing dominates", "citations": []}],
        "recommendation": [],
        "other_sections": [],
        "render_hint": {"kind": "markdown", "payload": {"kind": "markdown"}},
    }
    out = render_answer_text(answer)
    assert "Top theme is billing" in out
    assert "Facts" in out
    assert "billing has 10 mentions" in out
    assert "theme:billing" in out
