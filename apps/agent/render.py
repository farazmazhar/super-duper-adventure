"""App 3 — build render_hint payloads from gathered tool results.

The `answer` node uses these to attach structured visuals (charts, tables,
cards) to every answer, so both the FE (App 7) and CLI render them. The FE
consumes `{kind, data, spec}`; the CLI renders the same payload as ASCII.

Mapping (spec §answer): trend -> chart, comparisons/risk -> table,
customer drill-down -> cards, recommendations -> qa, else markdown.
"""

from __future__ import annotations

from typing import Any


def hint_from_results(
    routed_node: str,
    tool_results: dict[str, dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    """Build a render_hint payload from the routed node's tool results."""
    data = {k: (v.get("data") if isinstance(v, dict) else None) for k, v in tool_results.items()}

    if routed_node == "trend":
        return _trend_hint(data)
    if routed_node == "customer":
        return _customer_hint(data)
    if routed_node == "analytics":
        return _analytics_hint(data)
    if routed_node == "themes":
        return _themes_hint(data)
    if routed_node in ("rag", "general"):
        return _markdown_hint(question)
    return {"kind": "markdown", "data": None, "spec": {}, "text": ""}


def _trend_hint(data: dict[str, Any]) -> dict[str, Any]:
    """Usage trend -> line/bar chart of sessions over time (per customer)."""
    trend = data.get("get_usage_trend") or []
    if trend:
        rows = [{"date": str(r.get("date", "")), "sessions": r.get("sessions") or 0, "active_users": r.get("active_users") or 0} for r in trend]
        return {
            "kind": "chart",
            "data": rows,
            "spec": {"type": "line", "x": "date", "y": "sessions", "title": "Sessions over time"},
        }
    chg = data.get("get_usage_change")
    if chg and isinstance(chg, list) and chg:
        rows = [
            {
                "customer_id": r.get("customer_id", ""),
                "last_4_weeks": r.get("sessions_last_4_weeks") or 0,
                "previous_4_weeks": r.get("sessions_previous_4_weeks") or 0,
                "change_pct": round(r.get("sessions_change_percent") or 0, 1),
            }
            for r in chg[:10]
        ]
        return {
            "kind": "table",
            "data": rows,
            "spec": {"title": "Usage change (last 4 weeks vs previous 4 weeks)"},
        }
    return {"kind": "markdown", "data": None, "spec": {}, "text": "No usage trend data available."}


def _analytics_hint(data: dict[str, Any]) -> dict[str, Any]:
    """Risk ranking / revenue at risk -> tables + KPI cards."""
    ranked = data.get("rank_customer_risk") or []
    rev = data.get("calculate_revenue_at_risk")
    segments = data.get("calculate_segment_metrics") or []

    kpis = []
    if rev and isinstance(rev, dict):
        kpis.append({"label": "Revenue at risk", "value": f"${rev.get('revenue_at_risk', 0):,.0f}"})
        kpis.append({"label": "At-risk customers", "value": rev.get("at_risk_customers", 0)})
    payload: dict[str, Any] = {"kind": "cards", "data": kpis, "spec": {}}
    if kpis and (ranked or segments):
        # primary visual: risk table (most decision-relevant)
        payload["kind"] = "table"
        payload["data"] = _risk_rows(ranked)
        payload["spec"] = {"title": "Customers by risk score"}
    return payload


def _risk_rows(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "customer_id": r.get("customer_id", ""),
            "risk_score": r.get("risk_score"),
            "status": r.get("account_status", ""),
            "drivers": ", ".join(r.get("risk_drivers") or [])[:80],
        }
        for r in (ranked or [])[:10]
    ]


def _customer_hint(data: dict[str, Any]) -> dict[str, Any]:
    """Customer drill-down -> KPI cards + tickets table."""
    profile = data.get("get_customer_profile")
    kpis = []
    if profile and isinstance(profile, dict):
        kpis = [
            {"label": "Status", "value": profile.get("account_status", "")},
            {"label": "Plan", "value": profile.get("subscription_plan", "")},
            {"label": "Revenue", "value": f"${profile.get('monthly_revenue') or 0:,.0f}"},
            {"label": "Sessions Δ", "value": f"{profile.get('sessions_change_percent') or 0:.0f}%"},
        ]
    tickets = data.get("get_customer_tickets") or []
    payload: dict[str, Any] = {"kind": "cards", "data": kpis, "spec": {}}
    if kpis and tickets:
        payload["kind"] = "table"
        payload["data"] = [
            {
                "ticket_id": r.get("ticket_id", ""),
                "created_at": str(r.get("created_at", ""))[:10],
                "category": r.get("category", ""),
                "priority": r.get("priority", ""),
                "status": r.get("status", ""),
            }
            for r in tickets[:10]
        ]
        payload["spec"] = {"title": "Recent tickets"}
    return payload


def _themes_hint(data: dict[str, Any]) -> dict[str, Any]:
    """Feedback themes -> bar chart of theme counts."""
    themes = data.get("get_feedback_themes") or []
    if themes:
        return {
            "kind": "chart",
            "data": [{"theme": r.get("theme", ""), "count": r.get("feedback_count") or 0} for r in themes[:10]],
            "spec": {"type": "bar", "x": "theme", "y": "count", "title": "Feedback themes by volume"},
        }
    return {"kind": "markdown", "data": None, "spec": {}, "text": "No theme data available."}


def _markdown_hint(question: str) -> dict[str, Any]:
    return {"kind": "markdown", "data": None, "spec": {}, "text": question}
