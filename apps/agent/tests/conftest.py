"""Agent tests: use a fake MCP client (no real subprocess/server).

The graph only needs `mcp.call_tools(...)` / `mcp.call_tool(...)`; we inject a
stub returning canned tool payloads so tests are fast and deterministic.
"""

from __future__ import annotations

import pytest

from apps.agent.graph import AgentGraph


class FakeMcpClient:
    """Canned responses keyed by tool name (optionally per call order)."""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        self.calls.append((name, arguments or {}))
        return self.responses.get(name, {"data": None, "source_refs": [], "warnings": []})

    def call_tools(self, calls: list[tuple[str, dict]]) -> dict[str, dict]:
        out = {}
        for name, args in calls:
            out[name] = self.call_tool(name, args)
        return out


@pytest.fixture
def fake_mcp() -> FakeMcpClient:
    return FakeMcpClient(
        {
            "get_customer_profile": {
                "data": {
                    "customer_id": "CUST-0001", "customer_name": "VertexPath A",
                    "customer_segment": "Mid-Market", "subscription_plan": "Business",
                    "account_status": "active", "monthly_revenue": 3198.81,
                    "sessions_change_percent": -33.3, "tickets_open": 1,
                },
                "source_refs": ["CUST-0001"], "warnings": [],
            },
            "get_feedback_themes": {
                "data": [
                    {"theme": "integrations", "feedback_count": 61, "positive": 10, "negative": 2, "neutral": 49},
                    {"theme": "reporting", "feedback_count": 40, "positive": 5, "negative": 3, "neutral": 32},
                ],
                "source_refs": ["theme:integrations", "theme:reporting"], "warnings": [],
            },
            "rank_customer_risk": {
                "data": [
                    {"customer_id": "CUST-0009", "risk_score": 70.0, "risk_drivers": ["status=canceled", "no usage"]},
                    {"customer_id": "CUST-0018", "risk_score": 70.0, "risk_drivers": ["sessions -40%"]},
                ],
                "source_refs": ["CUST-0009", "CUST-0018"], "warnings": [],
            },
            "calculate_revenue_at_risk": {
                "data": {"revenue_at_risk": 53387.14, "at_risk_customers": 29, "at_risk_with_unknown_revenue": 3},
                "source_refs": [], "warnings": [],
            },
            "retrieve_sources": {
                "data": [
                    {"record_type": "feedback", "record_id": "FDB-00042", "customer_id": "CUST-0001",
                     "created_at": "2026-05-19", "text": "Mixed feelings about search", "score": 0.82},
                ],
                "source_refs": ["feedback:FDB-00042"], "warnings": [],
            },
            "list_customers": {
                "data": [
                    {"customer_id": "CUST-0001", "customer_name": "VertexPath A", "customer_segment": "Mid-Market",
                     "subscription_plan": "Business", "account_status": "active", "monthly_revenue": 3198.81},
                    {"customer_id": "CUST-0002", "customer_name": "AtlasDynamics B", "customer_segment": "Enterprise",
                     "subscription_plan": "Enterprise", "account_status": "active", "monthly_revenue": 4667.44},
                ],
                "source_refs": ["CUST-0001", "CUST-0002"], "warnings": [],
            },
            "get_usage_trend": {
                "data": [
                    {"date": "2026-06-20", "sessions": 50, "active_users": 10, "errors": 1},
                    {"date": "2026-06-25", "sessions": 55, "active_users": 11, "errors": 0},
                ],
                "source_refs": ["CUST-0001:2026-06-20"], "warnings": [],
            },
        }
    )


@pytest.fixture
def graph(fake_mcp: FakeMcpClient) -> AgentGraph:
    """Graph with the fake MCP client and NO LLM (deterministic reason fallback)."""
    return AgentGraph(mcp=fake_mcp, reason_agent=None)
