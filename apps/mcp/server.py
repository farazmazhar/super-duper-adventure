"""App 4 — MCP server (stdio) exposing the shared tool surface.

Run standalone:  python -m apps.mcp.server
        or via:   mcp run apps/mcp/server.py

Every tool is a thin wrapper around the shared implementations in
apps/mcp/tools.py (single source of truth, also imported by the agent).
Uses the mcp 2.x API: MCPServer (FastMCP was renamed in v2).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from apps.mcp import retrieval, semantic, tools  # noqa: E402

server = MCPServer("customer-intel")


@server.tool()
def get_customer_profile(customer_id: str) -> dict:
    """Attributes + derived features + risk inputs for one customer."""
    return tools.get_customer_profile(customer_id)


@server.tool()
def rank_customer_risk(limit: int = 10, segment: str | None = None, status: str | None = None) -> dict:
    """Rank customers by composite risk score; include top drivers per customer."""
    return tools.rank_customer_risk(limit=limit, segment=segment, status=status)


@server.tool()
def calculate_revenue_at_risk(segment: str | None = None) -> dict:
    """Sum monthly_revenue for high-risk + paused + canceled customers; NULL revenue flagged."""
    return tools.calculate_revenue_at_risk(segment=segment)


@server.tool()
def get_customer_tickets(
    customer_id: str, limit: int = 20, category: str | None = None,
    priority: str | None = None, status: str | None = None,
) -> dict:
    """Tickets for a customer with category/priority/status/satisfaction."""
    return tools.get_customer_tickets(
        customer_id=customer_id, limit=limit, category=category,
        priority=priority, status=status,
    )


@server.tool()
def get_customer_feedback(customer_id: str, limit: int = 20) -> dict:
    """Feedback for a customer with theme + sentiment."""
    return tools.get_customer_feedback(customer_id=customer_id, limit=limit)


@server.tool()
def get_feedback_themes(min_count: int = 1, segment: str | None = None) -> dict:
    """Theme counts from aggregate_theme (optional segment filter)."""
    return tools.get_feedback_themes(min_count=min_count, segment=segment)


@server.tool()
def get_ticket_breakdown(segment: str | None = None) -> dict:
    """Tickets grouped by category/priority/status with counts + avg resolution + avg satisfaction."""
    return tools.get_ticket_breakdown(segment=segment)


@server.tool()
def get_usage_change(customer_id: str | None = None, segment: str | None = None) -> dict:
    """Sessions/active_users change % + errors; per-customer or aggregated."""
    return tools.get_usage_change(customer_id=customer_id, segment=segment)


@server.tool()
def get_usage_trend(customer_id: str, weeks: int = 8) -> dict:
    """Daily/weekly sessions + errors series from fact_usage (for charts)."""
    return tools.get_usage_trend(customer_id=customer_id, weeks=weeks)


@server.tool()
def get_subscription_events(customer_id: str, limit: int = 10) -> dict:
    """Renewals/upgrades/downgrades/cancellations + revenue_change for a customer."""
    return tools.get_subscription_events(customer_id=customer_id, limit=limit)


@server.tool()
def calculate_segment_metrics(segment: str | None = None) -> dict:
    """Stats for a segment (or all) vs global benchmarks."""
    return tools.calculate_segment_metrics(segment=segment)


@server.tool()
def list_customers(
    segment: str | None = None, status: str | None = None,
    limit: int = 50, search: str | None = None,
) -> dict:
    """Discover customers by segment/status/search."""
    return tools.list_customers(segment=segment, status=status, limit=limit, search=search)


@server.tool()
def resolve_customer_name(name: str) -> dict:
    """Resolve a customer name (or fragment) to customer_id(s) — for NL queries
    like 'tell me about VertexPath A'."""
    return tools.resolve_customer_name(name)


@server.tool()
def semantic_query(query: dict) -> dict:
    """Semantic layer: query the data via a validated SemanticQuery (no raw SQL).

    query = {metric, of, of_dimension?, dimensions?, filters?, time_range?, limit?}
    Metric/entity/dimensions/filters must come from the curated catalog; unknown
    values are rejected. Executes read-only with a row cap + timeout.
    """
    return semantic.execute_semantic_query(query)


@server.tool()
def get_catalog() -> dict:
    """Return the semantic catalog: every queryable entity, its columns, and
    human-readable descriptions — so the agent can map natural-language
    questions to entities/columns at runtime and build SemanticQueries."""
    return semantic.get_catalog()


@server.tool()
def retrieve_sources(query: str, k: int = 20, filters: dict | None = None, rerank_enabled: bool | None = None) -> dict:
    """RAG retrieval over feedback/ticket embeddings (optional rerank).

    rerank_enabled overrides the env default (RERANK_ENABLED) per call — the
    FE chat toggle uses this.
    """
    return retrieval.retrieve_sources(query=query, k=k, filters=filters, rerank_enabled=rerank_enabled)


@server.tool()
def read_memory(key: str) -> dict:
    """Read a long-term memory value."""
    return tools.read_memory(key)


@server.tool()
def write_memory(key: str, value: str) -> dict:
    """Write a long-term memory value (upsert)."""
    return tools.write_memory(key, value)


@server.tool()
def list_memory() -> dict:
    """List all long-term memory entries."""
    return tools.list_memory()


if __name__ == "__main__":
    server.run()
