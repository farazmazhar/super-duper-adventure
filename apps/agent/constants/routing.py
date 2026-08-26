"""Routing + answer-format constants.

- ROUTE_MAP: intent -> specialized node (spec §2 route).
- Node tool lists: which MCP tools each specialized node calls (spec §3).
- Answer templates: safe messages for irrelevant/blocked/error paths (spec §5, §6).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Route map: intent -> node name (spec §2 route). `general` is the fallback.
# ---------------------------------------------------------------------------
ROUTE_MAP = {
    "customer_query": "customer",
    "ticket_query": "themes",  # scoped by ticket_id via get_ticket_by_id in the node
    "feedback_query": "themes",  # scoped by feedback_id
    "theme_sentiment": "themes",
    "analytics_exec": "analytics",
    "trend": "trend",
    "free_text_rag": "rag",
    "general": "general",
    "irrelevant": "blocked",  # special: no node runs, short reply, no visuals
}

# ---------------------------------------------------------------------------
# Node -> MCP tools to call (spec §3 specialized nodes).
# ---------------------------------------------------------------------------
NODE_TOOLS = {
    "customer": [
        "get_customer_profile",
        "get_customer_tickets",
        "get_customer_feedback",
        "get_usage_change",
        "get_subscription_events",
        "get_usage_trend",
    ],
    "themes": ["retrieve_sources", "get_feedback_themes", "get_ticket_breakdown"],
    "analytics": ["calculate_segment_metrics", "rank_customer_risk", "calculate_revenue_at_risk", "list_customers", "semantic_query"],
    "trend": ["get_usage_trend", "get_usage_change", "get_ticket_breakdown"],
    "rag": ["retrieve_sources", "get_feedback_themes", "get_ticket_breakdown", "semantic_query"],
    "general": [
        "get_customer_profile",
        "get_customer_feedback",
        "get_feedback_themes",
        "get_ticket_breakdown",
        "get_usage_change",
        "rank_customer_risk",
        "calculate_revenue_at_risk",
        "retrieve_sources",
        "semantic_query",
    ],
}

# ---------------------------------------------------------------------------
# Safe user-facing messages (never raw tracebacks).
# ---------------------------------------------------------------------------
IRRELEVANT_REPLY = (
    "I can help with customer-intelligence questions about this company's data — "
    "e.g. which customers need attention, feedback themes, usage trends, risk, "
    "and revenue at risk. That question is outside that scope."
)

BLOCKED_REPLY = (
    "I can't help with that. I'm a customer-intelligence assistant bounded to "
    "this company's synthetic data. Ask me about customers, tickets, feedback, "
    "usage trends, risk, or revenue."
)

ERROR_REPLY_TEMPLATE = (
    "I couldn't complete that. {reason} Please try rephrasing, or ask a "
    "different customer-intelligence question."
)

# Confidence default when a node produced no evidence.
LOW_CONFIDENCE_NOTE = "Insufficient data — confidence is low."
