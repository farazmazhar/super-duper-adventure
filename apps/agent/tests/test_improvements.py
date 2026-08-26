"""Tests for the agent improvements:
- query planning (count customers by country / general analytics)
- customer-name entity hint
- conversation threading (short-term context)
- holistic RAG node tool coverage
"""

from __future__ import annotations

from typing import Any

from apps.agent.classify import build_query_plan, classify, classify_intent, extract_entities


# --- query planning (issue #3: general/analytical queries) -------------------
def test_plan_count_customers_by_country() -> None:
    plan = build_query_plan("how many customers are there by country?", extract_entities("how many customers are there by country?"))
    assert plan == {"metric": "count", "of": "customer", "dimensions": ["country"]}


def test_plan_count_tickets_by_category() -> None:
    plan = build_query_plan("count of tickets by category", extract_entities("count of tickets by category"))
    assert plan == {"metric": "count", "of": "ticket", "dimensions": ["category"]}


def test_plan_sum_revenue_by_segment() -> None:
    plan = build_query_plan("total revenue by segment", extract_entities("total revenue by segment"))
    assert plan["metric"] == "sum"
    assert plan["of"] == "customer"
    assert plan["dimensions"] == ["customer_segment"]
    assert plan["of_dimension"] == "monthly_revenue"


def test_plan_none_for_non_analytical() -> None:
    assert build_query_plan("tell me about CUST-0001", extract_entities("tell me about CUST-0001")) is None
    assert build_query_plan("hello", extract_entities("hello")) is None


def test_classify_count_by_country_routes_to_analytics() -> None:
    r = classify("How many customers are there by country?")
    assert r["intent"] == "analytics_exec"
    assert r["plan"] is not None
    assert r["plan"]["dimensions"] == ["country"]


# --- customer-name flexibility (issue #2) ------------------------------------
def test_name_hint_detected() -> None:
    e = extract_entities("tell me about VertexPath A")
    assert e["customer_name_hint"] == "VertexPath A"


def test_name_hint_classifies_as_customer_query() -> None:
    r = classify("What can you tell me about AtlasDynamics?")
    assert r["intent"] == "customer_query"


def test_name_hint_not_for_plain_keywords() -> None:
    # No capitalized name -> no hint
    e = extract_entities("how many customers are active?")
    assert e["customer_name_hint"] is None


# --- conversation threading (issue #1) ---------------------------------------
def test_graph_accepts_conversation(graph) -> None:
    """run() forwards conversation into state; the prompt gets it via deps."""
    conv = [{"role": "user", "content": "Who is CUST-0001?"}, {"role": "assistant", "content": "VertexPath A is active."}]
    state = graph.run("and what about their tickets?", conversation=conv)
    assert state["conversation"] == conv


def test_system_prompt_includes_conversation() -> None:
    """The reason-agent prompt renders prior turns (short-term memory)."""
    from apps.agent.agent import build_reason_agent
    import asyncio

    agent = build_reason_agent()
    parts = asyncio.run(
        agent.system_prompt_parts(
            deps={
                "context": "customer CUST-0001: VertexPath A",
                "conversation": [
                    {"role": "user", "content": "Who is CUST-0001?"},
                    {"role": "assistant", "content": "VertexPath A is active."},
                ],
            }
        )
    )
    joined = " ".join(p.content for p in parts if hasattr(p, "content"))
    assert "Conversation so far" in joined
    assert "Who is CUST-0001?" in joined
    assert "VertexPath A is active." in joined


# --- holistic RAG (issue #4) -------------------------------------------------
def test_rag_node_calls_multiple_tools(graph, fake_mcp) -> None:
    """The rag node gathers retrieval + themes + breakdown + semantic — not just
    retrieve_sources — so answers are holistic, not a raw search dump."""
    state = graph.run("find feedback that mentions billing problems")
    assert state["routed_node"] == "rag"
    tool_names = {name for name, _ in fake_mcp.calls}
    assert "retrieve_sources" in tool_names
    assert "get_feedback_themes" in tool_names
    assert "get_ticket_breakdown" in tool_names
    assert "semantic_query" in tool_names


# --- trend node no longer calls get_usage_trend without a customer -----------
def test_trend_without_customer_does_not_error_tool(graph, fake_mcp) -> None:
    """get_usage_trend needs a customer_id; when none is in scope the tool must
    not be called with empty args (was raising a tool validation error)."""
    state = graph.run("How has usage trended over time?")
    for name, args in fake_mcp.calls:
        if name == "get_usage_trend":
            assert args.get("customer_id"), "get_usage_trend must receive a customer_id"
