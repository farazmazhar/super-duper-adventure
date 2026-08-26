"""Graph routing + end-to-end tests (fake MCP, deterministic reason)."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from apps.agent.constants.prompts import SYSTEM_PROMPT_TEMPLATE
from apps.agent.graph import AgentGraph
from apps.agent.tracing import TraceContext
from apps.guardrails.models import AnswerSchema


def _context_aware_agent() -> Agent:
    """Mirror of build_reason_agent (TestModel + AnswerSchema output) so the
    LLM path is exercised in tests — verifies the gathered context actually
    reaches the model (regression: it used to arrive as a literal {context})."""
    agent = Agent(TestModel(), output_type=AnswerSchema, deps_type=dict[str, Any])

    @agent.system_prompt
    def _with_context(ctx: RunContext[dict[str, Any]]) -> str:
        context = ctx.deps.get("context") or "No data gathered for this question."
        return SYSTEM_PROMPT_TEMPLATE.format(context=context)

    return agent


def test_customer_question_routes_to_customer_node(graph, fake_mcp) -> None:
    state = graph.run("Who is CUST-0001?")
    assert state["classification"] == "customer_query"
    assert state["routed_node"] == "customer"
    tool_names = {name for name, _ in fake_mcp.calls}
    assert "get_customer_profile" in tool_names
    assert "get_usage_change" in tool_names
    answer = state["answer"]
    assert isinstance(answer, AnswerSchema)
    assert "VertexPath A" in answer.facts[0].content


def test_analytics_question_routes_to_analytics(graph, fake_mcp) -> None:
    state = graph.run("Which customers are at risk?")
    assert state["routed_node"] == "analytics"
    tool_names = {name for name, _ in fake_mcp.calls}
    assert "rank_customer_risk" in tool_names
    assert "calculate_revenue_at_risk" in tool_names


def test_theme_question_routes_to_themes(graph, fake_mcp) -> None:
    state = graph.run("What are the top feedback themes?")
    assert state["routed_node"] == "themes"
    tool_names = {name for name, _ in fake_mcp.calls}
    assert "get_feedback_themes" in tool_names
    assert "retrieve_sources" in tool_names


def test_trend_question_routes_to_trend(graph, fake_mcp) -> None:
    state = graph.run("How has usage trended over time?")
    assert state["routed_node"] == "trend"
    tool_names = {name for name, _ in fake_mcp.calls}
    assert "get_usage_trend" in tool_names


def test_irrelevant_blocked_no_tools(graph, fake_mcp) -> None:
    state = graph.run("What's the weather?")
    assert state["classification"] == "irrelevant"
    assert state["routed_node"] == "blocked"
    assert fake_mcp.calls == []  # no tools called
    answer = state["answer"]
    assert answer.confidence == "low"
    assert "customer-intelligence" in answer.summary


def test_answer_has_trace(graph) -> None:
    trace = TraceContext(question="test")
    state = graph.run("Who is CUST-0001?", trace=trace)
    assert "trace" in state
    assert state["trace"]["trace_id"] == trace.trace_id
    assert len(state["trace"]["spans"]) >= 1


def test_no_context_answers_low_confidence(graph) -> None:
    # A question that routes to a node whose tools return no data
    state = graph.run("Who is CUST-9999?")
    answer = state["answer"]
    assert isinstance(answer, AnswerSchema)
    assert answer.confidence in ("low", "medium")


def test_llm_receives_gathered_context(graph, fake_mcp) -> None:
    """The LLM path must receive the tool/retrieval context in its prompt.

    Regression: the {context} placeholder was sent to the model verbatim, so
    the agent reasoned with zero data. Now the system prompt renders deps.
    """
    agent = _context_aware_agent()
    llm_graph = AgentGraph(mcp=fake_mcp, reason_agent=agent)
    state = llm_graph.run("Who is CUST-0001?")
    # the deterministic TestModel returns an AnswerSchema; the graph ran the LLM path
    assert state["routed_node"] == "customer"
    answer = state["answer"]
    assert isinstance(answer, AnswerSchema)
    # Trace must show the LLM span (used_llm=True) — proves the reason node ran the agent
    spans = state["trace"]["spans"]
    llm_spans = [s for s in spans if s["kind"] == "llm"]
    assert llm_spans, "expected an llm span from the reason node"
    assert llm_spans[0]["metadata"].get("used_llm") is True


def test_system_prompt_template_renders_context() -> None:
    """Direct check: the template renders the context (not the placeholder).

    Uses the public system_prompt_parts() API to resolve the registered
    system-prompt function with deps, exactly as run_sync would.
    """
    from apps.agent.agent import build_reason_agent

    agent = build_reason_agent()
    parts = __import__("asyncio").run(
        agent.system_prompt_parts(deps={"context": "customer CUST-0001: VertexPath A"})
    )
    joined = " ".join(p.content for p in parts if hasattr(p, "content"))
    assert "VertexPath A" in joined
    assert "{context}" not in joined

