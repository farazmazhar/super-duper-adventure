"""Agent semantic-tool binding tests (no real MCP subprocess / LLM).

Verifies that `build_reason_agent(mcp=...)` binds the `semantic_query` +
`get_catalog` tools and that they call through the provided MCP client.
"""

from __future__ import annotations

from apps.agent.agent import build_reason_agent
from apps.agent.mcp_client import McpClient
from apps.agent.tests.conftest import FakeMcpClient


def test_reason_agent_binds_semantic_tools() -> None:
    fake = FakeMcpClient(
        {
            "semantic_query": {"data": [{"country": "Japan", "value": 13}], "columns": ["country", "value"], "warnings": [], "source_refs": []},
            "get_catalog": {"data": {"entities": ["customer"], "metrics": ["count"]}, "columns": [], "warnings": [], "source_refs": []},
        }
    )
    agent = build_reason_agent(mcp=fake)  # type: ignore[arg-type]
    ts = getattr(agent, "_function_toolset")
    names = list(ts.tools.keys())
    assert "semantic_query" in names
    assert "get_catalog" in names


def test_reason_agent_no_mcp_has_no_tools() -> None:
    agent = build_reason_agent()
    ts = getattr(agent, "_function_toolset")
    assert list(ts.tools.keys()) == []


def test_semantic_tool_calls_through_mcp() -> None:
    fake = FakeMcpClient(
        {
            "semantic_query": {"data": [{"category": "bug", "value": 10}], "columns": ["category", "value"], "warnings": [], "source_refs": []},
        }
    )
    agent = build_reason_agent(mcp=fake)  # type: ignore[arg-type]
    ts = getattr(agent, "_function_toolset")
    tool = ts.tools["semantic_query"]
    result = tool.function(None, {"metric": "count", "of": "ticket", "dimensions": ["category"]})
    assert result["data"] == [{"category": "bug", "value": 10}]
    assert fake.calls[-1] == (
        "semantic_query",
        {"query": {"metric": "count", "of": "ticket", "dimensions": ["category"]}},
    )


def test_catalog_tool_calls_through_mcp() -> None:
    fake = FakeMcpClient(
        {"get_catalog": {"data": {"entities": ["ticket"]}, "columns": [], "warnings": [], "source_refs": []}}
    )
    agent = build_reason_agent(mcp=fake)  # type: ignore[arg-type]
    ts = getattr(agent, "_function_toolset")
    tool = ts.tools["get_catalog"]
    result = tool.function(None)
    assert result["data"] == {"entities": ["ticket"]}
    assert fake.calls[-1] == ("get_catalog", {})
