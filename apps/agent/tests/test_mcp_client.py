"""MCP client tests — against a mock session (no subprocess under pytest).

The client's core logic (payload parsing, multi-tool aggregation, error
handling) is tested by injecting a fake session. The real subprocess spawn is
covered by one `integration`-marked test (opt-in via `-m integration`).
"""

from __future__ import annotations

import json

import pytest

from apps.agent import mcp_client as mc
from apps.agent.mcp_client import McpClient, McpClientError


class FakeSession:
    """Minimal stand-in for the mcp ClientSession (records calls, returns text JSON)."""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def call_tool(self, name: str, arguments: dict) -> "_FakeResult":
        self.calls.append((name, arguments))
        payload = self.responses.get(name, {"data": None, "source_refs": [], "warnings": []})
        return _FakeResult(json.dumps(payload))


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.content = [{"type": "text", "text": text}]


class FakeMcpClient(McpClient):
    """McpClient with the subprocess/lifecycle replaced by a fake session.

    `call_tool` / `call_tools` run the fake session's coroutines directly (no
    background loop, no subprocess).
    """

    def __init__(self, session: FakeSession) -> None:
        super().__init__()
        self._session = session  # type: ignore[assignment]

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        import asyncio
        return asyncio.run(self._call_tool_async(name, arguments or {}))

    def call_tools(self, calls: list[tuple[str, dict]]) -> dict[str, dict]:
        import asyncio
        return asyncio.run(self._call_tools_async(calls))


def test_call_tool_parses_json() -> None:
    session = FakeSession({"list_customers": {"data": [{"customer_id": "CUST-0001"}], "source_refs": [], "warnings": []}})
    client = FakeMcpClient(session)
    res = client.call_tool("list_customers", {"limit": 1})
    assert res["data"] == [{"customer_id": "CUST-0001"}]
    assert session.calls == [("list_customers", {"limit": 1})]


def test_call_tool_non_json_fallback() -> None:
    class _RawResult:
        content = [{"type": "text", "text": "not-json"}]

    class _S(FakeSession):
        async def call_tool(self, name, arguments):
            return _RawResult()

    client = FakeMcpClient(_S())
    res = client.call_tool("x")
    assert res["data"] == "not-json"
    assert "non-JSON" in res["warnings"][0]


def test_call_tools_aggregates() -> None:
    session = FakeSession(
        {
            "a": {"data": [1], "source_refs": [], "warnings": []},
            "b": {"data": [2], "source_refs": [], "warnings": []},
        }
    )
    client = FakeMcpClient(session)
    out = client.call_tools([("a", {}), ("b", {})])
    assert list(out.keys()) == ["a", "b"]
    assert out["a"]["data"] == [1]
    assert out["b"]["data"] == [2]
    assert [c[0] for c in session.calls] == ["a", "b"]


def test_call_tools_survives_tool_error() -> None:
    class _Broken(FakeSession):
        async def call_tool(self, name, arguments):
            raise RuntimeError("boom")

    client = FakeMcpClient(_Broken())
    out = client.call_tools([("a", {})])
    assert "a" in out
    assert "boom" in out["a"]["warnings"][0]


def test_missing_server_script_raises() -> None:
    client = McpClient(server_script=__import__("pathlib").Path("/nonexistent/server.py"))
    with pytest.raises(McpClientError, match="not found"):
        # _start_async validates the script path before spawning
        import asyncio
        asyncio.run(client._start_async())


@pytest.mark.integration
def test_real_subprocess_smoke() -> None:
    """Opt-in: spawns the actual MCP server. Run with `-m integration`."""
    with McpClient() as client:
        res = client.call_tool("list_customers", {"limit": 2})
        assert len(res["data"]) >= 1
