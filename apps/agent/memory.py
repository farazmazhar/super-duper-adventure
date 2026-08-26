"""App 3 — memory: short-term (session) + long-term (via MCP tools).

Long-term memory lives in DuckDB `agent.agent_memory` (owned by App 4); the
agent reads/writes it through the MCP client — never a direct DB connection.
"""

from __future__ import annotations

from typing import Any

from apps.agent.mcp_client import McpClient


class SessionMemory:
    """Short-term conversation memory (LangGraph state / session list)."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self.messages: list[dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def history(self) -> list[dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []

    def _trim(self) -> None:
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]


class LongTermMemory:
    """Long-term memory over the MCP read_memory/write_memory/list_memory tools."""

    def __init__(self, client: McpClient) -> None:
        self._client = client

    def read(self, key: str) -> str | None:
        res = self._client.call_tool("read_memory", {"key": key})
        data = res.get("data")
        if isinstance(data, dict):
            return data.get("value")
        return None

    def write(self, key: str, value: str) -> None:
        self._client.call_tool("write_memory", {"key": key, "value": value})

    def list(self) -> list[dict[str, Any]]:
        res = self._client.call_tool("list_memory")
        return res.get("data") or []

    def summarize_session(self, history: list[dict[str, str]]) -> None:
        """Persist a compact session summary as long-term memory (spec §Memory)."""
        if not history:
            return
        # Rule-based summary: count questions; keeps it deterministic.
        n_user = sum(1 for m in history if m["role"] == "user")
        last_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        self.write("last_session", f"questions={n_user}; last_question={last_q[:200]}")
