"""App 3 — MCP client (stdio). The agent's ONLY data path.

Per the architecture (docs/internal/app3-agent.md), the agent never imports
apps.mcp.* or apps.embedding.* — it spawns the MCP server (App 4) as a
subprocess over stdio and calls tools through the MCP protocol. The server owns
the single runtime DuckDB connection.

A dedicated background event loop serves the sync API (call_tool / call_tools),
so callers never fight over asyncio.run() in a running loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = REPO_ROOT / "apps" / "mcp" / "server.py"


class McpClientError(RuntimeError):
    """Raised when the MCP server can't be started or a tool call fails."""


class McpClient:
    """Spawn apps/mcp/server.py over stdio and call tools (sync API)."""

    def __init__(self, server_script: Path | None = None) -> None:
        self.server_script = server_script or SERVER_SCRIPT
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stdio: Any = None
        self._started = threading.Event()
        self._stop = threading.Event()

    # -- lifecycle -----------------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._lifecycle())

    async def _lifecycle(self) -> None:
        """Single task owning the stdio/session context managers for the whole life."""
        try:
            await self._start_async()
            self._started.set()
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP client startup failed: %s", exc)
            self._started.set()
            return
        # keep the contexts alive until told to stop
        while not self._stop.is_set():
            await asyncio.sleep(0.2)
        await self._close_async()

    def start(self) -> None:
        """Start the server subprocess + event loop; blocks until ready."""
        if self._loop is not None:
            return
        self._stop.clear()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=30):
            raise McpClientError("MCP client failed to start (timeout).")

    async def _start_async(self) -> None:
        if self._session is not None:
            return
        if not self.server_script.exists():
            raise McpClientError(f"MCP server script not found: {self.server_script}")
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            cwd=str(REPO_ROOT),
            env=self._server_env(),
        )
        self._stdio = stdio_client(params)
        self._read, self._write = await self._stdio.__aenter__()
        self._session = await ClientSession(self._read, self._write).__aenter__()
        await self._session.initialize()

    def _server_env(self) -> dict[str, str]:
        """Environment for the server subprocess: inherit + silence the venv
        sys.prefix warning that pollutes stderr (and can tangle with capture)."""
        import os

        env = dict(os.environ)
        env.setdefault("PYTHONWARNINGS", "ignore")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def close(self) -> None:
        if self._loop is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._loop = None
        self._thread = None

    async def _close_async(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio is not None:
            await self._stdio.__aexit__(None, None, None)
            self._stdio = None

    def __enter__(self) -> "McpClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- tool calls -----------------------------------------------------------
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call an MCP tool (sync); returns its {data, source_refs, warnings} payload."""
        if self._loop is None:
            self.start()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(name, arguments or {}), self._loop
        )
        return future.result(timeout=60)

    def call_tools(self, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        """Call several tools sequentially on the same session; {tool: result}."""
        if self._loop is None:
            self.start()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(self._call_tools_async(calls), self._loop)
        return future.result(timeout=120)

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            await self._start_async()
        assert self._session is not None
        result = await self._session.call_tool(name, arguments)
        text = ""
        for block in getattr(result, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "") or ""
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "") or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("tool %s returned non-JSON payload: %.200s", name, text)
            return {"data": text, "source_refs": [], "warnings": ["Tool returned non-JSON output."]}

    async def _call_tools_async(self, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        await self._start_async()
        out: dict[str, dict[str, Any]] = {}
        for name, args in calls:
            try:
                out[name] = await self._call_tool_async(name, args)
            except Exception as exc:  # noqa: BLE001
                logger.error("tool %s failed: %s", name, exc)
                out[name] = {"data": None, "source_refs": [], "warnings": [f"Tool {name} failed: {exc}"]}
        return out
