"""Memory tests: session trimming + long-term over a fake MCP client."""

from __future__ import annotations

from apps.agent.memory import LongTermMemory, SessionMemory


def test_session_memory_roundtrip() -> None:
    mem = SessionMemory(max_messages=4)
    mem.add_user("hello")
    mem.add_assistant("hi there")
    hist = mem.history()
    assert hist[0] == {"role": "user", "content": "hello"}
    assert len(hist) == 2


def test_session_memory_trims() -> None:
    mem = SessionMemory(max_messages=2)
    for i in range(5):
        mem.add_user(f"q{i}")
    assert len(mem.history()) == 2
    assert mem.history()[0]["content"] == "q3"


def test_session_memory_clear() -> None:
    mem = SessionMemory()
    mem.add_user("x")
    mem.clear()
    assert mem.history() == []


class _FakeMemMcp:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        args = arguments or {}
        if name == "read_memory":
            val = self.store.get(args.get("key"))
            return {"data": {"key": args.get("key"), "value": val} if val else None, "source_refs": [], "warnings": []}
        if name == "write_memory":
            self.store[args["key"]] = args["value"]
            return {"data": {"key": args["key"], "value": args["value"]}, "source_refs": [], "warnings": []}
        if name == "list_memory":
            return {"data": [{"key": k, "value": v} for k, v in self.store.items()], "source_refs": [], "warnings": []}
        return {"data": None, "source_refs": [], "warnings": []}


def test_long_term_memory_roundtrip() -> None:
    mcp = _FakeMemMcp()
    mem = LongTermMemory(mcp)  # type: ignore[arg-type]
    assert mem.read("k") is None
    mem.write("k", "v")
    assert mem.read("k") == "v"
    assert mem.list() == [{"key": "k", "value": "v"}]


def test_long_term_summarize_session() -> None:
    mcp = _FakeMemMcp()
    mem = LongTermMemory(mcp)  # type: ignore[arg-type]
    mem.summarize_session([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}])
    val = mem.read("last_session")
    assert val is not None and "questions=1" in val
