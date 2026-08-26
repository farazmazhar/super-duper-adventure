"""App 3 — custom lightweight tracing (OTel-shaped, persisted via MCP).

TraceContext + span helpers. The answer node returns the full trace in-band to
the FE; the graph also persists spans via the MCP `write_memory`-style sink
(agent.traces / agent.trace_events in the shared DB, owned by App 4).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """One trace span (mirrors OTel span shape so Logfire/LangSmith can drop in)."""

    name: str
    kind: str = "internal"  # node | tool | llm | retrieval | guardrail | memory | error
    status: str = "ok"  # ok | error | skipped
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None  # JSON-able result of the action (tool response, retrieval, rerank)
    parent_span_id: str | None = None

    def end(self, status: str = "ok", result: dict[str, Any] | None = None, **metadata: Any) -> "Span":
        self.ended_at = time.time()
        self.status = status
        if result is not None:
            self.result = result
        self.metadata.update(metadata)
        return self

    @property
    def latency_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000.0, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "result": self.result,
            "parent_span_id": self.parent_span_id,
        }


@dataclass
class TraceContext:
    """Per-request trace: spans + events. Returned in-band with the answer."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    spans: list[Span] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    # -- span management ------------------------------------------------------
    def start_span(self, name: str, kind: str = "internal", **metadata: Any) -> Span:
        span = Span(name=name, kind=kind, metadata=metadata)
        self.spans.append(span)
        return span

    def add_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append({"event": event, "ts": time.time(), "payload": payload or {}})

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "spans": [s.to_dict() for s in self.spans],
            "events": self.events,
        }
