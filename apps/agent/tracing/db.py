"""Trace persistence — writes go through MCP (agent.traces / agent.trace_events).

The DB is owned by App 4 at runtime; this module only formats the spans for the
sink. The graph persists the trace after the answer node completes.
"""

from __future__ import annotations

from typing import Any

from apps.agent.tracing import TraceContext


def trace_payload(trace: TraceContext) -> dict[str, Any]:
    """Shape the trace for persistence (agent.traces + agent.trace_events)."""
    return {
        "trace_id": trace.trace_id,
        "question": trace.question,
        "spans": [s.to_dict() for s in trace.spans],
        "events": trace.events,
    }
