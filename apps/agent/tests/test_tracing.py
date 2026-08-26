"""Tracing tests: spans, events, serialization."""

from __future__ import annotations

import time

from apps.agent.tracing import Span, TraceContext
from apps.agent.tracing.db import trace_payload


def test_span_latency() -> None:
    s = Span(name="classify", kind="node")
    time.sleep(0.01)
    s.end()
    assert s.status == "ok"
    assert s.latency_ms is not None and s.latency_ms >= 5.0


def test_span_error_status() -> None:
    s = Span(name="tool", kind="tool")
    s.end(status="error", message="boom")
    assert s.status == "error"
    assert s.metadata["message"] == "boom"


def test_trace_collects_spans_and_events() -> None:
    t = TraceContext(question="q?")
    t.start_span("classify", kind="node").end()
    t.add_event("route", {"node": "analytics"})
    d = t.to_dict()
    assert d["trace_id"] == t.trace_id
    assert len(d["spans"]) == 1
    assert len(d["events"]) == 1


def test_trace_payload_for_db() -> None:
    t = TraceContext(question="q?")
    t.start_span("classify").end()
    p = trace_payload(t)
    assert p["trace_id"] == t.trace_id
    assert "spans" in p and "events" in p
