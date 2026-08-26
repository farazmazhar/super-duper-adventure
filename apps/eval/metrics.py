"""App 6 — Layer 1 deterministic metrics (from the agent's in-band trace + answer).

No LLM, no network. Each metric returns a dict {passed, score, detail}. The
runner aggregates them per question and across the golden set.

Trace shape (from apps/agent/tracing): the state's `trace` is
{spans: [{name, kind, status, metadata}], events: [...]}. Spans of interest:
- name="classify", metadata={intent, entities}
- name="route", metadata={intent, routed_node}
- name="node:<node>", metadata={tools: [...]}
- name="reason", metadata={used_llm, input_tokens, ...}
The answer is an AnswerSchema dict: {summary, facts, interpretation,
recommendation, other_sections, confidence, render_hint}.
"""

from __future__ import annotations

import re
from typing import Any

RECORD_ID_RE = re.compile(r"\b(?:CUST|TCK|FDB)-\d+\b")


def _span(trace: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not trace:
        return None
    for s in trace.get("spans") or []:
        if s.get("name") == name:
            return s
    return None


def metric_routing(trace: dict[str, Any], expected_route: str) -> dict[str, Any]:
    span = _span(trace, "route")
    actual = (span or {}).get("metadata", {}).get("routed_node")
    passed = actual == expected_route
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "detail": {"expected": expected_route, "actual": actual}}


def metric_intent(trace: dict[str, Any], expected_intent: str) -> dict[str, Any]:
    span = _span(trace, "classify")
    actual = (span or {}).get("metadata", {}).get("intent")
    passed = actual == expected_intent
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "detail": {"expected": expected_intent, "actual": actual}}


def metric_entities(trace: dict[str, Any], expected: dict[str, object]) -> dict[str, Any]:
    span = _span(trace, "classify")
    entities = (span or {}).get("metadata", {}).get("entities") or {}
    found: list[str] = []
    missing: list[str] = []
    for key, exp in expected.items():
        if key == "customer_ids":
            actual = set(entities.get("customer_ids") or [])
        elif key == "ticket_ids":
            actual = set(entities.get("ticket_ids") or [])
        elif key == "feedback_ids":
            actual = set(entities.get("feedback_ids") or [])
        else:
            actual = {entities.get(key)} if entities.get(key) else set()
        exp_set = {exp} if isinstance(exp, str) else set(exp or [])
        if exp_set and exp_set <= actual:
            found.append(key)
        else:
            missing.append(key)
    passed = not missing
    return {"passed": passed, "score": len(found) / max(len(expected), 1),
            "detail": {"found": found, "missing": missing, "entities": entities}}


def metric_tools(trace: dict[str, Any], expected_tools: list[str]) -> dict[str, Any]:
    """Right tools called (from the routed node's span metadata.tools)."""
    called: set[str] = set()
    for s in trace.get("spans") or []:
        name = s.get("name", "")
        if name.startswith("node:"):
            called.update(s.get("metadata", {}).get("tools") or [])
    missing = [t for t in expected_tools if t not in called]
    passed = not missing
    return {"passed": passed, "score": (len(expected_tools) - len(missing)) / max(len(expected_tools), 1),
            "detail": {"expected": expected_tools, "called": sorted(called), "missing": missing}}


def metric_retrieval(trace: dict[str, Any], answer: dict[str, Any], expected: list[str]) -> dict[str, Any]:
    """Expected record types surfaced (from retrieved docs in context / answer cites)."""
    if not expected:
        return {"passed": True, "score": 1.0, "detail": {"note": "no retrieval expected"}}
    texts = [answer.get("summary") or ""] + [s.get("content", "") for s in answer.get("facts", [])]
    blob = " ".join(texts).lower()
    found = [r for r in expected if r.lower() in blob]
    passed = len(found) == len(expected)
    return {"passed": passed, "score": len(found) / len(expected),
            "detail": {"expected": expected, "found": found}}


def metric_citations(answer: dict[str, Any]) -> dict[str, Any]:
    """Every Facts section cites a record id."""
    facts = answer.get("facts") or []
    if not facts:
        return {"passed": False, "score": 0.0, "detail": {"note": "no facts section"}}
    uncited = [f.get("heading", "?") for f in facts if not _section_has_citation(f)]
    passed = not uncited
    return {"passed": passed, "score": 1.0 if passed else 0.0, "detail": {"uncited": uncited}}


def _section_has_citation(section: dict[str, Any]) -> bool:
    if section.get("citations"):
        return True
    # fall back: does the content mention a record id?
    return bool(RECORD_ID_RE.search(section.get("content", "")))


def metric_render(answer: dict[str, Any], expected: str) -> dict[str, Any]:
    hint = answer.get("render_hint") or {}
    if isinstance(hint, dict):
        payload = hint.get("payload") or {}
        # Prefer the structured payload kind (the answer node attaches it);
        # fall back to the LLM's own kind.
        kind = (payload.get("kind") if isinstance(payload, dict) else None) or hint.get("kind") or "markdown"
    else:
        kind = getattr(hint, "kind", "markdown")
    passed = kind == expected
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "detail": {"expected": expected, "actual": kind}}


def metric_recommendations(answer: dict[str, Any], expected: bool) -> dict[str, Any]:
    has = bool(answer.get("recommendation"))
    # One-directional: when recommendations are expected they must be present;
    # when not expected, extra recommendations are fine (they're on-demand).
    passed = has if expected else True
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "detail": {"expected": expected, "actual": has}}


def metric_prioritization(answer: dict[str, Any], expected: bool) -> dict[str, Any]:
    others = answer.get("other_sections") or []
    headings = [s.get("heading", "").lower() for s in others]
    has = any("priorit" in h for h in headings) or any("priorit" in (s.get("content", "") or "").lower() for s in others)
    # One-directional, same as recommendations.
    passed = has if expected else True
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "detail": {"expected": expected, "actual": has, "headings": headings}}


def metric_irrelevant(answer: dict[str, Any], trace: dict[str, Any], expected: bool) -> dict[str, Any]:
    if not expected:
        return {"passed": True, "score": 1.0, "detail": {"note": "not an irrelevant question"}}
    conf = answer.get("confidence")
    hint = answer.get("render_hint") or {}
    kind = hint.get("kind") if isinstance(hint, dict) else getattr(hint, "kind", "markdown")
    passed = conf == "low" and kind == "markdown"
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "detail": {"confidence": conf, "render_kind": kind}}


def metric_confidence(answer: dict[str, Any], expected_low: bool) -> dict[str, Any]:
    conf = answer.get("confidence")
    if expected_low:
        passed = conf == "low"
    else:
        passed = conf in ("high", "medium")
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "detail": {"expected_low": expected_low, "actual": conf}}


ALL_METRICS = [
    "routing", "intent", "entities", "tools", "retrieval", "citations",
    "render", "recommendations", "prioritization", "irrelevant", "confidence",
]


def evaluate_question(
    golden: Any,
    trace: dict[str, Any],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Run all Layer-1 metrics for one golden question.

    Returns {question_id, passed_metrics, total_metrics, passed, metrics: {...}}.
    """
    metrics: dict[str, dict[str, Any]] = {
        "routing": metric_routing(trace, golden.expected_route),
        "intent": metric_intent(trace, golden.expected_intent),
        "entities": metric_entities(trace, golden.expected_entities),
        "tools": metric_tools(trace, golden.expected_tools),
        "retrieval": metric_retrieval(trace, answer, golden.expected_retrieval),
        # blocked/irrelevant replies legitimately carry no citations
        "citations": ({"passed": True, "score": 1.0, "detail": {"note": "skipped for irrelevant"}}
                      if golden.irrelevant else metric_citations(answer)),
        "render": metric_render(answer, golden.expected_render),
        "recommendations": metric_recommendations(answer, golden.expects_recommendations),
        "prioritization": metric_prioritization(answer, golden.expects_prioritization),
        "irrelevant": metric_irrelevant(answer, trace, golden.irrelevant),
        "confidence": metric_confidence(answer, golden.no_data),
    }
    passed = [k for k, m in metrics.items() if m["passed"]]
    return {
        "question_id": golden.id,
        "passed_metrics": len(passed),
        "total_metrics": len(metrics),
        "passed": len(passed) == len(metrics),
        "metrics": metrics,
    }
