"""Output guardrail tests: schema, confidence, citations, PII, tools, render."""

from __future__ import annotations

import pytest

from apps.guardrails.models import AnswerSchema, AnswerSection, RenderHint
from apps.guardrails.output_guardrails import (
    KNOWN_TOOLS,
    check_allowed_tool,
    check_answer_schema,
    check_citations,
    check_confidence,
    check_pii,
    check_render_safety,
    check_tool_arguments,
    evaluate_output,
    safe_fallback_answer,
    validate_tool_call,
)


def make_answer(**overrides) -> AnswerSchema:
    defaults = dict(
        facts=[AnswerSection(heading="Facts", content="CUST-0001 is at risk.", citations=["CUST-0001"])],
        confidence="high",
        render_hint=RenderHint(kind="table"),
        summary="CUST-0001 needs attention.",
    )
    defaults.update(overrides)
    return AnswerSchema(**defaults)


# --- schema -----------------------------------------------------------------
def test_answer_schema_valid() -> None:
    r = check_answer_schema(make_answer())
    assert r.severity == "pass"


def test_answer_schema_invalid_flags() -> None:
    r = check_answer_schema({"facts": "not a list", "confidence": "very"})
    assert r.severity == "flag"
    assert not r.passed


def test_answer_schema_accepts_dict() -> None:
    r = check_answer_schema(
        {
            "facts": [{"heading": "F", "content": "c", "citations": ["TCK-1"]}],
            "confidence": "medium",
        }
    )
    assert r.severity == "pass"


# --- confidence -------------------------------------------------------------
def test_confidence_forced_low_without_evidence() -> None:
    answer = make_answer(confidence="high")
    r = check_confidence(answer, has_sources=False, has_tool_results=False)
    assert r.severity == "flag"
    assert r.detail["forced"] == "low"


def test_confidence_ok_with_evidence() -> None:
    r = check_confidence(make_answer(), has_sources=True, has_tool_results=False)
    assert r.severity == "pass"


# --- citations --------------------------------------------------------------
def test_citations_ok() -> None:
    assert check_citations(make_answer()).severity == "pass"


def test_citations_missing_flags() -> None:
    answer = make_answer(facts=[AnswerSection(heading="Facts", content="No refs here", citations=[])])
    r = check_citations(answer)
    assert r.severity == "flag"
    assert r.detail["sections"] == ["Facts"]


# --- PII --------------------------------------------------------------------
def test_pii_clean() -> None:
    assert check_pii(make_answer()).severity == "pass"


def test_pii_email_flags() -> None:
    answer = make_answer(summary="Contact john.doe@example.com for details")
    r = check_pii(answer)
    assert r.severity == "flag"
    assert "email" in r.detail


# --- render safety ----------------------------------------------------------
def test_render_safety_known_kind() -> None:
    assert check_render_safety(make_answer(render_hint=RenderHint(kind="chart"))).severity == "pass"


def test_render_safety_falls_back_to_markdown_on_weird_input() -> None:
    # A valid AnswerSchema can't carry an unknown kind (Literal rejects it at
    # parse time), so the check is defensive: non-RenderHint input falls back to
    # markdown (the safe default) and passes — never crashes.
    class WeirdHint:
        kind = "fancy3d"

    class WeirdAnswer:
        render_hint = WeirdHint()

    r = check_render_safety(WeirdAnswer())  # type: ignore[arg-type]
    assert r.severity == "pass"


def test_unknown_render_kind_is_caught_by_schema_validation() -> None:
    # The Literal field type means an unknown kind fails AnswerSchema validation
    # (flagged), which is the realistic path for a bad render_hint.
    import pytest
    from pydantic import ValidationError

    raw = make_answer().model_dump()
    raw["render_hint"] = {"kind": "fancy3d"}
    with pytest.raises(ValidationError):
        AnswerSchema.model_validate(raw)


# --- tools ------------------------------------------------------------------
def test_allowed_tool_registry_has_mcp_surface() -> None:
    for name in (
        "get_customer_profile",
        "rank_customer_risk",
        "calculate_revenue_at_risk",
        "get_customer_tickets",
        "get_customer_feedback",
        "get_feedback_themes",
        "get_ticket_breakdown",
        "get_usage_change",
        "get_usage_trend",
        "get_subscription_events",
        "calculate_segment_metrics",
        "list_customers",
        "semantic_query",
        "retrieve_sources",
        "read_memory",
        "write_memory",
        "list_memory",
    ):
        assert name in KNOWN_TOOLS


def test_unknown_tool_blocks() -> None:
    r = check_allowed_tool("delete_all_customers")
    assert r.severity == "block"


def test_tool_arguments_valid() -> None:
    r = check_tool_arguments("get_customer_profile", {"customer_id": "CUST-0001"})
    assert r.severity == "pass"


def test_tool_arguments_unknown_key_blocks() -> None:
    r = check_tool_arguments("get_customer_profile", {"customer_id": "CUST-0001", "admin": True})
    assert r.severity == "block"
    assert r.rule == "tool_arguments"


def test_validate_tool_call_returns_both() -> None:
    # known tool, bad args -> allowed_tools passes, tool_arguments blocks
    results = validate_tool_call("get_customer_profile", {"admin": True})
    assert [r.rule for r in results] == ["allowed_tools", "tool_arguments"]
    assert results[0].severity == "pass"
    assert results[1].severity == "block"


def test_validate_tool_call_unknown_tool() -> None:
    results = validate_tool_call("nope", {})
    # both checks report the unknown tool as a block (arg check delegates)
    assert all(r.severity == "block" for r in results)


# --- evaluate_output --------------------------------------------------------
def test_evaluate_output_aggregates_all() -> None:
    results = evaluate_output(make_answer(), has_sources=True, has_tool_results=True)
    rules = [r.rule for r in results]
    assert rules == ["answer_schema", "confidence", "citations", "pii", "render_safety"]
    assert all(r.severity == "pass" for r in results)


def test_evaluate_output_flags_on_bad_answer() -> None:
    results = evaluate_output({"facts": "nope", "confidence": "x"}, has_sources=False, has_tool_results=False)
    assert results[0].severity == "flag"  # schema
    assert len(results) == 1  # stops after schema failure


def test_safe_fallback_answer_is_valid() -> None:
    a = safe_fallback_answer()
    assert check_answer_schema(a).severity == "pass"
    assert a.confidence == "low"
