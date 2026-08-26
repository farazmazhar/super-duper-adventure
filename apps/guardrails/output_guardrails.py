"""App 5 — output guardrails (run on the answer + tool calls before display).

Everything here is deterministic (G1) and pure (no DB/network): schema
validation via Pydantic, citation/PII regex checks, tool registry checks, and
render-safety. The agent loop (App 3) calls `validate_tool_call` before each
tool execution and `evaluate_output` on the answer node output.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from apps.guardrails.models import AnswerSchema, AnswerSection, GuardrailResult, RenderHint

# ---------------------------------------------------------------------------
# PII patterns (synthetic data won't trigger these, but the patterns are real).
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"
)
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": EMAIL_RE,
    "phone": PHONE_RE,
    "credit_card": CREDIT_CARD_RE,
}

# ---------------------------------------------------------------------------
# Tool registry (G8): the allowed tool names + their Pydantic argument schemas.
# Kept in sync with apps/mcp/server.py (the registered tool surface). Tool args
# are validated against these schemas before execution.
# ---------------------------------------------------------------------------
KNOWN_TOOLS: dict[str, type[BaseModel]] = {}


def register_tool(name: str, arg_schema: type[BaseModel]) -> None:
    """Register an allowed tool with its Pydantic argument schema."""
    KNOWN_TOOLS[name] = arg_schema


def _auto_schema(*names: str) -> type[BaseModel]:
    """Build a Pydantic model from a list of allowed parameter names.

    Keeps the registry compact: every field is Optional[Any], which is enough
    to validate the *shape* of arguments (unknown keys rejected, wrong types
    caught by the tool's own signature at call time). Tools that need stricter
    checks register a hand-written schema instead.
    """
    from pydantic import create_model

    fields = {n: (Any | None, None) for n in names}
    return create_model(
        "AutoArgSchema",
        __config__={"extra": "forbid"},
        **fields,
    )


# Mirror of the MCP tool surface (apps/mcp/server.py). Names must match exactly.
_AUTO = {
    "get_customer_profile": ("customer_id",),
    "rank_customer_risk": ("limit", "segment", "status"),
    "calculate_revenue_at_risk": ("segment",),
    "get_customer_tickets": ("customer_id", "limit", "category", "priority", "status"),
    "get_customer_feedback": ("customer_id", "limit"),
    "get_feedback_themes": ("min_count", "segment"),
    "get_ticket_breakdown": ("segment",),
    "get_usage_change": ("customer_id", "segment"),
    "get_usage_trend": ("customer_id", "weeks"),
    "get_subscription_events": ("customer_id", "limit"),
    "calculate_segment_metrics": ("segment",),
    "list_customers": ("segment", "status", "limit", "search"),
    "resolve_customer_name": ("name",),
    "semantic_query": ("query",),
    "get_catalog": (),
    "retrieve_sources": ("query", "k", "filters", "rerank_enabled"),
    "read_memory": ("key",),
    "write_memory": ("key", "value"),
    "list_memory": (),
}
for _name, _args in _AUTO.items():
    register_tool(_name, _auto_schema(*_args))


# ---------------------------------------------------------------------------
# Answer-level checks
# ---------------------------------------------------------------------------
def _section_citations(section: Any) -> list[str]:
    return list(getattr(section, "citations", []) or [])


def check_answer_schema(answer: Any) -> GuardrailResult:
    """Validate the structured answer against AnswerSchema."""
    if isinstance(answer, AnswerSchema):
        return GuardrailResult(passed=True, severity="pass", rule="answer_schema", message="Answer matches AnswerSchema.")
    try:
        AnswerSchema.model_validate(answer)
        return GuardrailResult(passed=True, severity="pass", rule="answer_schema", message="Answer matches AnswerSchema.")
    except ValidationError as exc:
        return GuardrailResult(
            passed=False, severity="flag", rule="answer_schema",
            message=f"Answer failed schema validation: {exc.errors()[0]['msg'] if exc.errors() else 'invalid'}.",
            detail={"errors": exc.errors()},
        )


def check_confidence(answer: AnswerSchema, has_sources: bool, has_tool_results: bool) -> GuardrailResult:
    """Confidence must be present (schema enforces it). Force low if no evidence.

    If retrieval returned no sources AND no tool results were used, the answer
    can't be grounded — force `low` + "insufficient data".
    """
    if not has_sources and not has_tool_results:
        return GuardrailResult(
            passed=False, severity="flag", rule="confidence",
            message="Insufficient data — confidence forced to low.",
            detail={"forced": "low"},
        )
    return GuardrailResult(passed=True, severity="pass", rule="confidence", message="Confidence is grounded.")


def check_citations(answer: AnswerSchema) -> GuardrailResult:
    """Every Facts section should cite record ids. Flag sections with zero citations."""
    uncited = [s.heading for s in answer.facts if not _section_citations(s)]
    if uncited:
        return GuardrailResult(
            passed=False, severity="flag", rule="citations",
            message="Facts section(s) with no record-id citations.",
            detail={"sections": uncited},
        )
    return GuardrailResult(passed=True, severity="pass", rule="citations", message="All Facts sections cite records.")


def check_pii(answer: AnswerSchema) -> GuardrailResult:
    """Scan the answer text for PII patterns; flag if found."""
    texts = [answer.summary] + [s.content for s in answer.facts + answer.interpretation + answer.recommendation + answer.other_sections]
    found: dict[str, list[str]] = {}
    for text in texts:
        if not text:
            continue
        for kind, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found.setdefault(kind, []).extend(matches[:3])
    if found:
        return GuardrailResult(
            passed=False, severity="flag", rule="pii",
            message="PII pattern detected in answer.",
            detail=found,
        )
    return GuardrailResult(passed=True, severity="pass", rule="pii", message="No PII patterns detected.")


def check_render_safety(answer: AnswerSchema) -> GuardrailResult:
    """render_hint.kind must be one of the known kinds; unknown -> fall back to markdown."""
    hint = answer.render_hint if isinstance(answer.render_hint, RenderHint) else RenderHint()
    if hint.kind in ("table", "chart", "qa", "cards", "markdown"):
        return GuardrailResult(passed=True, severity="pass", rule="render_safety", message=f"render_hint.kind={hint.kind} allowed.")
    return GuardrailResult(
        passed=False, severity="flag", rule="render_safety",
        message="Unknown render_hint.kind — falling back to markdown.",
        detail={"kind": hint.kind},
    )


# ---------------------------------------------------------------------------
# Tool-call checks (G8)
# ---------------------------------------------------------------------------
def check_allowed_tool(name: str) -> GuardrailResult:
    """Tool name must be in the registered registry."""
    if name in KNOWN_TOOLS:
        return GuardrailResult(passed=True, severity="pass", rule="allowed_tools", message=f"Tool '{name}' is registered.")
    return GuardrailResult(
        passed=False, severity="block", rule="allowed_tools",
        message=f"Tool '{name}' is not in the registry.",
        detail={"tool": name},
    )


def check_tool_arguments(name: str, args: dict[str, Any]) -> GuardrailResult:
    """Validate tool-call arguments against the tool's Pydantic schema.

    `block` the call on bad args (the caller also flags the turn). Unknown keys
    are rejected; type/required violations are reported in detail.
    """
    schema = KNOWN_TOOLS.get(name)
    if schema is None:
        return check_allowed_tool(name)
    try:
        schema.model_validate(args or {})
        return GuardrailResult(passed=True, severity="pass", rule="tool_arguments", message=f"Args for '{name}' are valid.")
    except ValidationError as exc:
        return GuardrailResult(
            passed=False, severity="block", rule="tool_arguments",
            message=f"Invalid arguments for tool '{name}'.",
            detail={"errors": exc.errors()},
        )


def validate_tool_call(name: str, args: dict[str, Any]) -> list[GuardrailResult]:
    """Run both tool checks (allowed-tools + argument validation)."""
    return [check_allowed_tool(name), check_tool_arguments(name, args)]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def evaluate_output(
    answer: Any,
    has_sources: bool = True,
    has_tool_results: bool = True,
) -> list[GuardrailResult]:
    """Run all answer-level output guardrails (schema -> confidence -> citations -> PII -> render).

    `has_sources` / `has_tool_results` are supplied by the agent loop (App 3):
    True when retrieval returned sources / tool results were used.
    """
    results = [check_answer_schema(answer)]
    if not isinstance(answer, AnswerSchema):
        try:
            answer = AnswerSchema.model_validate(answer)
        except ValidationError:
            return results  # schema failed; skip checks that need a valid answer
    results.append(check_confidence(answer, has_sources, has_tool_results))
    results.append(check_citations(answer))
    results.append(check_pii(answer))
    results.append(check_render_safety(answer))
    return results


def safe_fallback_answer(message: str = "The answer could not be validated. Showing a safe summary.") -> AnswerSchema:
    """Safe fallback template used when the answer is blocked or invalid (G2)."""
    return AnswerSchema(
        facts=[AnswerSection(heading="Facts", content=message, citations=[])],
        confidence="low",
        render_hint=RenderHint(kind="markdown"),
        summary=message,
    )
