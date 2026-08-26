"""App 5 — Guardrails (runtime safety).

Public API used by the agent loop (App 3):

    from apps.guardrails import evaluate_input, evaluate_output

    results = evaluate_input(user_text)            # deterministic + moderation
    results = evaluate_output(answer, has_sources, has_tool_results)

Both evaluate ALL checks then aggregate (G7): any `block` stops the step, and
`flag`s attach as annotations so the FE "Behind the scenes" tab shows
everything that was caught.
"""

from __future__ import annotations

from typing import Any

from apps.guardrails.input_guardrails import evaluate_input as _evaluate_input_deterministic
from apps.guardrails.moderation import ModerationClient, check_moderation
from apps.guardrails.output_guardrails import evaluate_output as _evaluate_output_checks
from apps.guardrails.output_guardrails import validate_tool_call
from apps.common.config import settings

__all__ = [
    "evaluate_input",
    "evaluate_output",
    "validate_tool_call",
    "GuardrailResult",
    "AnswerSchema",
]


def evaluate_input(
    text: Any,
    moderation_client: ModerationClient | None = None,
    moderation_enabled: bool | None = None,
) -> list:
    """Run all input guardrails: deterministic checks + moderation pre-check.

    Order (spec §ordering): cheap deterministic checks first, then the
    moderation LLM call. Returns the full list of GuardrailResults; callers
    stop on any `block`. `moderation_enabled` overrides the env toggle
    (used by the FE chat).
    """
    from apps.guardrails.models import GuardrailResult

    results: list[GuardrailResult] = _evaluate_input_deterministic(text)
    if not isinstance(text, str):
        return results  # already blocked on schema; skip moderation
    # Moderation is the last (LLM) check; fail-open handled inside.
    results.append(check_moderation(text, client=moderation_client, enabled=moderation_enabled))
    return results


def evaluate_output(
    answer: Any,
    has_sources: bool = True,
    has_tool_results: bool = True,
) -> list:
    """Run all answer-level output guardrails (evaluate-all, then aggregate)."""
    return _evaluate_output_checks(answer, has_sources=has_sources, has_tool_results=has_tool_results)


def any_block(results: list) -> bool:
    """True if any guardrail result blocks the step."""
    from apps.guardrails.models import GuardrailResult

    return any(r.severity == "block" for r in results if isinstance(r, GuardrailResult))
