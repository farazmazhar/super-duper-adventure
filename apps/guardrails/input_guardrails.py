"""App 5 — input guardrails (deterministic; run at graph entry before the LLM).

All checks are pure functions over text — no network, no DB — so they are cheap
and reliable in the serving path. Each returns a `GuardrailResult`; callers
aggregate (evaluate-all, then act on any `block`).
"""

from __future__ import annotations

import re
from typing import Any

from apps.common.config import settings
from apps.guardrails.models import GuardrailResult

# ---------------------------------------------------------------------------
# Prompt-injection heuristics (G3: deterministic pattern list, documented).
# ---------------------------------------------------------------------------
# High-confidence: explicit attempts to override the system or reveal the
# prompt. Blocks.
HIGH_CONFIDENCE_INJECTION = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"act\s+as\s+if\s+you\s+are\s+", re.I),
    re.compile(r"(reveal|show|print|output|leak)\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"forget\s+(all\s+)?(your|the)\s+(instructions?|rules?|prompts?)", re.I),
    re.compile(r"new\s+(instructions?|rules?)\s*:", re.I),
    re.compile(r"<\|?(system|im_start|im_end)\|?>", re.I),
]

# Suspicious: injection-adjacent phrasing that could be legitimate in a
# business-QA context. Flags only.
SUSPICIOUS_INJECTION = [
    re.compile(r"do\s+not\s+(tell|mention|say)\s+(anyone|the\s+user|them)", re.I),
    re.compile(r"pretend\s+", re.I),
    re.compile(r"role\s+play", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"developer\s+mode", re.I),
    re.compile(r"override", re.I),
    re.compile(r"ignore\s+", re.I),
    re.compile(r"regardless\s+of\s+(your|the)\s+(instructions?|rules?)", re.I),
]


def _matched(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    return [p.pattern for p in patterns if p.search(text)]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_input_schema(value: Any) -> GuardrailResult:
    """Input must be a (non-empty) string; reject non-text payloads."""
    if isinstance(value, str):
        return GuardrailResult(
            passed=True, severity="pass", rule="input_schema",
            message="Input is a string.",
        )
    return GuardrailResult(
        passed=False, severity="block", rule="input_schema",
        message=f"Input must be a string (got {type(value).__name__}).",
        detail={"type": type(value).__name__},
    )


def check_length(text: str, max_chars: int | None = None) -> GuardrailResult:
    """Block input over the length cap (default from settings)."""
    limit = max_chars if max_chars is not None else settings.max_input_chars
    n = len(text)
    if n <= limit:
        return GuardrailResult(
            passed=True, severity="pass", rule="length_cap",
            message=f"Input length {n} <= {limit}.",
        )
    return GuardrailResult(
        passed=False, severity="block", rule="length_cap",
        message=f"Input too long ({n} chars; cap {limit}).",
        detail={"length": n, "cap": limit},
    )


def check_prompt_injection(text: str) -> GuardrailResult:
    """Heuristic injection detection: block high-confidence, flag suspicious."""
    high = _matched(HIGH_CONFIDENCE_INJECTION, text)
    if high:
        return GuardrailResult(
            passed=False, severity="block", rule="prompt_injection",
            message="Prompt-injection pattern detected.",
            detail={"high_confidence": high},
        )
    suspicious = _matched(SUSPICIOUS_INJECTION, text)
    if suspicious:
        return GuardrailResult(
            passed=False, severity="flag", rule="prompt_injection",
            message="Suspicious prompt-injection pattern.",
            detail={"suspicious": suspicious},
        )
    return GuardrailResult(
        passed=True, severity="pass", rule="prompt_injection",
        message="No injection patterns detected.",
    )


def evaluate_input(text: Any) -> list[GuardrailResult]:
    """Run all deterministic input guardrails (schema -> length -> injection).

    Order: cheap deterministic checks first; the moderation LLM call is separate
    (moderation.py) and added by the aggregator in __init__.
    """
    results = [check_input_schema(text)]
    if not isinstance(text, str):
        return results  # non-string: blocked; further checks are meaningless
    results.append(check_length(text))
    results.append(check_prompt_injection(text))
    return results
