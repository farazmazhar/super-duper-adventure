"""App 6 — Layer 2 LLM-judge (togglable, currently skipped).

Per the user's direction, the LLM-as-judge is **not implemented yet**: the
evaluator runs deterministic Layer-1 metrics only, and the report notes
"judge skipped". This module is the seam where a faithfulness/usefulness judge
can be added later (same OpenRouter client as the agent), gated by
`EVAL_JUDGE_ENABLED` and fail-open on missing key.
"""

from __future__ import annotations

import os
from typing import Any


def judge_enabled() -> bool:
    """Whether the LLM-judge should run (env-gated; default off for now)."""
    return os.environ.get("EVAL_JUDGE_ENABLED", "false").lower() in ("1", "true", "yes")


def judge_question(golden: Any, answer: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Score one question with the LLM-judge.

    Not implemented (user decision): returns a skipped marker. When implemented,
    it should call the shared OpenRouter client with {question, answer,
    retrieved/tool context} and a fixed rubric, returning {faithfulness,
    usefulness, explanation}.
    """
    return {"skipped": True, "note": "LLM-as-judge not implemented (EVAL_JUDGE_ENABLED off by design)."}
