"""App 5 — shared guardrail + answer models.

`AnswerSchema` is the single schema for the agent's structured output (App 3
imports it too), so validation and generation share one definition.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["block", "flag", "pass"]


class GuardrailResult(BaseModel):
    """One guardrail evaluation outcome."""

    passed: bool
    severity: Severity
    rule: str  # e.g. "prompt_injection"
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)  # matched patterns, counts, etc.


class AnswerSection(BaseModel):
    """One claim section of an answer. `citations` are record ids."""

    heading: str
    content: str
    citations: list[str] = Field(default_factory=list)


class RenderHint(BaseModel):
    """FE rendering hint. `kind` must be one of the known kinds."""

    kind: Literal["table", "chart", "qa", "cards", "markdown"] = "markdown"
    # kind-specific payload (e.g. chart spec, table rows); free-form for the FE.
    payload: dict[str, Any] = Field(default_factory=dict)


class AnswerSchema(BaseModel):
    """Structured answer the agent returns (validated by output guardrails).

    Sections: Facts / Interpretation / Recommendation (+ optional Others).
    Confidence is required and must be high|medium|low.
    """

    facts: list[AnswerSection] = Field(description="Evidence-backed claims with record-id citations.")
    interpretation: list[AnswerSection] = Field(default_factory=list)
    recommendation: list[AnswerSection] = Field(default_factory=list)
    other_sections: list[AnswerSection] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(description="Overall confidence.")
    render_hint: RenderHint = Field(default_factory=RenderHint)
    summary: str = Field(default="", description="One-line executive summary for the FE.")
