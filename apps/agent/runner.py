"""App 3 — shared question runner (CLI + FE).

One implementation of the guarded question path: input guardrails -> graph ->
output guardrails. The FE (App 7) imports `run_question` — the one sanctioned
cross-app import — and the CLI uses it too, so both surface the same behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.agent.graph import AgentGraph  # noqa: E402
from apps.agent.tracing import TraceContext  # noqa: E402
from apps.guardrails import evaluate_input  # noqa: E402
from apps.guardrails.models import AnswerSchema, AnswerSection, RenderHint  # noqa: E402
from apps.guardrails.output_guardrails import evaluate_output  # noqa: E402


def run_question(
    graph: AgentGraph,
    question: str,
    trace: TraceContext | None = None,
    retrieve_kwargs: dict[str, Any] | None = None,
    moderation_enabled: bool | None = None,
    conversation: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run input guardrails -> graph -> output guardrails. Returns the final state.

    - `retrieve_kwargs` (e.g. {"rerank_enabled": False}) forwarded to the graph.
    - `moderation_enabled` overrides the env toggle for the moderation pre-check.
    - `conversation` = prior turns ([{role, content}]) for short-term context;
      forwarded to the graph so follow-ups resolve against earlier turns.
    """
    results = evaluate_input(question, moderation_enabled=moderation_enabled)
    blocked = [r for r in results if r.severity == "block"]
    trace = trace or TraceContext(question=question)
    if blocked:
        trace.add_event("guardrail_blocked", {"rules": [r.rule for r in blocked]})
        from apps.agent.constants.routing import BLOCKED_REPLY

        answer = AnswerSchema(
            facts=[AnswerSection(heading="Facts", content=BLOCKED_REPLY, citations=[])],
            confidence="low",
            render_hint=RenderHint(kind="markdown"),
            summary=BLOCKED_REPLY,
        )
        return {"answer": answer, "trace": trace.to_dict(), "guardrails": [r.model_dump() for r in results]}

    state = graph.run(question, trace=trace, retrieve_kwargs=retrieve_kwargs, conversation=conversation)
    answer = state.get("answer")
    gr = evaluate_output(
        answer,
        has_sources=bool(state.get("context")),
        has_tool_results=bool(state.get("tool_results")),
    )
    state["guardrails"] = [r.model_dump() for r in results + gr]
    return state
