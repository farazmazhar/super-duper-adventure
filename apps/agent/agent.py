"""App 3 — PydanticAI agent wiring.

Builds the single PydanticAI Agent used by the `reason` node. The agent does
NOT call data tools via cross-app imports — it reaches data only through the
shared MCP client (the same stdio path the graph uses). It is bound two tools:

- `semantic_query(query)` — the App 4 semantic layer: the agent expresses any
  question the catalog supports as a structured SemanticQuery, the layer
  validates/translates/executes read-only, and the result feeds the answer.
- `get_catalog()` — returns the semantic catalog (entities/metrics/dimensions)
  so the agent can map natural language onto valid query fields at runtime.

Guardrails (input + output) are wired in by the graph/CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402
from pydantic_ai.tools import RunContext  # noqa: E402

from apps.agent.constants.prompts import (
    CONVERSATION_TEMPLATE,
    SEMANTIC_TOOL_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
)
from apps.common.config import settings  # noqa: E402
from apps.guardrails.models import AnswerSchema  # noqa: E402


def build_reason_agent(mcp: Any | None = None) -> Agent:
    """Create the PydanticAI agent producing an AnswerSchema from context.

    The per-question `context` (the tool/retrieval evidence the graph gathered)
    is injected into the system prompt at call time via a registered
    `@agent.system_prompt` function reading `deps["context"]`. The graph passes
    it as `run_sync(question, deps={"context": context})`. Without this the
    model would never see the data — the {context} placeholder would reach the
    model verbatim (the bug this fixes).

    When `mcp` (an `McpClient`) is provided, the agent is also bound the
    `semantic_query` + `get_catalog` tools so it can fetch additional data the
    routed node didn't gather — through the same MCP transport (no cross-app
    imports).
    """
    model = OpenAIChatModel(
        settings.openai_model,
        provider=OpenAIProvider(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key or "missing-key",
        ),
    )

    agent = Agent(
        model,
        output_type=AnswerSchema,
        deps_type=dict[str, Any],
    )

    @agent.system_prompt
    def _with_context(ctx: RunContext[dict[str, Any]]) -> str:
        context = ctx.deps.get("context") or "No data gathered for this question."
        conv = ctx.deps.get("conversation") or []
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        if conv:
            lines = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in conv[-6:])
            prompt += CONVERSATION_TEMPLATE.format(conversation=lines)
        return prompt

    if mcp is not None:
        _bind_semantic_tools(agent, mcp)

    return agent


def _bind_semantic_tools(agent: Agent, mcp: Any) -> None:
    """Bind the semantic-layer tools to the agent, backed by the MCP client.

    The tools call the shared `McpClient` (the agent's only data path) — the
    implementation lives in App 4 (apps/mcp/semantic.py); this is a thin
    transport wrapper, no cross-app import of the MCP internals. Sync functions:
    `McpClient.call_tool` drives the background event loop.
    """

    @agent.tool
    def semantic_query(ctx: RunContext[dict[str, Any]], query: dict[str, Any]) -> dict[str, Any]:
        """Query the data through the semantic layer.

        `query` = {metric, of, of_dimension?, dimensions?, filters?, time_range?,
        limit?}. Fields must come from the semantic catalog (see get_catalog).
        Executes read-only. Returns {data, columns, warnings, source_refs}.
        Use this to answer any question the fixed tools did not cover.
        """
        return mcp.call_tool("semantic_query", {"query": query})

    @agent.tool
    def get_catalog(ctx: RunContext[dict[str, Any]]) -> dict[str, Any]:
        """Return the semantic catalog: entities, metrics, dimensions, filters.

        Use this first when you need to build a semantic_query — it tells you
        which metric/entity/dimension/filter values are valid.
        """
        return mcp.call_tool("get_catalog", {})


def build_reason_agent_deps() -> dict[str, Any]:
    """Deps payload passed to the agent (currently just the repo root)."""
    return {"repo_root": REPO_ROOT}
