"""App 3 — LangGraph graph: classify -> route -> node -> reason -> answer.

Implements docs/internal/app3-agent.md:
- classify (rule-based intent + entities)
- route (intent -> specialized node; blocks irrelevant)
- specialized nodes (customer/themes/analytics/trend/rag/general) gathering
  context via MCP tools
- reason (PydanticAI agent synthesizes from gathered context only)
- answer (structured AnswerSchema + render_hint + confidence + trace)
- error_handler (trace + retry once for transient errors + safe message)

State is a plain dict (LangGraph StateGraph with TypedDict-ish keys).
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from apps.agent.classify import classify
from apps.agent.constants.routing import (
    BLOCKED_REPLY,
    ERROR_REPLY_TEMPLATE,
    IRRELEVANT_REPLY,
    LOW_CONFIDENCE_NOTE,
    NODE_TOOLS,
    ROUTE_MAP,
)
from apps.agent.mcp_client import McpClient
from apps.agent.tracing import TraceContext
from apps.guardrails.models import AnswerSchema, AnswerSection, RenderHint

logger = logging.getLogger(__name__)

# Transient errors worth one retry (spec §6 error_handler).
TRANSIENT_ERRORS = ("rate limit", "timeout", "timed out", "429", "503", "502", "overloaded")


def _is_transient(message: str) -> bool:
    m = message.lower()
    return any(t in m for t in TRANSIENT_ERRORS)


class GraphState(TypedDict, total=False):
    """LangGraph state schema — every key the nodes read/write."""

    question: str
    classification: str
    entities: dict[str, Any]
    plan: dict[str, Any] | None  # semantic-query plan from classify
    routed_node: str
    tool_results: dict[str, dict[str, Any]]
    context: str
    answer: AnswerSchema
    trace: TraceContext
    retried: bool
    error: str | None
    retrieve_kwargs: dict[str, Any]  # per-call overrides for retrieve_sources
    conversation: list[dict[str, str]]  # short-term context (prior turns)


class AgentGraph:
    """The LangGraph pipeline. `mcp` is the agent's only data path."""

    def __init__(self, mcp: McpClient, reason_agent: Any | None = None) -> None:
        self.mcp = mcp
        self.reason_agent = reason_agent  # PydanticAI agent (or a stub in tests)
        self.graph = self._build()

    # -- node implementations -------------------------------------------------
    def _trace(self, state: dict[str, Any]) -> TraceContext | None:
        return state.get("trace")

    def _classify_node(self, state: dict[str, Any]) -> dict[str, Any]:
        question = state.get("question", "")
        span = self._trace(state).start_span("classify", kind="node") if self._trace(state) else None
        result = classify(question)
        if span is not None:
            span.end(intent=result["intent"], entities=result["entities"])
        state["classification"] = result["intent"]
        state["entities"] = result["entities"]
        state["plan"] = result.get("plan")
        return {
            "classification": result["intent"],
            "entities": result["entities"],
            "plan": result.get("plan"),
        }

    def _route_node(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = state.get("classification", "general")
        node = ROUTE_MAP.get(intent, "general")
        trace = self._trace(state)
        if trace is not None:
            trace.start_span("route", kind="node", intent=intent, routed_node=node).end()
        state["routed_node"] = node
        return {"routed_node": node}

    def _gather_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the routed node's MCP tool calls and write context into state."""
        node = state.get("routed_node", "general")
        entities = dict(state.get("entities", {}))
        customer_ids = list(entities.get("customer_ids") or [])
        segment = entities.get("segment")
        plan = state.get("plan")

        # Resolve a customer-name hint to ids (query flexibility: "VertexPath A").
        name_hint = entities.get("customer_name_hint")
        if name_hint and not customer_ids:
            resolved = self.mcp.call_tool("resolve_customer_name", {"name": name_hint})
            rows = resolved.get("data") or []
            customer_ids = [r["customer_id"] for r in rows if r.get("customer_id")]
            if customer_ids:
                entities["customer_ids"] = customer_ids
                state["entities"] = entities

        # The catalog gives the reasoner the full schema (entities, columns,
        # descriptions) so it can build semantic queries for arbitrary NL
        # questions — and gives the RAG path the same grounding.
        catalog_result = self.mcp.call_tool("get_catalog", {})

        # Trend questions without a named customer: resolve a sample of customers
        # to trend (the usage-trend tool is per-customer).
        if node == "trend" and not customer_ids:
            listing = self.mcp.call_tool("list_customers", {"limit": 3, "segment": segment})
            rows = listing.get("data") or []
            customer_ids = [r["customer_id"] for r in rows if r.get("customer_id")]

        calls: list[tuple[str, dict[str, Any]]] = []
        for tool in NODE_TOOLS.get(node, []):
            args: dict[str, Any] = {}
            if tool == "get_customer_profile" and customer_ids:
                args["customer_id"] = customer_ids[0]
            elif tool == "get_customer_tickets" and customer_ids:
                args["customer_id"] = customer_ids[0]
            elif tool == "get_customer_feedback" and customer_ids:
                args["customer_id"] = customer_ids[0]
            elif tool == "get_usage_change":
                if customer_ids:
                    args["customer_id"] = customer_ids[0]
                if segment:
                    args["segment"] = segment
            elif tool == "get_usage_trend":
                # Only callable per-customer; skip when no customer is in scope
                # (fixes the "get_usage_trend missing customer_id" tool error).
                if customer_ids:
                    args["customer_id"] = customer_ids[0]
                    args["weeks"] = 8
                else:
                    continue
            elif tool == "get_subscription_events" and customer_ids:
                args["customer_id"] = customer_ids[0]
            elif tool == "list_customers":
                if segment:
                    args["segment"] = segment
            elif tool == "calculate_segment_metrics":
                if segment:
                    args["segment"] = segment
            elif tool == "retrieve_sources":
                args["query"] = state.get("question", "")
                args["k"] = 20
                if customer_ids:
                    args["filters"] = {"customer_id": customer_ids[0]}
                # Per-call override from the caller (e.g. the FE chat rerank toggle).
                args.update(state.get("retrieve_kwargs") or {})
            elif tool == "semantic_query":
                # Use the classify-stage query plan when present (e.g. "count
                # customers by country"); fall back to a sensible default.
                query = plan or {
                    "metric": "count",
                    "of": "ticket",
                    "dimensions": ["category"],
                }
                if segment and "filters" not in query:
                    query = {**query, "filters": {"customer_segment": segment}}
                args["query"] = query
            calls.append((tool, args))

        trace = self._trace(state)
        span = trace.start_span(f"node:{node}", kind="node") if trace is not None else None
        results = self.mcp.call_tools(calls) if calls else {}
        # Always surface the catalog to the reasoner (schema grounding for NL).
        results["get_catalog"] = catalog_result
        if span is not None:
            span.end(
                tools=list(results.keys()),
                result=_results_summary(results),
            )
        return {"tool_results": results, "context": _format_context(results)}

    def _reason_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """PydanticAI agent (or deterministic stub) synthesizes the answer.

        For blocked questions the answer is already produced (no reason step);
        we pass it through unchanged.
        """
        if state.get("routed_node") == "blocked" and isinstance(state.get("answer"), AnswerSchema):
            return {"answer": state["answer"]}
        context = state.get("context", "")
        question = state.get("question", "")
        trace = self._trace(state)
        span = trace.start_span("reason", kind="llm") if trace is not None else None
        if self.reason_agent is None:
            # Deterministic fallback for tests / no-key: summarize the context.
            answer = _rule_based_answer(question, context)
        else:
            result = self.reason_agent.run_sync(
                question,
                deps={"context": context, "conversation": state.get("conversation") or []},
            )
            answer = result.output
            # Capture LLM token usage into the span for the FE "Behind the scenes" panel.
            usage = getattr(result, "usage", None)
            if usage is not None and span is not None:
                span.end(
                    status="ok",
                    used_llm=True,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                    cache_read_tokens=getattr(usage, "cache_read_tokens", None),
                    cache_write_tokens=getattr(usage, "cache_write_tokens", None),
                    cost=getattr(usage, "cost", None),
                    requests=getattr(usage, "requests", None),
                    tool_calls=getattr(usage, "tool_calls", None),
                )
        if span is not None and self.reason_agent is None:
            span.end(status="ok", used_llm=False)
        return {"answer": answer}

    def _answer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Assemble the final structured answer + trace.

        Attaches a structured render_hint.payload from the gathered tool results
        (chart for trends, tables for risk/breakdowns, cards for profiles) so the
        FE and CLI render visuals instead of raw text.
        """
        answer = state.get("answer")
        trace: TraceContext | None = state.get("trace")
        if not isinstance(answer, AnswerSchema):
            answer = _rule_based_answer(state.get("question", ""), state.get("context", ""))

        # Attach the visual payload (unless blocked — no visuals for those).
        if state.get("routed_node") != "blocked":
            from apps.agent.render import hint_from_results

            hint = hint_from_results(
                state.get("routed_node", "general"),
                state.get("tool_results") or {},
                state.get("question", ""),
            )
            answer.render_hint = RenderHint(
                kind=hint.get("kind", "markdown"),
                payload=hint,
            )

        if trace is not None:
            trace.start_span("answer", kind="node", summary=answer.summary[:120]).end()
            trace.add_event("answer_ready", {"summary": answer.summary[:120]})
            trace_payload = trace.to_dict()
        else:
            trace_payload = {}
        return {"answer": answer, "trace": trace_payload}

    def _blocked_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Irrelevant / blocked: short bounded reply, no visuals, low confidence."""
        intent = state.get("classification", "general")
        reply = IRRELEVANT_REPLY if intent == "irrelevant" else BLOCKED_REPLY
        answer = AnswerSchema(
            facts=[AnswerSection(heading="Facts", content=reply, citations=[])],
            interpretation=[],
            recommendation=[],
            other_sections=[],
            confidence="low",
            render_hint=RenderHint(kind="markdown"),
            summary=reply,
        )
        return {"answer": answer, "routed_node": "blocked"}

    def _error_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Catch node errors: trace, retry once for transient, else safe message."""
        error = state.get("error")
        trace: TraceContext | None = state.get("trace")
        message = str(error) if error else "unknown error"
        if trace is not None:
            trace.add_event("error", {"message": message})

        # retry once for transient errors
        if _is_transient(message) and not state.get("retried"):
            state["retried"] = True
            state.pop("error", None)
            # re-run the failing node (routed_node still set)
            return {"retried": True, "error": None}

        safe = ERROR_REPLY_TEMPLATE.format(reason=message[:200])
        answer = AnswerSchema(
            facts=[AnswerSection(heading="Facts", content=safe, citations=[])],
            interpretation=[],
            recommendation=[],
            other_sections=[],
            confidence="low",
            render_hint=RenderHint(kind="markdown"),
            summary=safe,
        )
        return {"answer": answer, "routed_node": "error"}

    # -- graph construction ----------------------------------------------------
    def _build(self) -> StateGraph:
        g = StateGraph(GraphState)

        g.add_node("classify", self._classify_node)
        g.add_node("route", self._route_node)
        g.add_node("customer", self._gather_node)
        g.add_node("themes", self._gather_node)
        g.add_node("analytics", self._gather_node)
        g.add_node("trend", self._gather_node)
        g.add_node("rag", self._gather_node)
        g.add_node("general", self._gather_node)
        g.add_node("blocked", self._blocked_node)
        g.add_node("reason", self._reason_node)
        g.add_node("answer", self._answer_node)
        g.add_node("error", self._error_node)

        g.add_edge(START, "classify")
        g.add_edge("classify", "route")
        # route returns {"routed_node": node}; the conditional edge dispatches on it.
        g.add_conditional_edges("route", self._route_condition, {
            "customer": "customer", "themes": "themes", "analytics": "analytics",
            "trend": "trend", "rag": "rag", "general": "general",
            "blocked": "blocked",  # blocked node produces the reply, reason passes it through
        })
        for node in ("customer", "themes", "analytics", "trend", "rag", "general", "blocked"):
            g.add_edge(node, "reason")
        g.add_edge("reason", "answer")
        g.add_edge("answer", END)
        g.add_edge("error", "answer")  # error -> answer with safe message
        return g.compile()

    def _route_condition(self, state: dict[str, Any]) -> str:
        # LangGraph passes the node's returned dict (or the merged state).
        node = state.get("routed_node", "general") if isinstance(state, dict) else str(state)
        return node if node in NODE_TOOLS or node == "blocked" else "general"

    # -- entry -----------------------------------------------------------------
    def run(
        self,
        question: str,
        trace: TraceContext | None = None,
        retrieve_kwargs: dict[str, Any] | None = None,
        conversation: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run the graph for one question; returns final state (with answer + trace).

        `retrieve_kwargs` are merged into the `retrieve_sources` tool call
        (e.g. {"rerank_enabled": False}) — used by the FE chat toggles.
        `conversation` is the short-term context: prior turns as
        [{"role": "user"|"assistant", "content": "..."}] so follow-ups like
        "and what about their tickets?" resolve against earlier turns.
        """
        trace = trace or TraceContext(question=question)
        initial: dict[str, Any] = {
            "question": question,
            "trace": trace,
            "classification": None,
            "entities": {},
            "routed_node": None,
            "tool_results": {},
            "context": "",
            "answer": None,
            "retried": False,
            "error": None,
            "retrieve_kwargs": retrieve_kwargs or {},
            "conversation": conversation or [],
        }
        try:
            result = self.graph.invoke(initial)
        except Exception as exc:  # noqa: BLE001 - last-resort catch
            logger.exception("graph failed")
            trace.add_event("error", {"message": str(exc)})
            result = dict(initial)
            result["answer"] = _rule_based_answer(question, "", error=str(exc))
            result["trace"] = trace.to_dict()
            return result
        result["trace"] = trace.to_dict()
        return result


# ---------------------------------------------------------------------------
# Deterministic reason fallback (no LLM / tests). Builds an AnswerSchema from
# the gathered context with a simple extraction of the first facts.
# ---------------------------------------------------------------------------
def _rule_based_answer(question: str, context: str, error: str | None = None) -> AnswerSchema:
    if error:
        safe = ERROR_REPLY_TEMPLATE.format(reason=error[:200])
        return AnswerSchema(
            facts=[AnswerSection(heading="Facts", content=safe, citations=[])],
            confidence="low",
            render_hint=RenderHint(kind="markdown"),
            summary=safe,
        )
    if not context:
        return AnswerSchema(
            facts=[AnswerSection(heading="Facts", content=LOW_CONFIDENCE_NOTE, citations=[])],
            confidence="low",
            render_hint=RenderHint(kind="markdown"),
            summary=LOW_CONFIDENCE_NOTE,
        )
    # extract first context lines as pseudo-facts (deterministic baseline)
    lines = [ln for ln in context.splitlines() if ln.strip()][:8]
    return AnswerSchema(
        facts=[AnswerSection(heading="Facts", content="\n".join(lines), citations=[])],
        interpretation=[AnswerSection(heading="Interpretation", content="(deterministic fallback — no LLM configured)")],
        recommendation=[],
        confidence="medium" if lines else "low",
        render_hint=RenderHint(kind="markdown"),
        summary=lines[0][:150] if lines else "No data returned.",
    )


def _format_catalog_result(data: dict[str, Any]) -> str:
    """Render the semantic catalog for the LLM: entities + columns + dims."""
    from apps.agent.constants.prompts import _format_catalog

    lines = ["=== get_catalog ==="]
    for ent in data.get("entities") or []:
        lines.append(_format_catalog(ent))
    dims = data.get("dimensions") or []
    if dims:
        lines.append("groupable dimensions: " + ", ".join(dims))
    metrics = data.get("metrics") or []
    if metrics:
        lines.append("metrics: " + ", ".join(metrics))
    return "\n".join(lines)


def _results_summary(results: dict[str, dict[str, Any]], max_rows: int = 5) -> dict[str, Any]:
    """Compact, JSON-able summary of tool results for the trace (row counts +
    a few sample rows so the FE "Behind the scenes" can show what each tool
    returned without dumping megabytes)."""
    out: dict[str, Any] = {}
    for tool, res in results.items():
        data = res.get("data")
        if isinstance(data, list):
            sample = data[:max_rows]
            out[tool] = {"row_count": len(data), "sample": sample}
        elif isinstance(data, dict):
            # catalog: just note entity count
            if tool == "get_catalog":
                out[tool] = {"entities": len(data.get("entities") or []), "dimensions": len(data.get("dimensions") or [])}
            else:
                out[tool] = data
        else:
            out[tool] = {"value": data, "warnings": res.get("warnings", [])[:3]}
    return out


def _format_context(results: dict[str, dict[str, Any]]) -> str:
    """Flatten tool results into text for the LLM context."""
    from apps.agent.constants.prompts import RETRIEVED_DOC_TEMPLATE, TOOL_RESULT_FORMATTERS, _fmt

    parts: list[str] = []
    for tool_name, res in results.items():
        data = res.get("data")
        if not data:
            continue

        # RAG chunks: render each retrieved doc as a citable line (id, customer,
        # date, score) so the LLM can cite it and the evidence is readable.
        if tool_name == "retrieve_sources" and isinstance(data, list):
            for doc in data:
                if isinstance(doc, dict):
                    try:
                        parts.append(RETRIEVED_DOC_TEMPLATE.format(**doc))
                    except (KeyError, ValueError):
                        parts.append(f"retrieve_sources: {doc}")
            continue

        # Catalog: render the full schema compactly (entities + columns + dims).
        if tool_name == "get_catalog" and isinstance(data, dict):
            parts.append(_format_catalog_result(data))
            continue

        # Revenue at risk: render the summary AND the per-customer list with
        # names, so the LLM can refer to customers by name, not just ids.
        if tool_name == "calculate_revenue_at_risk" and isinstance(data, dict):
            fmt = TOOL_RESULT_FORMATTERS.get(tool_name)
            if fmt:
                try:
                    parts.append(fmt(data))
                except (KeyError, ValueError):
                    parts.append(f"{tool_name}: {data}")
            for c in data.get("customers") or []:
                if isinstance(c, dict):
                    parts.append(
                        f"at-risk customer {c.get('customer_id')} ({c.get('customer_name', '?')}) "
                        f"status={c.get('account_status')} revenue={_fmt(c.get('monthly_revenue'))} risk={_fmt(c.get('risk_score'))}"
                    )
            continue

        fmt = TOOL_RESULT_FORMATTERS.get(tool_name)
        if fmt:
            if isinstance(data, dict):
                try:
                    parts.append(fmt(data))
                except (KeyError, ValueError):
                    parts.append(f"{tool_name}: {data}")
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        try:
                            parts.append(fmt(item))
                        except (KeyError, ValueError):
                            parts.append(f"{tool_name}: {item}")
            else:
                parts.append(f"{tool_name}: {data}")
        else:
            parts.append(f"{tool_name}: {data}")
    return "\n".join(parts)
