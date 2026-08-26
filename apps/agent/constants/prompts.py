"""Prompt templates for the agent (reason/answer synthesis).

Kept in constants/ so the LLM-facing text is reviewable in one place.
`SYSTEM_PROMPT_TEMPLATE` is formatted with `str.format(context=...)` by
apps/agent/agent.py at call time (the graph's gathered tool/retrieval context
is injected into the {context} placeholder) — never sent to the model with the
placeholder unsubstituted.

The system prompt encodes the answer discipline from docs/internal/app3-agent.md:
- Facts (with citations) / Interpretation / Recommendations / Prioritized list /
  Uncertainty & Confidence sections
- Recommendations + prioritization are first-class when asked
- compact by default, structured, render_hint for visuals
"""

SYSTEM_PROMPT_TEMPLATE = """You are a customer-intelligence analyst for a B2B SaaS company. You answer questions using ONLY the context provided below, which was gathered by deterministic tools (DuckDB queries + retrieval over support tickets and feedback). You do not fetch data yourself.

Answer discipline (mandatory):
1. Lead with the direct answer, then structure the rest.
2. Use sections: **Facts** (every claim cites record ids like CUST-0001, TCK-00042, FDB-00042), **Interpretation**, **Recommendations** (when asked or when clearly useful), **Prioritized list** (when the question asks what to do first / most important / biggest impact), **Uncertainty & Confidence**.
3. Recommendations must be concrete, actionable, and tied to the cited facts.
4. Compact by default: answer + at most 5 prioritized points; no wall of text.
5. If the question asks for prioritization or recommendations, they are REQUIRED, not optional.
6. Never invent numbers, record ids, or customers not in the context. If context is insufficient, say so and mark confidence low.
7. Confidence: high (direct evidence), medium (partial), low (insufficient/blocked).
8. Set render_hint: 'chart' for trends, 'table' for comparisons, 'cards' for customer drill-down, 'qa' for recommendations, else 'markdown'.
9. Prefer human-friendly names over bare ids: when the context includes a customer name (e.g. "CUST-0009 (LumenPartners)"), refer to the customer by name in prose, with the id on first mention for traceability (e.g. "LumenPartners (CUST-0009)"). The first mention keeps both; later mentions use the name only.

Schema grounding:
- The context includes a `=== get_catalog ===` section describing every queryable
  entity (customers, tickets, feedback, usage, subscription events, aggregates),
  its columns, and their meanings. Use it to interpret natural-language questions:
  e.g. "customers by country" -> group customer by country; "average rating by
  source" -> avg rating on feedback grouped by feedback_source.
- If the gathered data doesn't answer the question, say what additional query
  would (naming the entity/columns from the catalog) and mark confidence low.
- Never invent column names — only use columns listed in the catalog.

Semantic query tool (available when bound):
- You have a `semantic_query` tool: pass a SemanticQuery object with fields
  metric, of, of_dimension, dimensions, filters, time_range, limit.
  Metric/entity/dimension/filter values must come from the catalog (use the
  `get_catalog` tool if unsure). If the context lacks data the question needs
  and the catalog supports it, CALL semantic_query to fetch it before answering
  — never guess numbers.
- Examples: count tickets by category -> a query with metric count, of ticket,
  dimensions category; average satisfaction on resolved tickets -> metric
  satisfaction_avg, of ticket; revenue at risk -> metric revenue_at_risk, of
  customer_features.

Context:
{context}
"""

# Tells the agent how to use the bound semantic tools (appended to the system
# prompt when the tools are bound). Kept here so the instruction text is
# reviewable in one place.
SEMANTIC_TOOL_PROMPT = """
You have two tools for fetching data you need:
- `get_catalog()` — list the valid entities, metrics, dimensions, and filters.
- `semantic_query(query)` — run a validated query and get rows back.
Use them when the provided context does not answer the question but the catalog
supports it. Prefer the gathered context when it already answers the question.
"""

# Prior conversation (short-term memory) — rendered before the current context
# so follow-ups ("and what about their tickets?") resolve against earlier turns.
CONVERSATION_TEMPLATE = """
Conversation so far (earlier turns, most recent last):
{conversation}
"""


# RAG retrieved-doc formatting: one line per chunk with ids + score.
# Used by graph._format_context for the `retrieve_sources` tool result, so
# retrieved evidence reaches the LLM as citable lines (record id, customer,
# date, score) instead of a raw dict repr.
RETRIEVED_DOC_TEMPLATE = (
    "[{record_type} {record_id} | {customer_id} | {created_at} | score {score:.2f}]\n"
    '"{text}"'
)

# Tool-result formatting: table-ish key:value lines, compact.
TOOL_RESULT_TEMPLATE = "=== {tool_name} ===\n{formatted}"

# Per-tool formatter: convert a tool's {data, source_refs, warnings} into text.
# Values are rounded to 2dp so long float strings never leak into the answer
# (also avoids false PII hits on numeric-looking sequences).
def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


TOOL_RESULT_FORMATTERS = {
    "get_customer_profile": lambda r: (
        f"customer {r['customer_id']}: {r['customer_name']} ({r['customer_segment']}, {r['subscription_plan']}), "
        f"status={r['account_status']}, revenue={_fmt(r.get('monthly_revenue'))}"
    ),
    "get_usage_change": lambda r: (
        f"customer {r['customer_id']}: sessions last4wk={_fmt(r.get('sessions_last_4_weeks'))} "
        f"prev4wk={_fmt(r.get('sessions_previous_4_weeks'))} change={_fmt(r.get('sessions_change_percent'))}%"
    ),
    "get_customer_feedback": lambda r: (
        f"{r['feedback_id']} [{r.get('created_at')}] rating={_fmt(r.get('rating'))} "
        f"theme={r.get('theme')} sentiment={r.get('sentiment')}: {r.get('feedback_text')}"
    ),
    "get_customer_tickets": lambda r: (
        f"{r['ticket_id']} [{r.get('created_at')}] {r.get('category')}/{r.get('priority')}/{r.get('status')} "
        f"resolution={_fmt(r.get('resolution_time_hours'))}: {r.get('subject')}"
    ),
    "get_subscription_events": lambda r: (
        f"{r.get('event_date')} {r.get('event_type')} {r.get('previous_plan')}->{r.get('new_plan')} "
        f"revenue_change={_fmt(r.get('revenue_change'))}"
    ),
    "get_feedback_themes": lambda r: (
        f"theme={r['theme']} count={r.get('feedback_count')} (pos={r.get('positive')} neg={r.get('negative')} neu={r.get('neutral')})"
    ),
    "get_ticket_breakdown": lambda r: (
        f"category={r.get('category')} priority={r.get('priority')} status={r.get('status')} count={r.get('ticket_count')} "
        f"avg_res={_fmt(r.get('average_resolution_time_hours'))} avg_sat={_fmt(r.get('average_satisfaction_score'))}"
    ),
    "get_usage_trend": lambda r: f"{r.get('date')}: sessions={r.get('sessions')} active={r.get('active_users')} errors={r.get('errors')}",
    "calculate_segment_metrics": lambda r: (
        f"segment={r.get('segment')} plan={r.get('plan')} customers={r.get('customers')} revenue={_fmt(r.get('revenue'))} "
        f"cancel_rate={_fmt(r.get('cancel_rate'))} vs global cancel {_fmt(r.get('global_cancel_rate'))}"
    ),
    "rank_customer_risk": lambda r: (
        f"{r['customer_id']} ({r.get('customer_name', '?')}) risk={_fmt(r.get('risk_score'))} drivers={r.get('risk_drivers')}"
    ),
    "calculate_revenue_at_risk": lambda r: (
        f"revenue_at_risk={_fmt(r.get('revenue_at_risk'))} customers={r.get('at_risk_customers')} "
        f"unknown_revenue={r.get('at_risk_with_unknown_revenue')}"
    ),
    "list_customers": lambda r: (
        f"{r['customer_id']} {r.get('customer_name')} {r.get('customer_segment')} {r.get('subscription_plan')} "
        f"{r.get('account_status')} revenue={_fmt(r.get('monthly_revenue'))}"
    ),
    # Semantic layer: rows are already column->value dicts; render as key=value lines.
    "semantic_query": lambda r: ", ".join(f"{k}={_fmt(v)}" for k, v in r.items()),
    # Catalog: compact schema listing — one line per entity with its columns.
    "get_catalog": lambda r: _format_catalog(r),
}


def _format_catalog(entry: dict) -> str:
    """Render one catalog entity as a compact line for the LLM context."""
    ent = entry.get("id", "?")
    cols = entry.get("columns") or []
    col_str = "; ".join(f"{c.get('name')}: {c.get('description') or ''}" for c in cols)
    return f"entity {ent} (table {entry.get('table')}) columns: {col_str}"
