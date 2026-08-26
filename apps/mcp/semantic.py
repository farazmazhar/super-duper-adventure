"""App 4 — semantic layer: catalog + SemanticQuery validation + SQL translation.

Replaces the old raw-SQL escape hatch (`run_sql_query`). The agent expresses a
query as a structured `SemanticQuery` (metric / entity / dimensions / filters /
time_range / limit) over a **curated catalog**; this layer validates every field
against the catalog (unknown values are rejected, never passed through),
translates to parameterized DuckDB SQL, and executes read-only.

The catalog is curated from App 1's tables; keep it in sync with them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.common import config as common_config  # noqa: E402

MAX_ROWS = 100
TIMEOUT_SECONDS = 5

# ---------------------------------------------------------------------------
# Column descriptions (for the catalog tool — lets the agent map natural
# language to columns at runtime). Keys are (entity, column) -> description.
# ---------------------------------------------------------------------------
COLUMN_DESCRIPTIONS: dict[tuple[str, str], str] = {
    # customer
    ("customer", "customer_id"): "Unique customer identifier (CUST-xxxx).",
    ("customer", "customer_name"): "Customer display name.",
    ("customer", "customer_segment"): "Segment: SMB, Mid-Market, or Enterprise.",
    ("customer", "country"): "Customer country.",
    ("customer", "subscription_plan"): "Plan: Free, Team, Business, Enterprise.",
    ("customer", "monthly_revenue"): "Monthly recurring revenue (NULL = unknown, not imputed).",
    ("customer", "account_created_at"): "Date the account was created.",
    ("customer", "account_status"): "active, paused, or canceled.",
    # ticket
    ("ticket", "ticket_id"): "Unique ticket identifier (TCK-xxxx).",
    ("ticket", "customer_id"): "Customer the ticket belongs to.",
    ("ticket", "created_at"): "When the ticket was created.",
    ("ticket", "subject"): "Ticket subject line.",
    ("ticket", "message"): "Ticket message body (may be NULL).",
    ("ticket", "category"): "Ticket category: billing, bug, feature_request, general_question, technical_issue, onboarding, account_access.",
    ("ticket", "priority"): "low, medium, high, urgent.",
    ("ticket", "resolution_time_hours"): "Hours to resolution (NULL if unresolved).",
    ("ticket", "status"): "open or resolved.",
    ("ticket", "satisfaction_score"): "1-5 satisfaction (NULL if unresolved).",
    # feedback
    ("feedback", "feedback_id"): "Unique feedback identifier (FDB-xxxx).",
    ("feedback", "customer_id"): "Customer who gave feedback.",
    ("feedback", "created_at"): "When feedback was given.",
    ("feedback", "feedback_text"): "Free-text feedback (may be NULL).",
    ("feedback", "feedback_source"): "support_chat, email, nps_survey, app_store_review.",
    ("feedback", "rating"): "1-5 rating (normalized from 10-point).",
    # usage
    ("usage", "customer_id"): "Customer the usage row belongs to.",
    ("usage", "date"): "Usage date.",
    ("usage", "active_users"): "Active users that day.",
    ("usage", "sessions"): "Sessions that day.",
    ("usage", "feature_usage"): "Feature used (e.g. search, dashboards).",
    ("usage", "errors"): "Errors that day.",
    ("usage", "average_session_duration"): "Avg session minutes (NULL if missing).",
    # subscription events
    ("subscription_event", "customer_id"): "Customer the event belongs to.",
    ("subscription_event", "event_date"): "Event date.",
    ("subscription_event", "event_type"): "upgrade, downgrade, cancellation, renewal.",
    ("subscription_event", "previous_plan"): "Plan before the event.",
    ("subscription_event", "new_plan"): "Plan after the event.",
    ("subscription_event", "revenue_change"): "Monthly revenue delta from the event.",
    # customer_features (aggregates)
    ("customer_features", "customer_id"): "Customer identifier.",
    ("customer_features", "churn_proxy"): "True when account_status is paused (canceled customers are already lost, not at risk).",
    ("customer_features", "ticket_count"): "Total tickets for the customer.",
    ("customer_features", "tickets_open"): "Open tickets.",
    ("customer_features", "tickets_urgent"): "Urgent tickets.",
    ("customer_features", "average_resolution_time_hours"): "Avg ticket resolution time.",
    ("customer_features", "average_satisfaction_score"): "Avg ticket satisfaction.",
    ("customer_features", "sessions_last_4_weeks"): "Sessions in the last 4 weeks.",
    ("customer_features", "sessions_previous_4_weeks"): "Sessions in the prior 4 weeks.",
    ("customer_features", "sessions_change_percent"): "Session change % (last vs prior 4wk).",
    ("customer_features", "errors_total"): "Total errors in usage.",
    ("customer_features", "feedback_count"): "Feedback count.",
    ("customer_features", "average_rating"): "Average feedback rating.",
    ("customer_features", "cancellations"): "Cancellation events.",
    ("customer_features", "downgrades"): "Downgrade events.",
    ("customer_features", "upgrades"): "Upgrade events.",
    ("customer_features", "renewals"): "Renewal events.",
    ("customer_features", "revenue_change_sum"): "Net revenue change from plan events.",
    ("customer_features", "last_event_date"): "Most recent subscription event date.",
    ("customer_features", "last_event_type"): "Most recent event type.",
    # segment_metrics
    ("segment_metrics", "segment"): "Customer segment (or NULL for plan rows).",
    ("segment_metrics", "plan"): "Subscription plan (or NULL for segment rows).",
    ("segment_metrics", "customers"): "Customers in the group.",
    ("segment_metrics", "revenue"): "Total monthly revenue of the group.",
    ("segment_metrics", "ticket_count"): "Total tickets in the group.",
    ("segment_metrics", "complaint_rate"): "Share of customers with urgent/bug/billing tickets.",
    ("segment_metrics", "average_resolution_time_hours"): "Avg resolution time.",
    ("segment_metrics", "average_satisfaction_score"): "Avg satisfaction.",
    ("segment_metrics", "cancel_rate"): "Cancel rate vs global benchmark.",
}


def get_catalog() -> dict[str, Any]:
    """Return the full semantic catalog: entities, columns, and descriptions.

    Lets the agent (and RAG) map natural-language questions to queryable
    entities/columns at runtime. Returns the standard {data, source_refs,
    warnings} contract; data = {entities, dimensions, metrics}.
    """
    entities_out = []
    for entity, table in ENTITIES.items():
        # Column set: base columns come from the entity's table; derive a
        # reasonable list from the known catalog (dimensions/numeric/filters).
        columns = [{"name": c, "description": COLUMN_DESCRIPTIONS.get((entity, c), "")} for c in _entity_columns(entity)]
        entities_out.append({"id": entity, "table": table, "columns": columns})
    return {
        "data": {
            "entities": entities_out,
            "dimensions": sorted(DIMENSIONS),
            "metrics": sorted(METRICS),
        },
        "source_refs": [],
        "warnings": [],
    }


def _entity_columns(entity: str) -> list[str]:
    """Best-effort column list for an entity (from the catalog + descriptions)."""
    known = [c for (e, c) in COLUMN_DESCRIPTIONS if e == entity]
    # Add record filters / dimensions that apply to the entity.
    extra = {
        "customer": ["customer_id", "customer_name", "country", "customer_segment", "subscription_plan", "account_status", "monthly_revenue", "account_created_at"],
        "ticket": ["ticket_id", "customer_id", "created_at", "subject", "message", "category", "priority", "status", "resolution_time_hours", "satisfaction_score"],
        "feedback": ["feedback_id", "customer_id", "created_at", "feedback_text", "feedback_source", "rating"],
        "usage": ["customer_id", "date", "active_users", "sessions", "feature_usage", "errors", "average_session_duration"],
        "subscription_event": ["customer_id", "event_date", "event_type", "previous_plan", "new_plan", "revenue_change"],
        "customer_features": ["customer_id"],
        "segment_metrics": ["segment", "plan"],
    }
    cols = known or extra.get(entity, [])
    # de-dup preserving order
    return list(dict.fromkeys(cols))

# ---------------------------------------------------------------------------
# Semantic catalog (curated from App 1's tables — docs/internal/app4-mcp.md)
# ---------------------------------------------------------------------------

# Entity -> base table. Facts are the primary entities; `customer_features` and
# `segment_metrics` expose the precomputed aggregates.
ENTITIES: dict[str, str] = {
    "customer": "main.dimension_customer",
    "ticket": "main.fact_ticket",
    "feedback": "main.fact_feedback",
    "usage": "main.fact_usage",
    "subscription_event": "main.fact_subscription_event",
    "customer_features": "main.aggregate_customer_features",
    "segment_metrics": "main.aggregate_segment_metrics",
}

# Dimension -> SQL expression (validated group-by / filter keys).
DIMENSIONS: dict[str, str] = {
    "customer_id": "customer_id",
    "customer_segment": "customer_segment",
    "subscription_plan": "subscription_plan",
    "account_status": "account_status",
    "country": "country",
    "category": "category",
    "priority": "priority",
    "status": "status",
    "feedback_source": "feedback_source",
    "feature_usage": "feature_usage",
    "event_type": "event_type",
    # Derived date dimensions.
    "date": "date",
    "week": "date_trunc('week', date)::DATE",
    "created_week": "date_trunc('week', created_at)::DATE",
    "event_week": "date_trunc('week', event_date)::DATE",
}

# Entities that carry their own date column for time-range filtering.
ENTITY_DATE_COLUMN = {
    "customer": "account_created_at",
    "ticket": "created_at",
    "feedback": "created_at",
    "usage": "date",
    "subscription_event": "event_date",
    "customer_features": "last_event_date",
    "segment_metrics": None,
}

# Numeric columns usable with sum/avg/min/max (per entity).
NUMERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "customer": ("monthly_revenue",),
    "ticket": ("resolution_time_hours", "satisfaction_score"),
    "feedback": ("rating",),
    "usage": ("active_users", "sessions", "errors", "average_session_duration"),
    "subscription_event": ("revenue_change",),
    "customer_features": (
        "monthly_revenue", "ticket_count", "sessions_last_4_weeks",
        "sessions_previous_4_weeks", "sessions_change_percent", "errors_total",
        "average_session_duration", "feedback_count", "average_rating",
        "cancellations", "downgrades", "upgrades", "renewals", "revenue_change_sum",
    ),
    "segment_metrics": ("customers", "revenue", "ticket_count", "complaint_rate", "cancel_rate"),
}

# Metrics (name -> SQL expression template; {dim} for count_distinct/sum/avg/min/max).
METRICS: dict[str, str] = {
    "count": "count(*)",
    "count_distinct": "count(DISTINCT {dim})",
    "sum": "sum({dim})",
    "avg": "avg({dim})",
    "min": "min({dim})",
    "max": "max({dim})",
    # Revenue at risk = revenue from customers who could still churn (paused, or
    # active with churn signals). Canceled customers are already lost — not at risk.
    "revenue_at_risk": "sum(monthly_revenue) FILTER (WHERE account_status <> 'canceled' AND (account_status = 'paused' OR churn_proxy))",
    "usage_change_pct": "avg(sessions_change_percent)",
    "satisfaction_avg": "avg(satisfaction_score)",
}

# Filters allowed beyond dimensions: direct record ids + date_range.
RECORD_FILTERS = ("customer_id", "ticket_id", "feedback_id", "date_range")

MetricName = Literal["count", "count_distinct", "sum", "avg", "min", "max",
                     "revenue_at_risk", "usage_change_pct", "satisfaction_avg"]


class SemanticQueryError(ValueError):
    """Raised when a SemanticQuery violates the catalog (clear, actionable)."""


def validate_semantic_query(q: dict[str, Any]) -> None:
    """Validate a SemanticQuery dict against the catalog. Raises on any violation."""
    if not isinstance(q, dict):
        raise SemanticQueryError("semantic_query must be a JSON object.")

    metric = q.get("metric")
    entity = q.get("of")

    if metric not in METRICS:
        raise SemanticQueryError(
            f"Unknown metric '{metric}'. Allowed: {', '.join(sorted(METRICS))}."
        )
    if entity not in ENTITIES:
        raise SemanticQueryError(
            f"Unknown entity '{entity}'. Allowed: {', '.join(sorted(ENTITIES))}."
        )

    # metric -> dimension compatibility (sum/avg/min/max/count_distinct need a dim)
    dim = q.get("of_dimension")
    if metric in ("sum", "avg", "min", "max", "count_distinct"):
        if not dim:
            raise SemanticQueryError(f"Metric '{metric}' requires 'of_dimension'.")
        if dim not in NUMERIC_COLUMNS.get(entity, ()):
            raise SemanticQueryError(
                f"Dimension '{dim}' is not numeric/valid for entity '{entity}'. "
                f"Allowed: {', '.join(NUMERIC_COLUMNS.get(entity, ())) or 'none'}."
            )

    # group-by dimensions
    for d in q.get("dimensions") or []:
        if d not in DIMENSIONS:
            raise SemanticQueryError(
                f"Unknown dimension '{d}'. Allowed: {', '.join(sorted(DIMENSIONS))}."
            )

    # filters
    filters = q.get("filters") or {}
    if not isinstance(filters, dict):
        raise SemanticQueryError("'filters' must be an object of dimension -> value.")
    for key in filters:
        if key not in DIMENSIONS and key not in RECORD_FILTERS:
            raise SemanticQueryError(
                f"Unknown filter key '{key}'. Allowed: {', '.join(sorted(set(DIMENSIONS) | set(RECORD_FILTERS)))}."
            )

    # time_range
    time_range = q.get("time_range") or {}
    if not isinstance(time_range, dict):
        raise SemanticQueryError("'time_range' must be an object with from/to.")
    for k in time_range:
        if k not in ("from", "to"):
            raise SemanticQueryError(f"Unknown time_range key '{k}'. Allowed: from, to.")

    # limit
    limit = q.get("limit", MAX_ROWS)
    if not isinstance(limit, int) or not (1 <= limit <= MAX_ROWS):
        raise SemanticQueryError(f"limit must be an int in 1..{MAX_ROWS}.")


def translate_semantic_query(q: dict[str, Any]) -> tuple[str, list[Any]]:
    """Translate a validated SemanticQuery into (parameterized SQL, params)."""
    metric = q["metric"]
    entity = q["of"]
    table = ENTITIES[entity]
    dim = q.get("of_dimension")
    group_dims = q.get("dimensions") or []
    filters = q.get("filters") or {}
    time_range = q.get("time_range") or {}
    limit = int(q.get("limit", MAX_ROWS))

    # metric SQL
    if metric in ("count_distinct", "sum", "avg", "min", "max"):
        metric_sql = METRICS[metric].format(dim=dim)
    else:
        metric_sql = METRICS[metric]

    # joins for dimensions/filters on customer attributes
    joins: list[str] = []
    needs_customer = any(
        d in ("customer_segment", "subscription_plan", "account_status", "country")
        for d in group_dims + list(filters.keys())
    )
    if needs_customer and entity != "customer" and table != "main.aggregate_segment_metrics":
        joins.append("LEFT JOIN main.dimension_customer dim_cust USING (customer_id)")

    # dimension expressions (group-by)
    select_dims = [DIMENSIONS[d] for d in group_dims]
    group_by = [DIMENSIONS[d] for d in group_dims]

    # WHERE clauses (parameterized)
    where: list[str] = []
    params: list[Any] = []
    for key, value in filters.items():
        if key == "date_range":
            continue
        if key == "customer_id":
            where.append("customer_id = ?")
            params.append(value)
        elif key == "ticket_id":
            where.append("ticket_id = ?")
            params.append(value)
        elif key == "feedback_id":
            where.append("feedback_id = ?")
            params.append(value)
        else:
            # dimension filter: qualify with table or alias
            col = DIMENSIONS[key]
            where.append(f"{col} = ?")
            params.append(value)

    # time_range on the entity's date column
    date_col = ENTITY_DATE_COLUMN.get(entity)
    if time_range and date_col:
        if time_range.get("from"):
            where.append(f"{date_col} >= ?")
            params.append(time_range["from"])
        if time_range.get("to"):
            where.append(f"{date_col} <= ?")
            params.append(time_range["to"])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    # aggregate SQL — the LIMIT is limit+1 so the caller can detect truncation
    # (execute_semantic_query fetches limit+1 and trims to limit).
    if group_dims:
        sql = (
            f"SELECT {', '.join(select_dims)}, {metric_sql} AS value "
            f"FROM {table} {' '.join(joins)} {where_sql} "
            f"GROUP BY {', '.join(group_by)} "
            f"ORDER BY value DESC LIMIT {limit + 1}"
        )
    else:
        sql = (
            f"SELECT {metric_sql} AS value "
            f"FROM {table} {' '.join(joins)} {where_sql} "
            f"LIMIT {limit + 1}"
        )
    return sql, params


def execute_semantic_query(q: dict[str, Any]) -> dict[str, Any]:
    """Validate + translate + execute a SemanticQuery read-only.

    Returns {data, columns, warnings, source_refs, total, truncated}. Invalid
    queries are reported in `warnings` (the MCP tool contract), not raised — so
    the caller (agent/LLM) always receives structured output. `truncated=True`
    when more rows exist than returned (the limit cap), so the FE can paginate
    or the agent can narrow the query.
    """
    try:
        validate_semantic_query(q)
    except SemanticQueryError as exc:
        return {"data": [], "columns": [], "warnings": [str(exc)], "source_refs": []}

    limit = int(q.get("limit", MAX_ROWS))
    sql, params = translate_semantic_query(q)

    # Read-only connection (defense-in-depth); scoped by the catalog itself.
    con = duckdb.connect(str(common_config.DB_PATH), read_only=True)
    try:
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(limit + 1)  # +1 to detect truncation
        truncated = len(rows) > limit
        rows = rows[:limit]
        data = [dict(zip(cols, r)) for r in rows]
    except duckdb.Error as exc:
        return {
            "data": [], "columns": [], "warnings": [f"Query execution failed: {exc}"],
            "source_refs": [],
        }
    finally:
        con.close()

    warnings: list[str] = []
    if truncated:
        warnings.append(f"Results truncated to {limit} rows — narrow the query or paginate.")
    if not data:
        warnings.append("Query returned no rows.")
    return {
        "data": data,
        "columns": cols,
        "warnings": warnings,
        "source_refs": [],
        "total": len(data),
        "truncated": truncated,
    }
