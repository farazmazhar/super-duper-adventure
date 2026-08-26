"""App 4 — shared deterministic tool implementations (single source of truth).

Pure DuckDB logic: no imports from the agent or any LLM. Imported directly by
the PydanticAI agent (App 3) and wrapped as MCP tools by apps/mcp/server.py.

Every tool returns a dict with keys:
    data        — the result set (list of dicts, or a scalar summary)
    source_refs — record ids the result is based on (for citations)
    warnings    — human-readable notes (e.g. NULL revenue treated as unknown)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import duckdb

# Make the repo root importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.common import config as common_config  # noqa: E402

# Schemas we allow fixed tools to read.
ALLOWED_READ_SCHEMAS = ("main", "vector", "agent")


def _connect() -> duckdb.DuckDBPyConnection:
    # Read DB_PATH dynamically so tests can point at a fixture DB.
    return duckdb.connect(str(common_config.DB_PATH), read_only=True)

# Composite risk weights (documented; tuned to the current data).
#   churn status  -> strong signal
#   usage decline -> strong signal
#   open/urgent tickets, low satisfaction -> supporting signals
RISK_WEIGHTS = {
    "churn_proxy": 30.0,
    "sessions_decline": 25.0,
    "tickets_open": 15.0,
    "tickets_urgent": 15.0,
    "low_satisfaction": 10.0,
    "no_recent_usage": 5.0,
}

# Customers with no usage rows in the last 4 weeks get flagged.
USAGE_WINDOW_WEEKS = 4


def _rows_to_dicts(
    con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None
) -> list[dict[str, Any]]:
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Customer name resolution (query flexibility: "tell me about VertexPath A")
# ---------------------------------------------------------------------------
def resolve_customer_name(name: str) -> dict[str, Any]:
    """Resolve a customer name (or name fragment) to customer_id(s).

    Case-insensitive substring match over customer_name. Returns the standard
    {data, source_refs, warnings} contract; data is a list of matches.
    """
    name = (name or "").strip()
    if not name:
        return {"data": [], "source_refs": [], "warnings": ["Empty customer name."]}
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            """
            SELECT customer_id, customer_name, customer_segment, subscription_plan, account_status
            FROM main.dimension_customer
            WHERE customer_name ILIKE ?
            ORDER BY customer_id
            LIMIT 10
            """,
            [f"%{name}%"],
        )
    return {
        "data": rows,
        "source_refs": [r["customer_id"] for r in rows],
        "warnings": [] if rows else [f"No customer found matching '{name}'."],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _customer_exists(con: duckdb.DuckDBPyConnection, customer_id: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM main.dimension_customer WHERE customer_id = ?",
        [customer_id],
    ).fetchone()[0] > 0


def _normalize_status(status: str | None) -> str | None:
    """Map the caller's status filter to the canonical lowercase set."""
    if status is None:
        return None
    s = status.strip().lower()
    if s in ("active", "canceled", "paused", "churned"):
        return "canceled" if s == "churned" else s
    return None


# ---------------------------------------------------------------------------
# Customer profile
# ---------------------------------------------------------------------------
def get_customer_profile(customer_id: str) -> dict[str, Any]:
    """Attributes + derived features + risk inputs for one customer."""
    with _connect() as con:
        if not _customer_exists(con, customer_id):
            return {"data": None, "source_refs": [], "warnings": [f"Customer {customer_id} not found."]}
        rows = _rows_to_dicts(
            con,
            """
            SELECT d.customer_id, d.customer_name, d.customer_segment, d.country,
                   d.subscription_plan, d.monthly_revenue, d.account_created_at,
                   d.account_status, d.revenue_imputed,
                   f.churn_proxy, f.ticket_count, f.tickets_open, f.tickets_urgent,
                   f.average_resolution_time_hours, f.average_satisfaction_score,
                   f.sessions_last_4_weeks, f.sessions_previous_4_weeks,
                   f.sessions_change_percent, f.errors_total, f.feedback_count,
                   f.average_rating, f.cancellations, f.downgrades, f.upgrades,
                   f.renewals, f.revenue_change_sum, f.last_event_date, f.last_event_type
            FROM main.dimension_customer d
            LEFT JOIN main.aggregate_customer_features f ON f.customer_id = d.customer_id
            WHERE d.customer_id = ?
            """,
            [customer_id],
        )
    warnings = []
    if rows[0]["monthly_revenue"] is None:
        warnings.append("monthly_revenue is NULL (unknown); revenue-at-risk excludes it.")
    return {"data": rows[0], "source_refs": [customer_id], "warnings": warnings}


# ---------------------------------------------------------------------------
# Risk ranking
# ---------------------------------------------------------------------------
def _risk_score(row: dict[str, Any]) -> float:
    """Composite risk score 0-100 from feature inputs. Higher = more at risk."""
    score = 0.0
    if row.get("churn_proxy"):
        score += RISK_WEIGHTS["churn_proxy"]
    chg = row.get("sessions_change_percent")
    if chg is not None and chg < -20.0:
        score += RISK_WEIGHTS["sessions_decline"]
    score += RISK_WEIGHTS["tickets_open"] * min(int(row.get("tickets_open") or 0), 2) / 2.0
    score += RISK_WEIGHTS["tickets_urgent"] * min(int(row.get("tickets_urgent") or 0), 2) / 2.0
    sat = row.get("average_satisfaction_score")
    if sat is not None and sat < 3.0:
        score += RISK_WEIGHTS["low_satisfaction"]
    if row.get("sessions_last_4_weeks") in (None, 0):
        score += RISK_WEIGHTS["no_recent_usage"]
    return round(min(score, 100.0), 1)


def rank_customer_risk(limit: int = 10, segment: str | None = None, status: str | None = None) -> dict[str, Any]:
    """Rank customers by composite risk score; include top drivers per customer.

    Canceled customers are excluded by default — they are already lost, not at
    risk. Pass status="canceled" (or "churned") explicitly to see them.
    """
    where: list[str] = []
    params: list[Any] = []
    if segment:
        where.append("d.customer_segment = ?")
        params.append(segment)
    status = _normalize_status(status)
    if status:
        where.append("d.account_status = ?")
        params.append(status)
    else:
        # At-risk ranking is forward-looking: exclude already-lost customers.
        where.append("d.account_status <> 'canceled'")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            f"""
            SELECT f.customer_id, d.customer_name, d.customer_segment, d.subscription_plan,
                   d.account_status, d.monthly_revenue, f.churn_proxy,
                   f.sessions_last_4_weeks, f.sessions_previous_4_weeks, f.sessions_change_percent,
                   f.tickets_open, f.tickets_urgent, f.average_satisfaction_score
            FROM main.aggregate_customer_features f
            JOIN main.dimension_customer d ON d.customer_id = f.customer_id
            {where_sql}
            ORDER BY f.customer_id
            """,
            params,
        )
    for r in rows:
        r["risk_score"] = _risk_score(r)
        r["risk_drivers"] = _risk_drivers(r)
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    return {
        "data": rows[:limit],
        "source_refs": [r["customer_id"] for r in rows[:limit]],
        "warnings": ["Composite heuristic risk score (0-100); not a model. Canceled customers excluded (already lost)."],
    }


def _risk_drivers(row: dict[str, Any]) -> list[str]:
    drivers: list[str] = []
    if row.get("churn_proxy"):
        drivers.append(f"status={row['account_status']}")
    chg = row.get("sessions_change_percent")
    if chg is not None and chg < -20.0:
        drivers.append(f"sessions {chg:.0f}% vs prior 4wk")
    if row.get("tickets_open"):
        drivers.append(f"{row['tickets_open']} open tickets")
    if row.get("tickets_urgent"):
        drivers.append(f"{row['tickets_urgent']} urgent tickets")
    if (row.get("average_satisfaction_score") or 0) < 3.0:
        drivers.append("low avg satisfaction (<3)")
    if row.get("sessions_last_4_weeks") in (None, 0):
        drivers.append("no usage in last 4 weeks")
    return drivers


# ---------------------------------------------------------------------------
# Revenue at risk
# ---------------------------------------------------------------------------
def calculate_revenue_at_risk(segment: str | None = None) -> dict[str, Any]:
    """Sum monthly_revenue for customers who could still churn.

    At risk = paused customers, or active customers with churn signals
    (high risk score). Canceled customers are already lost — they are NOT
    counted as revenue at risk (they are no longer customers).

    NULL revenue is treated as unknown: excluded from the sum and flagged.
    """
    # risk_score is computed in Python, so we fetch all candidates and filter here.
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            """
            SELECT d.customer_id, d.customer_name, d.customer_segment, d.monthly_revenue,
                   d.account_status, f.churn_proxy, f.sessions_change_percent,
                   f.tickets_open, f.tickets_urgent, f.average_satisfaction_score,
                   f.sessions_last_4_weeks
            FROM main.dimension_customer d
            LEFT JOIN main.aggregate_customer_features f ON f.customer_id = d.customer_id
            """,
        )
    if segment:
        rows = [r for r in rows if r["customer_segment"] == segment]

    flagged: list[dict[str, Any]] = []
    at_risk: list[dict[str, Any]] = []
    total = 0.0
    for r in rows:
        # Canceled customers are already lost — excluded from revenue at risk.
        if r["account_status"] == "canceled":
            continue
        r["risk_score"] = _risk_score(r)
        at_risk_flag = (
            r["account_status"] == "paused"
            or r["churn_proxy"]
            or r["risk_score"] >= 50.0
        )
        if not at_risk_flag:
            continue
        if r["monthly_revenue"] is None:
            flagged.append(r)
        else:
            total += r["monthly_revenue"]
            at_risk.append(r)

    warnings = []
    if flagged:
        warnings.append(
            f"{len(flagged)} at-risk customer(s) have NULL monthly_revenue — excluded from the sum (not imputed)."
        )
    return {
        "data": {
            "revenue_at_risk": round(total, 2),
            "at_risk_customers": len(at_risk),
            "at_risk_with_unknown_revenue": len(flagged),
            "segment_filter": segment,
            "customers": [
                {
                    "customer_id": c["customer_id"],
                    "customer_name": c["customer_name"],
                    "account_status": c["account_status"],
                    "monthly_revenue": c["monthly_revenue"],
                    "risk_score": c["risk_score"],
                }
                for c in at_risk
            ],
        },
        "source_refs": [c["customer_id"] for c in at_risk + flagged],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Tickets / feedback / themes
# ---------------------------------------------------------------------------
def get_customer_tickets(
    customer_id: str, limit: int = 20, category: str | None = None,
    priority: str | None = None, status: str | None = None,
) -> dict[str, Any]:
    where = ["customer_id = ?"]
    params: list[Any] = [customer_id]
    if category:
        where.append("category = ?")
        params.append(category.lower())
    if priority:
        where.append("priority = ?")
        params.append(priority.lower())
    if status:
        where.append("status = ?")
        params.append(status.lower())
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            f"""
            SELECT ticket_id, created_at, subject, category, priority, status,
                   resolution_time_hours, satisfaction_score
            FROM main.fact_ticket
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params + [limit],
        )
    return {
        "data": rows,
        "source_refs": [r["ticket_id"] for r in rows],
        "warnings": [],
    }


def get_customer_feedback(customer_id: str, limit: int = 20) -> dict[str, Any]:
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            """
            SELECT f.feedback_id, f.created_at, f.feedback_text, f.feedback_source,
                   f.rating, t.theme, t.sentiment
            FROM main.fact_feedback f
            LEFT JOIN main.aggregate_theme t ON t.feedback_id = f.feedback_id
            WHERE f.customer_id = ?
            ORDER BY f.created_at DESC
            LIMIT ?
            """,
            [customer_id, limit],
        )
    return {
        "data": rows,
        "source_refs": [r["feedback_id"] for r in rows],
        "warnings": [],
    }


def get_feedback_themes(min_count: int = 1, segment: str | None = None) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if segment:
        where.append("d.customer_segment = ?")
        params.append(segment)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            f"""
            SELECT t.theme, count(*) AS feedback_count,
                   count(*) FILTER (WHERE t.sentiment = 'positive') AS positive,
                   count(*) FILTER (WHERE t.sentiment = 'negative') AS negative,
                   count(*) FILTER (WHERE t.sentiment = 'neutral') AS neutral
            FROM main.aggregate_theme t
            JOIN main.dimension_customer d ON d.customer_id = t.customer_id
            {where_sql}
            GROUP BY t.theme
            HAVING count(*) >= ?
            ORDER BY feedback_count DESC
            """,
            params + [min_count],
        )
    return {
        "data": rows,
        "source_refs": [f"theme:{r['theme']}" for r in rows],
        "warnings": ["Themes are rule-based (source='rule'); LLM overlay may upgrade them."],
    }


def get_ticket_breakdown(segment: str | None = None) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if segment:
        where.append("d.customer_segment = ?")
        params.append(segment)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            f"""
            SELECT t.category, t.priority, t.status,
                   count(*) AS ticket_count,
                   avg(t.resolution_time_hours) AS average_resolution_time_hours,
                   avg(t.satisfaction_score) AS average_satisfaction_score
            FROM main.fact_ticket t
            JOIN main.dimension_customer d ON d.customer_id = t.customer_id
            {where_sql}
            GROUP BY t.category, t.priority, t.status
            ORDER BY ticket_count DESC
            """,
            params,
        )
    return {
        "data": rows,
        "source_refs": [],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
def get_usage_change(customer_id: str | None = None, segment: str | None = None) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if customer_id:
        where.append("f.customer_id = ?")
        params.append(customer_id)
    if segment:
        where.append("f.customer_segment = ?")
        params.append(segment)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            f"""
            SELECT f.customer_id, d.customer_name, d.customer_segment,
                   f.sessions_last_4_weeks, f.sessions_previous_4_weeks,
                   f.sessions_change_percent,
                   f.active_users_last_4_weeks, f.active_users_previous_4_weeks,
                   f.active_users_change_percent, f.errors_total
            FROM main.aggregate_customer_features f
            JOIN main.dimension_customer d ON d.customer_id = f.customer_id
            {where_sql}
            ORDER BY f.customer_id
            """,
            params,
        )
    if customer_id is None and not segment:
        # aggregate across all customers
        n = len(rows)
        agg = {
            "customers": n,
            "sessions_last_4_weeks": sum(r["sessions_last_4_weeks"] or 0 for r in rows),
            "sessions_previous_4_weeks": sum(r["sessions_previous_4_weeks"] or 0 for r in rows),
            "errors_total": sum(r["errors_total"] or 0 for r in rows),
            "customers_with_sessions_decline": sum(
                1 for r in rows if (r["sessions_change_percent"] or 0) < -20.0
            ),
        }
        prev = agg["sessions_previous_4_weeks"]
        agg["sessions_change_percent"] = (
            round((agg["sessions_last_4_weeks"] - prev) * 100.0 / prev, 1) if prev else None
        )
        return {"data": agg, "source_refs": [r["customer_id"] for r in rows], "warnings": []}
    return {
        "data": rows,
        "source_refs": [r["customer_id"] for r in rows],
        "warnings": [],
    }


def get_usage_trend(customer_id: str, weeks: int = 8) -> dict[str, Any]:
    """Daily sessions/errors series for charts (last N weeks relative to data max date)."""
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            """
            SELECT date, active_users, sessions, errors, feature_usage, average_session_duration
            FROM main.fact_usage
            WHERE customer_id = ?
              AND date >= (SELECT max(date) FROM main.fact_usage) - INTERVAL (? - 1) WEEK
            ORDER BY date
            """,
            [customer_id, weeks],
        )
    return {
        "data": rows,
        "source_refs": [f"{customer_id}:{r['date']}" for r in rows],
        "warnings": [],
    }


def get_subscription_events(customer_id: str, limit: int = 10) -> dict[str, Any]:
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            """
            SELECT event_date, event_type, previous_plan, new_plan, revenue_change
            FROM main.fact_subscription_event
            WHERE customer_id = ?
            ORDER BY event_date DESC
            LIMIT ?
            """,
            [customer_id, limit],
        )
    return {
        "data": rows,
        "source_refs": [f"{customer_id}:{r['event_date']}:{r['event_type']}" for r in rows],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Segments / discovery
# ---------------------------------------------------------------------------
def calculate_segment_metrics(segment: str | None = None) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if segment:
        where.append("(segment = ? OR plan = ?)")
        params += [segment, segment]
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            f"""
            SELECT segment, plan, customers, revenue, revenue_customers,
                   customers_with_tickets, ticket_count, complaint_rate,
                   average_resolution_time_hours, average_satisfaction_score,
                   cancel_rate, global_customers, global_revenue,
                   global_average_resolution, global_average_satisfaction,
                   global_churn, global_cancel_rate
            FROM main.aggregate_segment_metrics
            {where_sql}
            ORDER BY segment NULLS LAST, plan NULLS LAST
            """,
            params,
        )
    return {"data": rows, "source_refs": [], "warnings": []}


def list_customers(
    segment: str | None = None, status: str | None = None,
    limit: int = 50, search: str | None = None,
) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if segment:
        where.append("d.customer_segment = ?")
        params.append(segment)
    status = _normalize_status(status)
    if status:
        where.append("d.account_status = ?")
        params.append(status)
    if search:
        where.append("(d.customer_id ILIKE ? OR d.customer_name ILIKE ?)")
        params += [f"%{search}%", f"%{search}%"]
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with _connect() as con:
        rows = _rows_to_dicts(
            con,
            f"""
            SELECT d.customer_id, d.customer_name, d.customer_segment, d.country,
                   d.subscription_plan, d.monthly_revenue, d.account_status,
                   f.churn_proxy, f.sessions_last_4_weeks, f.sessions_change_percent,
                   f.tickets_open, f.tickets_urgent, f.average_satisfaction_score
            FROM main.dimension_customer d
            LEFT JOIN main.aggregate_customer_features f ON f.customer_id = d.customer_id
            {where_sql}
            ORDER BY d.customer_id
            LIMIT ?
            """,
            params + [limit],
        )
    for r in rows:
        # risk_score is computed on the fly from the features we fetched.
        r["risk_score"] = _risk_score(r)
    return {"data": rows, "source_refs": [r["customer_id"] for r in rows], "warnings": []}


# ---------------------------------------------------------------------------
# Memory (long-term, written to agent schema)
# ---------------------------------------------------------------------------
def _ensure_memory_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS agent")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS agent.agent_memory (
            key VARCHAR PRIMARY KEY,
            value VARCHAR,
            updated_at TIMESTAMP DEFAULT now()
        )
        """
    )


def read_memory(key: str) -> dict[str, Any]:
    with _connect() as con:
        rows = _rows_to_dicts(
            con, "SELECT key, value, updated_at FROM agent.agent_memory WHERE key = ?", [key]
        )
    return {"data": rows[0] if rows else None, "source_refs": [], "warnings": []}


def write_memory(key: str, value: str) -> dict[str, Any]:
    # memory writes need a read-write connection
    with duckdb.connect(str(common_config.DB_PATH)) as con:
        _ensure_memory_table(con)
        con.execute(
            """
            INSERT INTO agent.agent_memory (key, value, updated_at) VALUES (?, ?, now())
            ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = now()
            """,
            [key, value],
        )
    return {"data": {"key": key, "value": value}, "source_refs": [], "warnings": []}


def list_memory() -> dict[str, Any]:
    with _connect() as con:
        rows = _rows_to_dicts(
            con, "SELECT key, value, updated_at FROM agent.agent_memory ORDER BY updated_at DESC"
        )
    return {"data": rows, "source_refs": [], "warnings": []}


# ---------------------------------------------------------------------------
# Retrieval lives in retrieval.py (runtime RAG, owns the EmbeddingClient).
# See apps/mcp/retrieval.py — no cross-app imports from the embedding app.
# ---------------------------------------------------------------------------
