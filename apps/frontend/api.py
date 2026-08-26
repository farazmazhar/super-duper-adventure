"""App 7 — JSON/SSE API for the Starlette FE.

Every endpoint is thin: read-only DuckDB via `apps.frontend.db`, or the one
sanctioned FE→agent import (`apps.agent.runner.run_question`). The FE never
opens a write connection; memory/traces are written by the agent/MCP only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.frontend import db  # noqa: E402

# ---------------------------------------------------------------------------
# Agent bundle (built once per process; reused across requests)
# ---------------------------------------------------------------------------
_AGENT_BUNDLE: dict[str, Any] | None = None


def get_agent_bundle() -> dict[str, Any] | None:
    """Lazily build (graph, mcp) once per process. None if the agent is unavailable."""
    global _AGENT_BUNDLE
    if _AGENT_BUNDLE is not None:
        return _AGENT_BUNDLE
    try:
        from apps.agent.agent import build_reason_agent
        from apps.agent.graph import AgentGraph
        from apps.agent.mcp_client import McpClient
        from apps.common.config import settings

        mcp = McpClient()
        mcp.start()
        reason_agent = None if settings.openai_api_key is None else build_reason_agent(mcp=mcp)
        graph = AgentGraph(mcp, reason_agent=reason_agent)
        _AGENT_BUNDLE = {"graph": graph, "mcp": mcp}
        return _AGENT_BUNDLE
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"agent unavailable: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Chat (SSE) helpers
# ---------------------------------------------------------------------------
def run_chat(question: str, rerank_enabled: bool | None, moderation_enabled: bool | None,
             conversation: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Run one question through the guarded agent path; returns the FE payload."""
    bundle = get_agent_bundle()
    if bundle is None:
        return {"error": "Agent is not available. Check that the MCP server can start.", "answer": None, "trace": None, "guardrails": []}

    from apps.agent.runner import run_question
    from apps.agent.tracing import TraceContext

    retrieve_kwargs = {"rerank_enabled": rerank_enabled} if rerank_enabled is not None else {}
    trace = TraceContext(question=question)
    state = run_question(
        bundle["graph"],
        question,
        trace=trace,
        retrieve_kwargs=retrieve_kwargs,
        moderation_enabled=moderation_enabled,
        conversation=conversation,
    )
    answer = state.get("answer")
    answer_dict = answer.model_dump() if hasattr(answer, "model_dump") else answer
    return {
        "answer": answer_dict,
        "trace": state.get("trace"),
        "guardrails": state.get("guardrails", []),
        "routed_node": state.get("routed_node"),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Dashboard / drill-down / admin (read-only DuckDB)
# ---------------------------------------------------------------------------
def dashboard_payload() -> dict[str, Any]:
    """Exec dashboard: KPI cards + chart data (themes, risk, tickets, usage)."""
    out: dict[str, Any] = {"kpis": [], "themes": [], "risk": [], "tickets_by_category": [], "revenue_series": []}
    try:
        out["kpis"] = db.query(
            """
            SELECT
                (SELECT round(sum(monthly_revenue)) FROM main.dimension_customer WHERE account_status = 'active') AS mrr,
                (SELECT count(*) FROM main.dimension_customer) AS customers,
                (SELECT count(*) FROM main.dimension_customer WHERE account_status IN ('canceled','paused')) AS churned,
                (SELECT count(*) FROM main.fact_ticket WHERE status = 'open') AS open_tickets
            """
        )
    except Exception:
        pass
    try:
        out["themes"] = db.query(
            "SELECT theme, count(*) AS count FROM main.aggregate_theme GROUP BY theme ORDER BY count DESC LIMIT 10"
        )
    except Exception:
        pass
    try:
        # At-risk = paused customers (canceled are already lost, not at risk).
        out["risk"] = db.query(
            "SELECT customer_id, account_status, monthly_revenue FROM main.dimension_customer "
            "WHERE account_status = 'paused' ORDER BY monthly_revenue DESC NULLS LAST LIMIT 10"
        )
    except Exception:
        pass
    try:
        out["tickets_by_category"] = db.query(
            "SELECT category, count(*) AS count FROM main.fact_ticket GROUP BY category ORDER BY count DESC"
        )
    except Exception:
        pass
    return out


def customer_payload(customer_id: str) -> dict[str, Any]:
    """Customer drill-down: profile + tickets + feedback + usage trend."""
    out: dict[str, Any] = {"profile": None, "tickets": [], "feedback": [], "usage_trend": []}
    try:
        rows = db.query(
            "SELECT * FROM main.dimension_customer WHERE customer_id = ?", [customer_id]
        )
        out["profile"] = rows[0] if rows else None
    except Exception:
        pass
    try:
        out["tickets"] = db.query(
            "SELECT ticket_id, created_at, category, priority, status, subject "
            "FROM main.fact_ticket WHERE customer_id = ? ORDER BY created_at DESC LIMIT 20",
            [customer_id],
        )
    except Exception:
        pass
    try:
        out["feedback"] = db.query(
            "SELECT feedback_id, created_at, feedback_text, rating FROM main.fact_feedback "
            "WHERE customer_id = ? ORDER BY created_at DESC LIMIT 20",
            [customer_id],
        )
    except Exception:
        pass
    try:
        out["usage_trend"] = db.query(
            "SELECT date, sessions, active_users FROM main.fact_usage WHERE customer_id = ? ORDER BY date LIMIT 60",
            [customer_id],
        )
    except Exception:
        pass
    return out


def admin_payload() -> dict[str, Any]:
    """System status: app health + quality report + trace/memory counts."""
    out: dict[str, Any] = {"quality_report": [], "embedding_meta": [], "memory_count": 0, "trace_count": 0, "tables": []}
    try:
        out["quality_report"] = db.query(
            "SELECT rule, table_name, description, count FROM main.quality_report ORDER BY table_name, rule"
        )
    except Exception:
        pass
    try:
        out["embedding_meta"] = db.query(
            "SELECT model, dimension, embedded_at FROM vector.embedding_meta ORDER BY embedded_at DESC LIMIT 5"
        )
    except Exception:
        pass
    try:
        out["memory_count"] = db.query("SELECT count(*) AS n FROM agent.agent_memory")[0]["n"]
    except Exception:
        pass
    try:
        out["trace_count"] = db.query("SELECT count(*) AS n FROM agent.traces")[0]["n"]
    except Exception:
        pass
    try:
        out["tables"] = db.query(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema IN ('main','vector','agent') ORDER BY table_schema, table_name"
        )
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Serialization helpers (DuckDB returns date/datetime objects)
# ---------------------------------------------------------------------------
def json_default(o: Any) -> Any:
    import datetime

    if isinstance(o, (datetime.date, datetime.datetime, datetime.time)):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o)}")
