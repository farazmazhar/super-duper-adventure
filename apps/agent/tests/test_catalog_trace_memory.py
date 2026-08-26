"""Tests for the catalog + trace-result + pagination + memory improvements."""

from __future__ import annotations

import pytest

from apps.agent.classify import classify
from apps.agent.graph import AgentGraph
from apps.mcp import semantic

CANNED_CATALOG = {
    "data": {
        "entities": [
            {"id": "customer", "table": "main.dimension_customer", "columns": [{"name": "customer_id", "description": "id"}, {"name": "customer_name", "description": "name"}]},
            {"id": "ticket", "table": "main.fact_ticket", "columns": [{"name": "ticket_id", "description": "id"}]},
            {"id": "feedback", "table": "main.fact_feedback", "columns": [{"name": "feedback_id", "description": "id"}]},
            {"id": "usage", "table": "main.fact_usage", "columns": [{"name": "date", "description": "date"}]},
            {"id": "subscription_event", "table": "main.fact_subscription_event", "columns": []},
            {"id": "customer_features", "table": "main.aggregate_customer_features", "columns": []},
            {"id": "segment_metrics", "table": "main.aggregate_segment_metrics", "columns": []},
        ],
        "dimensions": ["country", "customer_segment"],
        "metrics": ["count", "sum"],
    },
    "source_refs": [],
    "warnings": [],
}


@pytest.fixture
def catalog_graph(fake_mcp) -> AgentGraph:
    """Graph whose fake MCP knows get_catalog (returns the canned catalog)."""
    fake_mcp.responses["get_catalog"] = CANNED_CATALOG
    return AgentGraph(mcp=fake_mcp, reason_agent=None)


# --- catalog (item 4: agent knows all entities/columns) ----------------------
def test_get_catalog_lists_all_entities() -> None:
    cat = semantic.get_catalog()["data"]
    ids = [e["id"] for e in cat["entities"]]
    for expected in ("customer", "ticket", "feedback", "usage", "subscription_event", "customer_features", "segment_metrics"):
        assert expected in ids
    assert "country" in cat["dimensions"]
    assert "count" in cat["metrics"]


def test_get_catalog_has_column_descriptions() -> None:
    cat = semantic.get_catalog()["data"]
    customer = next(e for e in cat["entities"] if e["id"] == "customer")
    cols = {c["name"]: c["description"] for c in customer["columns"]}
    assert "customer_name" in cols
    assert "country" in cols
    assert cols["customer_name"]  # non-empty description


def test_graph_surfaces_catalog_in_trace(catalog_graph, fake_mcp) -> None:
    """The gather node calls get_catalog and its result lands in the trace."""
    state = catalog_graph.run("How many customers are there by country?")
    trace = state["trace"]
    node_spans = [s for s in trace["spans"] if s.get("kind") == "node" and s.get("name", "").startswith("node:")]
    # the node span result should include get_catalog with entity count
    results = {}
    for s in node_spans:
        if s.get("result") and "get_catalog" in s["result"]:
            results = s["result"]["get_catalog"]
    assert results.get("entities") == 7  # canned catalog has 7 entities


# --- trace result JSON (item 2) ----------------------------------------------
def test_span_carries_result_json(graph, fake_mcp) -> None:
    state = graph.run("What are the top feedback themes?")
    trace = state["trace"]
    node_span = next(s for s in trace["spans"] if s.get("name") == "node:themes")
    assert node_span.get("result") is not None
    # retrieve_sources result should include row_count + sample (with scores)
    retrieval = node_span["result"].get("retrieve_sources")
    assert retrieval["row_count"] == 1
    assert "sample" in retrieval


def test_span_result_is_json_serializable(graph, fake_mcp) -> None:
    import json

    state = graph.run("Which customers are at risk?")
    for s in state["trace"]["spans"]:
        if s.get("result") is not None:
            json.dumps(s["result"], default=str)  # must not raise


# --- pagination (item 1) -----------------------------------------------------
def _tiny_db(tmp_path) -> None:
    """Build a minimal DB for semantic-query pagination tests."""
    import duckdb
    from apps.common import config as common_config

    db_path = tmp_path / "tiny.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE main.dimension_customer (customer_id VARCHAR, customer_segment VARCHAR, monthly_revenue DOUBLE)")
    con.executemany(
        "INSERT INTO main.dimension_customer VALUES (?,?,?)",
        [
            ("CUST-0001", "SMB", 100.0),
            ("CUST-0002", "SMB", 200.0),
            ("CUST-0003", "Enterprise", 300.0),
        ],
    )
    con.close()
    common_config.DB_PATH = db_path


def test_semantic_query_reports_truncation(tmp_path, monkeypatch) -> None:
    """A query capped by limit reports truncated=True + total."""
    _tiny_db(tmp_path)
    res = semantic.execute_semantic_query(
        {"metric": "count", "of": "customer", "dimensions": ["customer_segment"], "limit": 1}
    )
    # 2 SMB + 1 Enterprise grouped -> 2 groups; limit 1 truncates
    assert res["truncated"] is True
    assert res["total"] == 1
    assert any("truncated" in w for w in res["warnings"])


def test_semantic_query_no_truncation_when_under_limit(tmp_path) -> None:
    _tiny_db(tmp_path)
    res = semantic.execute_semantic_query(
        {"metric": "count", "of": "customer", "dimensions": ["customer_segment"]}
    )
    assert res["truncated"] is False
    assert res["total"] == len(res["data"]) == 2


# --- short-term memory (item 3) ----------------------------------------------
def test_graph_forwards_conversation(graph) -> None:
    conv = [{"role": "user", "content": "Who is CUST-0001?"}, {"role": "assistant", "content": "VertexPath A."}]
    state = graph.run("and what about their tickets?", conversation=conv)
    assert state["conversation"] == conv


def test_classify_name_hint_not_plain_noun() -> None:
    # "Paris" is a plain noun -> not misrouted as a customer
    assert classify("What's the weather like in Paris?")["intent"] == "irrelevant"
    # brand-like names route to customer
    assert classify("tell me about VertexPath A")["intent"] == "customer_query"
