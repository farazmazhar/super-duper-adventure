"""Layer-1 metric tests on synthetic traces + answers (no network)."""

from __future__ import annotations

from apps.eval.golden_set import GoldenQuestion, get_golden_set
from apps.eval.metrics import (
    evaluate_question,
    metric_citations,
    metric_entities,
    metric_render,
    metric_routing,
    metric_tools,
)


def _trace(intent="analytics_exec", routed_node="analytics", tools=("rank_customer_risk", "calculate_revenue_at_risk"), entities=None):
    return {
        "spans": [
            {"name": "classify", "metadata": {"intent": intent, "entities": entities or {"customer_ids": ["CUST-0001"]}}},
            {"name": "route", "metadata": {"intent": intent, "routed_node": routed_node}},
            {"name": "node:analytics", "metadata": {"tools": list(tools)}},
            {"name": "reason", "metadata": {"used_llm": True}},
        ],
        "events": [],
    }


def _answer(facts=None, conf="high", render="table", recs=False, others=None):
    return {
        "summary": "Top risk customer is CUST-0001 (VertexPath)",
        "facts": facts or [{"heading": "Facts", "content": "CUST-0001 is at risk (risk 70)", "citations": ["CUST-0001"]}],
        "interpretation": [],
        "recommendation": [{"heading": "Recommendations", "content": "Contact VertexPath", "citations": ["CUST-0001"]}] if recs else [],
        "other_sections": others or [],
        "confidence": conf,
        "render_hint": {"kind": render, "payload": {"kind": render}},
    }


def test_routing_metric() -> None:
    assert metric_routing(_trace(routed_node="analytics"), "analytics")["passed"] is True
    assert metric_routing(_trace(routed_node="themes"), "analytics")["passed"] is False


def test_entities_metric() -> None:
    t = _trace(entities={"customer_ids": ["CUST-0001"], "segment": "Enterprise"})
    r = metric_entities(t, {"customer_ids": ["CUST-0001"]})
    assert r["passed"] is True
    r2 = metric_entities(t, {"customer_ids": ["CUST-9999"]})
    assert r2["passed"] is False


def test_tools_metric() -> None:
    r = metric_tools(_trace(tools=("rank_customer_risk", "calculate_revenue_at_risk")), ["rank_customer_risk"])
    assert r["passed"] is True
    r2 = metric_tools(_trace(tools=("rank_customer_risk",)), ["get_feedback_themes"])
    assert r2["passed"] is False


def test_citations_metric() -> None:
    assert metric_citations(_answer())["passed"] is True
    no_cite = _answer(facts=[{"heading": "Facts", "content": "no ids here", "citations": []}])
    assert metric_citations(no_cite)["passed"] is False


def test_render_metric() -> None:
    assert metric_render(_answer(render="chart"), "chart")["passed"] is True
    assert metric_render(_answer(render="table"), "chart")["passed"] is False


def test_recommendations_one_directional() -> None:
    from apps.eval.metrics import metric_recommendations

    # expected but absent -> fail
    assert metric_recommendations(_answer(recs=False), True)["passed"] is False
    # expected and present -> pass
    assert metric_recommendations(_answer(recs=True), True)["passed"] is True
    # not expected but present -> pass (on-demand)
    assert metric_recommendations(_answer(recs=True), False)["passed"] is True


def test_evaluate_question_all_pass() -> None:
    g = GoldenQuestion(
        id="t1", question="q", expected_intent="analytics_exec", expected_route="analytics",
        expected_entities={"customer_ids": ["CUST-0001"]},
        expected_tools=["rank_customer_risk"], expected_render="table",
    )
    r = evaluate_question(g, _trace(), _answer())
    assert r["passed"] is True
    assert r["passed_metrics"] == r["total_metrics"]


def test_evaluate_question_wrong_route_fails() -> None:
    g = GoldenQuestion(
        id="t2", question="q", expected_intent="theme_sentiment", expected_route="themes",
        expected_render="chart",
    )
    r = evaluate_question(g, _trace(routed_node="analytics"), _answer(render="table"))
    assert r["passed"] is False
    assert r["metrics"]["routing"]["passed"] is False


def test_irrelevant_metric() -> None:
    from apps.eval.metrics import metric_irrelevant

    ans = _answer(conf="low", render="markdown")
    assert metric_irrelevant(ans, _trace(), True)["passed"] is True
    ans2 = _answer(conf="high", render="table")
    assert metric_irrelevant(ans2, _trace(), True)["passed"] is False


def test_golden_set_is_well_formed() -> None:
    gs = get_golden_set()
    ids = [g.id for g in gs]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for g in gs:
        assert g.question
        assert g.expected_intent
        assert g.expected_route
