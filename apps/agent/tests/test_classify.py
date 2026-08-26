"""Classification + routing tests."""

from __future__ import annotations

from apps.agent.classify import classify, classify_intent, extract_entities
from apps.agent.constants.routing import ROUTE_MAP


def test_extract_customer_id() -> None:
    e = extract_entities("tell me about CUST-0001")
    assert e["customer_ids"] == ["CUST-0001"]


def test_extract_ticket_and_feedback_ids() -> None:
    e = extract_entities("look at TCK-00042 and FDB-00007")
    assert e["ticket_ids"] == ["TCK-00042"]
    assert e["feedback_ids"] == ["FDB-00007"]


def test_extract_segment_and_plan() -> None:
    e = extract_entities("enterprise segment, business plan")
    assert e["segment"] == "enterprise"
    assert e["plan"] == "business"


def test_classify_customer_query() -> None:
    r = classify("Who is CUST-0001 and what's their profile?")
    assert r["intent"] == "customer_query"


def test_classify_ticket_query() -> None:
    r = classify("Show me ticket TCK-00042")
    assert r["intent"] == "ticket_query"


def test_classify_theme_sentiment() -> None:
    r = classify("What are the top complaints and themes?")
    assert r["intent"] == "theme_sentiment"


def test_classify_analytics_exec() -> None:
    r = classify("Which segment has the highest churn? compare revenue at risk")
    assert r["intent"] == "analytics_exec"


def test_classify_trend() -> None:
    r = classify("How have sessions changed over the last 4 weeks?")
    assert r["intent"] == "trend"


def test_classify_rag() -> None:
    r = classify("Find feedback that mentions billing problems")
    assert r["intent"] == "free_text_rag"


def test_classify_general_fallback() -> None:
    r = classify("Hello there")
    assert r["intent"] == "general"


def test_classify_irrelevant() -> None:
    r = classify("What's the weather like in Paris?")
    assert r["intent"] == "irrelevant"


def test_route_map_covers_all_intents() -> None:
    from apps.agent.constants.classify import INTENTS
    for intent in INTENTS:
        assert intent in ROUTE_MAP
    assert ROUTE_MAP["irrelevant"] == "blocked"
    assert ROUTE_MAP["customer_query"] == "customer"
