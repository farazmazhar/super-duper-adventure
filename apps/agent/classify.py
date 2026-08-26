"""Rule-based classification + entity extraction (deterministic; no LLM).

Implements the `classify` node of the graph (spec §1): intent labels and entity
extraction via regex + keyword scoring, using the lexicons in constants/classify.py.
"""

from __future__ import annotations

import re
from typing import Any

from apps.agent.constants.classify import (
    ANALYTICS_KEYWORDS,
    CUSTOMER_ID_RE,
    CUSTOMER_KEYWORDS,
    FEEDBACK_ID_RE,
    FEEDBACK_KEYWORDS,
    IRRELEVANT_KEYWORDS,
    PLANS,
    RAG_KEYWORDS,
    SEGMENTS,
    THEME_KEYWORDS,
    TICKET_ID_RE,
    TICKET_KEYWORDS,
    TREND_KEYWORDS,
)

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------
def extract_entities(text: str) -> dict[str, Any]:
    """Extract customer_id / ticket_id / feedback_id / segment / plan / date hints."""
    lower = text.lower()
    entities: dict[str, Any] = {
        "customer_ids": list(dict.fromkeys(m.group(0).upper() for m in CUSTOMER_ID_RE.finditer(text))),
        "ticket_ids": list(dict.fromkeys(m.group(0).upper() for m in TICKET_ID_RE.finditer(text))),
        "feedback_ids": list(dict.fromkeys(m.group(0).upper() for m in FEEDBACK_ID_RE.finditer(text))),
        "segment": next((s for s in SEGMENTS if s in lower), None),
        "plan": next((p for p in PLANS if p in lower), None),
        "date_hints": _extract_dates(lower),
        # Name-token hint (resolved to customer_id by the gather node via MCP).
        "customer_name_hint": _customer_name_hint(text),
    }
    return entities


# Words that are clearly not part of a customer name (stopwords for the
# name-hint detector). A capitalized phrase that isn't these and isn't a known
# keyword is treated as a possible customer name.
_NAME_STOP = {
    "a", "an", "the", "and", "or", "of", "for", "with", "about", "what", "who",
    "which", "how", "many", "much", "show", "tell", "give", "me", "is", "are",
    "was", "were", "customer", "customers", "ticket", "tickets", "feedback",
    "usage", "trend", "risk", "revenue", "segment", "plan", "country", "churn",
    "open", "count", "total", "by", "in", "at", "on", "from", "last", "week",
    "month", "their", "they", "has", "have", "had", "been", "being", "do",
    "does", "did", "can", "could", "would", "should", "please", "all", "any",
    "most", "top", "high", "low", "status", "active", "paused", "canceled",
    "enterprise", "smb", "mid-market", "business", "free", "team",
    # sentence-openers / question words / common verbs that start sentences
    "find", "look", "search", "show", "tell", "give", "list", "get", "what",
    "which", "who", "how", "why", "where", "when", "hello", "hi", "hey",
    "there", "please", "can", "could", "would", "should", "do", "does", "did",
    "is", "are", "was", "were", "have", "has", "had", "will", "would", "may",
    "might", "must", "let", "make", "write", "create", "explain", "describe",
    "summarize", "compare", "analyze", "review", "check", "see", "help",
    "the", "a", "an", "and", "but", "or", "so", "for", "with", "from", "this",
    "that", "these", "those", "it", "its", "you", "your", "our", "we", "us",
    "they", "them", "their", "i", "my", "me",
    # id-prefix tokens that look capitalized but aren't names
    "cust", "tck", "fdb",
}


def _customer_name_hint(text: str) -> str | None:
    """Detect a likely customer name: a capitalized token sequence.

    Conservative: only returns a hint when the capitalized token is not a
    common English word (sentence-openers like "Find", "Hello" are excluded)
    so ordinary questions never trigger it. The gather node resolves the hint
    against the DB (cheap) and ignores it if nothing matches.
    """
    import re as _re

    matches = _re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]*)?\b", text)
    for m in matches:
        tokens = m.split()
        meaningful = [t for t in tokens if len(t) > 1]
        if not meaningful:
            continue
        # every multi-char token must be a *non-stopword* (a name-like token)
        if all(t.lower() not in _NAME_STOP for t in meaningful) and len(m) >= 3:
            return m
    return None


# ---------------------------------------------------------------------------
# Query planning for general/analytical questions (semantic_query support)
# ---------------------------------------------------------------------------
# Entity word -> semantic entity id (matches apps/mcp/semantic.py ENTITIES).
_ENTITY_WORDS = {
    "customer": "customer",
    "customers": "customer",
    "account": "customer",
    "accounts": "customer",
    "ticket": "ticket",
    "tickets": "ticket",
    "feedback": "feedback",
    "reviews": "feedback",
    "usage": "usage",
    "session": "usage",
    "sessions": "usage",
    "subscription": "subscription_event",
    "subscriptions": "subscription_event",
    "plan change": "subscription_event",
}

# Dimension word -> semantic dimension id (matches semantic.py DIMENSIONS).
_DIMENSION_WORDS = {
    "country": "country",
    "countries": "country",
    "segment": "customer_segment",
    "segments": "customer_segment",
    "plan": "subscription_plan",
    "plans": "subscription_plan",
    "status": "account_status",
    "category": "category",
    "priority": "priority",
    "source": "feedback_source",
    "feature": "feature_usage",
}

_METRIC_WORDS = {
    "count": "count",
    "how many": "count",
    "total": "count",
    "number of": "count",
    "average": "avg",
    "avg": "avg",
    "mean": "avg",
    "sum": "sum",
    "revenue": "sum",
}


def build_query_plan(text: str, entities: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a general/analytical question into a SemanticQuery plan.

    Returns a dict matching the semantic_query tool contract, or None if the
    question has no clear metric/entity (the gather node then uses the
    deterministic default).
    """
    lower = text.lower()
    plan: dict[str, Any] = {}

    # metric — "revenue"/"average" are more specific than "total"; check them first.
    metric = None
    if "revenue" in lower or "mrr" in lower or "earning" in lower:
        metric = "sum"
    elif "average" in lower or " avg" in lower or "mean" in lower:
        metric = "avg"
    else:
        metric = next((m for w, m in _METRIC_WORDS.items() if w in lower), None)
    if metric is None:
        # "how many X" / "count of X"
        if "how many" in lower or "count of" in lower or "number of" in lower:
            metric = "count"
    if metric is None:
        return None
    plan["metric"] = metric

    # entity
    entity = next((e for w, e in _ENTITY_WORDS.items() if w in lower), None)
    # "revenue" without an explicit entity implies the customer base (MRR).
    if entity is None and "revenue" in lower:
        entity = "customer"
    if entity is None:
        return None
    plan["of"] = entity

    # dimension (group-by) — "by country / per segment / by plan"
    dim = None
    if " by " in lower or " per " in lower:
        for w, d in _DIMENSION_WORDS.items():
            if w in lower:
                dim = d
                break
    if dim:
        plan["dimensions"] = [dim]

    # numeric dimension for sum/avg
    if metric in ("sum", "avg") and entity:
        plan["of_dimension"] = _default_numeric_dim(entity)

    # filters from entities
    filters: dict[str, Any] = {}
    if entities.get("segment"):
        filters["customer_segment"] = entities["segment"].title()
    if entities.get("plan"):
        filters["subscription_plan"] = entities["plan"].title()
    if filters:
        plan["filters"] = filters

    return plan


def _default_numeric_dim(entity: str) -> str:
    if entity == "customer":
        return "monthly_revenue"
    if entity == "ticket":
        return "resolution_time_hours"
    if entity == "feedback":
        return "rating"
    if entity == "usage":
        return "sessions"
    if entity == "subscription_event":
        return "revenue_change"
    return "monthly_revenue"


def _extract_dates(lower: str) -> list[str]:
    hints: list[str] = []
    for token in ("last 4 weeks", "last four weeks", "last month", "this month", "last week", "this week", "last 8 weeks", "last 3 months"):
        if token in lower:
            hints.append(token)
    # ISO-ish dates
    hints += re.findall(r"\b\d{4}-\d{2}-\d{2}\b", lower)
    return hints


# ---------------------------------------------------------------------------
# Intent scoring
# ---------------------------------------------------------------------------
def _score(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for kw in keywords if kw in text)


def classify_intent(text: str, entities: dict[str, Any]) -> str:
    """Return one of the INTENTS labels. Ids outrank keywords; specificity wins."""
    lower = text.lower()

    if entities.get("customer_ids"):
        return "customer_query"
    if entities.get("ticket_ids"):
        return "ticket_query"
    if entities.get("feedback_ids"):
        return "feedback_query"

    # A capitalized name hint (resolved later) is a customer query — but only
    # when it *looks* like a brand name (camelCase/compound, e.g. "VertexPath",
    # "AtlasDynamics") or the question has customer framing ("tell me about X").
    # Plain capitalized nouns ("Paris", "Monday") must NOT misroute.
    name_hint = entities.get("customer_name_hint")
    if name_hint:
        looks_brand = any(c.isupper() for c in name_hint[1:]) or len(name_hint) >= 10
        customer_framed = _score(lower, CUSTOMER_KEYWORDS) > 0 or " about " in lower or "customer" in lower
        if looks_brand or customer_framed:
            return "customer_query"

    # Count/analytical questions ("how many customers by country") → analytics.
    if ("how many" in lower or "count of" in lower or "number of" in lower or " total " in lower):
        return "analytics_exec"

    scores = {
        "theme_sentiment": _score(lower, THEME_KEYWORDS),
        "analytics_exec": _score(lower, ANALYTICS_KEYWORDS),
        "trend": _score(lower, TREND_KEYWORDS),
        "customer_query": _score(lower, CUSTOMER_KEYWORDS),
        "ticket_query": _score(lower, TICKET_KEYWORDS),
        "feedback_query": _score(lower, FEEDBACK_KEYWORDS),
        "free_text_rag": _score(lower, RAG_KEYWORDS),
    }

    # Irrelevant beats weak data intents only when no data keyword hit at all.
    irrelevant_score = _score(lower, IRRELEVANT_KEYWORDS)
    if irrelevant_score > 0 and sum(scores.values()) == 0:
        return "irrelevant"

    best = max(scores, key=lambda k: (scores[k], _SPECIFICITY[k]))
    if scores[best] == 0:
        return "general"
    return best


# Tie-breaker: more specific intents win when scores are equal.
_SPECIFICITY = {
    "theme_sentiment": 5,
    "analytics_exec": 4,
    "trend": 4,
    "ticket_query": 3,
    "feedback_query": 3,
    "customer_query": 2,
    "free_text_rag": 1,
}


def classify(text: str) -> dict[str, Any]:
    """Full classify-node output: {intent, entities, plan}."""
    entities = extract_entities(text)
    intent = classify_intent(text, entities)
    plan = build_query_plan(text, entities) if intent == "analytics_exec" else None
    return {"intent": intent, "entities": entities, "plan": plan}
