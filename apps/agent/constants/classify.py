"""Classify-stage constants: intent labels, entity regexes, and lexicons.

The classify node (apps/agent/graph.py) uses these to determine intent and
extract entities without an LLM (deterministic, testable, cheap). The spec's
`route` node then maps the intent to a specialized node.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Intent labels (from docs/internal/app3-agent.md §1 classify)
# ---------------------------------------------------------------------------
INTENTS = (
    "customer_query",
    "ticket_query",
    "feedback_query",
    "theme_sentiment",
    "analytics_exec",
    "trend",
    "free_text_rag",
    "general",
    "irrelevant",
)

# ---------------------------------------------------------------------------
# Entity extraction regexes (documented in the spec: CUST-xxxx, TCK-xxxx, FDB-xxxx)
# ---------------------------------------------------------------------------
CUSTOMER_ID_RE = re.compile(r"\bCUST-\d{3,}\b", re.IGNORECASE)
TICKET_ID_RE = re.compile(r"\bTCK-\d{3,}\b", re.IGNORECASE)
FEEDBACK_ID_RE = re.compile(r"\bFDB-\d{3,}\b", re.IGNORECASE)

# Segment / plan mentions (lowercased match targets).
SEGMENTS = ("smb", "mid-market", "enterprise")
PLANS = ("free", "team", "business", "enterprise")

# ---------------------------------------------------------------------------
# Intent-detection lexicons. A question is classified by counting keyword hits
# per intent (most specific intent wins on ties). Order in the classifier:
# customer/ticket/feedback ids first, then keyword scores.
# ---------------------------------------------------------------------------
CUSTOMER_KEYWORDS = (
    "customer",
    "account",
    "who is",
    "profile",
    "drill down",
    "drill-down",
)
TICKET_KEYWORDS = ("ticket", "tickets", "support case", "issue", "bug report", "open cases")
FEEDBACK_KEYWORDS = ("feedback", "review", "rating", "nps", "app store", "survey")
THEME_KEYWORDS = (
    "theme",
    "themes",
    "sentiment",
    "complaint",
    "complaints",
    "what are customers saying",
    "top problems",
    "common problems",
    "pain point",
    "pain points",
)
ANALYTICS_KEYWORDS = (
    "segment",
    "segments",
    "compare",
    "comparison",
    "revenue at risk",
    "at-risk",
    "at risk",
    "risk",
    "mrr",
    "churn",
    "cancel rate",
    "executive",
    "overview",
    "which segment",
    "which plan",
    "prioritize",
    "priority",
    "most important",
    "biggest impact",
    "what should we do",
    "recommend",
    "recommendation",
    "action",
    "actions",
)
TREND_KEYWORDS = (
    "trend",
    "over time",
    "last 4 weeks",
    "last four weeks",
    "last month",
    "weekly",
    "daily",
    "change",
    "increased",
    "decreased",
    "going up",
    "going down",
    "usage trend",
)
RAG_KEYWORDS = (
    "mention",
    "said",
    "saying",
    "evidence",
    "find",
    "search",
    "look for",
    "what do customers",
    "quote",
    "comments",
    "mentions",
)
# ---------------------------------------------------------------------------
# Irrelevant/out-of-scope detection — phrases that are not customer-intelligence.
# If the question matches these (and no data intent), it's blocked.
# ---------------------------------------------------------------------------
IRRELEVANT_KEYWORDS = (
    "weather",
    "sports",
    "recipe",
    "cooking",
    "joke",
    "poem",
    "write a story",
    "who won",
    "stock market",
    "bitcoin",
    "crypto price",
    "translate",
    "what is the capital",
    "tell me a",
    "make me a",
    "your name",
    "who are you",
    "what can you do",
    "are you human",
)
