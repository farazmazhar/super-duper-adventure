"""Rule-based theme + sentiment baseline for feedback enrichment.

Deterministic (no LLM). Used to (re)seed `main.aggregate_theme` with
`source='rule'`; App 3's LLM enrichment can overlay it later.

Themes: canonical set with a keyword lexicon. Order matters — most specific
first, `other` fallback last.
Sentiment: rating-led with keyword override:
  - rating <= 2 -> 'negative', rating >= 4 -> 'positive' (unless keywords say
    otherwise), rating 3 or NULL -> keyword-based.
  - Keyword evidence overrides the rating baseline (e.g. "great" with a low
    rating is still counted positive — words carry more signal than a stale
    star).
"""

from __future__ import annotations

import re

# --- Theme lexicon (ordered: most specific first) -----------------------------
# Ordering contract: the classifier returns the FIRST lexicon hit, so a text
# like "integration keeps failing" must hit `integrations` (topic) before the
# generic `product_quality` bucket — the topic word is the stronger signal.
# Concrete rules that make this true:
#   * topical themes (billing, search, integrations, ...) come BEFORE the
#     generic `product_quality` bucket, AND
#   * the generic bucket must not contain tokens that topical regexes also
#     match (e.g. 'fails?' is in the generic bucket but 'export' is a
#     reporting word — "crashes on export" is a reporting complaint, not a
#     generic quality one).
# Keep generic words ('slow', 'fail') out of topical themes, and keep topical
# words ('export', 'import', 'connect', 'docs') out of the generic bucket.
THEME_LEXICON: list[tuple[str, re.Pattern[str]]] = [
    ("billing", re.compile(r"(invoice|billing|payment|charge|refund|pricing|price|plan cost|renewal|itemized)", re.I)),
    ("search", re.compile(r"(search|find|discover|browse)", re.I)),
    ("integrations", re.compile(r"(api|integration|webhook|import|export|sso|sync|connect)", re.I)),
    ("reporting", re.compile(r"(report|dashboard|analytics|metric|chart|export)", re.I)),
    ("collaboration", re.compile(r"(collaborat|team|share|permission|role|access|multi-user)", re.I)),
    ("onboarding", re.compile(r"(onboard|setup|getting started|tutorial|learn|training|documentation|docs)", re.I)),
    ("support", re.compile(r"(support|help|cs|assistance|response time|reply)", re.I)),
    ("mobile", re.compile(r"(mobile|iphone|android|app)", re.I)),
    ("performance", re.compile(r"(performance|speed|fast|slow|lag|latency)", re.I)),
    ("product_quality", re.compile(r"(bug|crash|error|broken|glitch|outage|freeze|stuck|down)", re.I)),
]

# --- Sentiment keywords -------------------------------------------------------
POSITIVE_WORDS = re.compile(
    r"(great|love|excellent|amazing|awesome|happy|impressed|best|easy|fast|good|nice|works well|seamless)", re.I
)
NEGATIVE_WORDS = re.compile(
    r"(bad|worst|terrible|hate|awful|frustrat|disappoint|annoy|angry|unhappy|slow|broken|bug|crash|error|fails?|downtime)", re.I
)


def classify_theme(text: str | None) -> str:
    """Map free text to one canonical theme (most specific lexicon hit first)."""
    if not text:
        return "other"
    lowered = text.lower()
    for theme, pattern in THEME_LEXICON:
        if pattern.search(lowered):
            return theme
    return "other"


def classify_sentiment(text: str | None, rating: int | float | None) -> str:
    """Sentiment baseline: rating-led, keyword override."""
    if text:
        lowered = text.lower()
        has_pos = bool(POSITIVE_WORDS.search(lowered))
        has_neg = bool(NEGATIVE_WORDS.search(lowered))
        if has_pos and not has_neg:
            return "positive"
        if has_neg and not has_pos:
            return "negative"
        if has_pos and has_neg:
            return "mixed"
    if rating is not None:
        if rating <= 2:
            return "negative"
        if rating >= 4:
            return "positive"
        return "neutral"
    return "neutral"


def enrich_feedback_row(
    feedback_id: str, customer_id: str | None, created_at, text: str | None, rating: int | float | None
) -> tuple[str, str | None, object, str, str]:
    """Classify one feedback row -> (feedback_id, customer_id, created_at, theme, sentiment)."""
    return (feedback_id, customer_id, created_at, classify_theme(text), classify_sentiment(text, rating))
