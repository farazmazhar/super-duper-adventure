"""Theme + sentiment lexicon tests (deterministic, no DB)."""

from __future__ import annotations

from apps.embedding.themes import classify_sentiment, classify_theme, enrich_feedback_row


def test_theme_classification() -> None:
    assert classify_theme("The invoice was wrong") == "billing"
    assert classify_theme("The screen crashes and freezes") == "product_quality"
    assert classify_theme("Search is slow to find results") == "search"
    assert classify_theme("API integration failed") == "integrations"
    assert classify_theme("The dashboard chart looks wrong") == "reporting"
    assert classify_theme("Multi-user permissions for the team") == "collaboration"
    assert classify_theme("Setup docs and onboarding were great") == "onboarding"
    assert classify_theme("Mobile app on android") == "mobile"
    assert classify_theme("Everything is fast, no latency") == "performance"
    assert classify_theme("No idea what to say here") == "other"
    assert classify_theme(None) == "other"


def test_theme_lexicon_order_first_match_wins() -> None:
    # "search" outranks the generic "slow" (product_quality) — a search complaint
    # is a search problem, not a general quality issue
    assert classify_theme("Search is slow to find results") == "search"
    # "export" (integrations) beats generic failure words
    assert classify_theme("It crashes on export") == "integrations"
    # "dashboard" (reporting) beats generic failure words
    assert classify_theme("The dashboard crashes on load") == "reporting"
    # no topic keyword -> falls through to the generic quality bucket
    assert classify_theme("The screen crashes and freezes") == "product_quality"


def test_sentiment_rating_led_with_keyword_override() -> None:
    # keyword evidence overrides a stale rating
    assert classify_sentiment("Love the search feature", 2) == "positive"
    assert classify_sentiment("The invoice was terrible", 4) == "negative"
    assert classify_sentiment("It crashes constantly", None) == "negative"
    # neutral text falls back to rating baseline
    assert classify_sentiment("The search feature", 5) == "positive"
    assert classify_sentiment("The search feature", 1) == "negative"
    assert classify_sentiment("The search feature", 3) == "neutral"
    assert classify_sentiment(None, 3) == "neutral"
    assert classify_sentiment(None, None) == "neutral"


def test_sentiment_mixed() -> None:
    assert classify_sentiment("Great API docs but slow support", 4) == "mixed"


def test_enrich_feedback_row_shape() -> None:
    row = enrich_feedback_row("FDB-0001", "CUST-0001", "2026-05-10", "Love the search", 5)
    assert row == ("FDB-0001", "CUST-0001", "2026-05-10", "search", "positive")
