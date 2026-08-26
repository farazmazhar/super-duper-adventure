"""App 6 — golden set: curated questions with expected agent behavior.

Layer-1 (deterministic) evaluation compares the agent's in-band trace + answer
against these expectations. This set is intentionally **happy-path heavy** (per
the user: "mostly happy scenarios") — it validates that the common questions
work end-to-end: correct routing, right tools, cited facts, and the right
render kind. A few edge cases (irrelevant, no-data) are kept to check bounded
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenQuestion:
    id: str
    question: str
    expected_intent: str
    expected_route: str
    expected_entities: dict[str, object] = field(default_factory=dict)
    expected_tools: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)  # substrings that must appear
    expected_retrieval: list[str] = field(default_factory=list)  # record types
    expected_render: str = "markdown"
    expects_recommendations: bool = False
    expects_prioritization: bool = False
    irrelevant: bool = False
    no_data: bool = False
    notes: str = ""


GOLDEN_SET: list[GoldenQuestion] = [
    # -- customer query ------------------------------------------------------
    GoldenQuestion(
        id="eval-001",
        question="Who is CUST-0001 and what is their account status?",
        expected_intent="customer_query",
        expected_route="customer",
        expected_entities={"customer_ids": ["CUST-0001"]},
        expected_tools=["get_customer_profile", "get_customer_tickets", "get_usage_change"],
        expected_facts=["CUST-0001", "VertexPath"],
        expected_render="table",
        notes="Core customer drill-down: profile + tickets + usage.",
    ),
    GoldenQuestion(
        id="eval-002",
        question="Show me everything about customer CUST-0003",
        expected_intent="customer_query",
        expected_route="customer",
        expected_entities={"customer_ids": ["CUST-0003"]},
        expected_tools=["get_customer_profile", "get_usage_change"],
        expected_facts=["CUST-0003"],
        expected_render="table",
        notes="Customer drill-down variant.",
    ),
    # -- analytics / exec ----------------------------------------------------
    GoldenQuestion(
        id="eval-003",
        question="Which customers are most at risk, and why?",
        expected_intent="analytics_exec",
        expected_route="analytics",
        expected_tools=["rank_customer_risk", "calculate_revenue_at_risk"],
        expected_facts=["risk"],
        expected_render="table",
        expects_prioritization=True,
        notes="Risk ranking is the core management question.",
    ),
    GoldenQuestion(
        id="eval-004",
        question="How much revenue is at risk this quarter?",
        expected_intent="analytics_exec",
        expected_route="analytics",
        expected_tools=["calculate_revenue_at_risk"],
        expected_facts=["revenue"],
        expected_render="table",
        notes="Revenue-at-risk exec metric.",
    ),
    GoldenQuestion(
        id="eval-005",
        question="Which segment has the highest churn?",
        expected_intent="analytics_exec",
        expected_route="analytics",
        expected_tools=["calculate_segment_metrics", "rank_customer_risk"],
        expected_facts=["churn", "segment"],
        expected_render="table",
        notes="Segment comparison against global.",
    ),
    # -- themes / sentiment --------------------------------------------------
    GoldenQuestion(
        id="eval-006",
        question="What are the top feedback themes?",
        expected_intent="theme_sentiment",
        expected_route="themes",
        expected_tools=["get_feedback_themes", "retrieve_sources"],
        expected_facts=["theme", "feedback"],
        expected_retrieval=["feedback", "ticket"],
        expected_render="chart",
        notes="Theme volume -> bar chart.",
    ),
    GoldenQuestion(
        id="eval-007",
        question="What are customers complaining about most?",
        expected_intent="theme_sentiment",
        expected_route="themes",
        expected_tools=["get_feedback_themes", "get_ticket_breakdown"],
        expected_facts=["complaint", "theme"],
        expected_render="chart",
        notes="Complaints = themes + ticket breakdown.",
    ),
    # -- trend (must produce a chart) ---------------------------------------
    GoldenQuestion(
        id="eval-008",
        question="How has usage changed over the last 4 weeks?",
        expected_intent="trend",
        expected_route="trend",
        expected_tools=["get_usage_trend", "get_usage_change"],
        expected_facts=["sessions", "usage"],
        expected_render="chart",
        notes="Trend must render a chart.",
    ),
    # -- ticket / feedback by id --------------------------------------------
    GoldenQuestion(
        id="eval-009",
        question="What is the status of ticket TCK-00042?",
        expected_intent="ticket_query",
        expected_route="themes",
        expected_entities={"ticket_ids": ["TCK-00042"]},
        expected_tools=["retrieve_sources"],
        expected_facts=["TCK-00042"],
        notes="Direct ticket question resolves the record.",
    ),
    GoldenQuestion(
        id="eval-010",
        question="Show me feedback FDB-00042",
        expected_intent="feedback_query",
        expected_route="themes",
        expected_entities={"feedback_ids": ["FDB-00042"]},
        expected_tools=["retrieve_sources"],
        expected_facts=["FDB-00042"],
        notes="Direct feedback question.",
    ),
    # -- recommendations -----------------------------------------------------
    GoldenQuestion(
        id="eval-011",
        question="What should we do about our most at-risk customers?",
        expected_intent="analytics_exec",
        expected_route="analytics",
        expected_tools=["rank_customer_risk", "calculate_revenue_at_risk"],
        expected_facts=["risk", "recommend"],
        expected_render="table",
        expects_recommendations=True,
        expects_prioritization=True,
        notes="Recommendations + prioritization are first-class here.",
    ),
    # -- NL / general --------------------------------------------------------
    GoldenQuestion(
        id="eval-012",
        question="Are complaints linked to reduced usage?",
        expected_intent="general",
        expected_route="general",
        expected_tools=["retrieve_sources", "get_usage_change"],
        expected_retrieval=["feedback", "ticket"],
        expected_facts=["usage"],
        notes="Cross-cutting NL question routes to general + RAG.",
    ),
    # -- edge: irrelevant ----------------------------------------------------
    GoldenQuestion(
        id="eval-013",
        question="What is the weather in Paris?",
        expected_intent="irrelevant",
        expected_route="blocked",
        irrelevant=True,
        expected_facts=["customer-intelligence"],
        expected_render="markdown",
        notes="Out-of-scope question must be blocked with a bounded reply.",
    ),
    # -- edge: no data -------------------------------------------------------
    GoldenQuestion(
        id="eval-014",
        question="Tell me about customer CUST-9999",
        expected_intent="customer_query",
        expected_route="customer",
        no_data=True,
        expected_facts=["CUST-9999"],
        notes="Unknown customer -> insufficient data, low confidence.",
    ),
]


def get_golden_set() -> list[GoldenQuestion]:
    return GOLDEN_SET
