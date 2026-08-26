"""FE API endpoint tests via Starlette TestClient (no browser needed).

These exercise the read-only dashboard/customer/admin endpoints and the page
routes. The chat endpoint (/api/ask) spawns the agent (MCP subprocess) so it is
tested at the `run_chat` unit level with the agent bundle mocked — the full
subprocess path is covered by the agent integration tests.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from apps.frontend.server import app


def test_chat_page_returns_html() -> None:
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "chat" in r.text.lower()


def test_dashboard_page_returns_html() -> None:
    with TestClient(app) as client:
        r = client.get("/dashboard")
        assert r.status_code == 200


def test_customer_page_returns_html() -> None:
    with TestClient(app) as client:
        r = client.get("/customer/CUST-0001")
        assert r.status_code == 200
        assert "CUST-0001" in r.text


def test_admin_page_returns_html() -> None:
    with TestClient(app) as client:
        r = client.get("/admin")
        assert r.status_code == 200


def test_api_dashboard_json() -> None:
    with TestClient(app) as client:
        r = client.get("/api/dashboard")
        assert r.status_code == 200
        payload = r.json()
        assert "kpis" in payload and "themes" in payload and "risk" in payload


def test_api_customer_json() -> None:
    with TestClient(app) as client:
        r = client.get("/api/customer/CUST-0001")
        assert r.status_code == 200
        payload = r.json()
        assert "profile" in payload and "tickets" in payload and "usage_trend" in payload


def test_api_customer_unknown_returns_null_profile() -> None:
    with TestClient(app) as client:
        r = client.get("/api/customer/CUST-9999")
        assert r.status_code == 200
        assert r.json()["profile"] is None


def test_api_admin_json() -> None:
    with TestClient(app) as client:
        r = client.get("/api/admin")
        assert r.status_code == 200
        payload = r.json()
        assert "quality_report" in payload and "tables" in payload


def test_api_ask_empty_question_errors() -> None:
    with TestClient(app) as client:
        r = client.post("/api/ask", json={"question": ""})
        assert r.status_code == 200
        body = r.text
        assert "error" in body


def test_run_chat_agent_down(monkeypatch) -> None:
    from apps.frontend import api

    monkeypatch.setattr(api, "get_agent_bundle", lambda: None)
    result = api.run_chat("hello", rerank_enabled=True, moderation_enabled=False)
    assert result["error"] is not None
    assert result["answer"] is None


def test_run_chat_with_fake_bundle(monkeypatch) -> None:
    """run_chat threads toggles + conversation into the guarded runner."""
    from apps.frontend import api
    from apps.guardrails.models import AnswerSchema, AnswerSection, RenderHint

    calls: dict = {}

    class _FakeGraph:
        def run(self, question, trace=None, retrieve_kwargs=None, conversation=None):
            calls["retrieve_kwargs"] = retrieve_kwargs
            calls["conversation"] = conversation
            return {
                "answer": AnswerSchema(
                    facts=[AnswerSection(heading="Facts", content="ok", citations=[])],
                    confidence="high",
                    render_hint=RenderHint(kind="markdown"),
                    summary="ok",
                ),
                "context": "ctx",
                "tool_results": {"x": {}},
            }

    class _FakeBundle(dict):
        def __init__(self):
            super().__init__(graph=_FakeGraph(), mcp=None)

    monkeypatch.setattr(api, "get_agent_bundle", lambda: _FakeBundle())
    result = api.run_chat(
        "who is CUST-0001?",
        rerank_enabled=False,
        moderation_enabled=True,
        conversation=[{"role": "user", "content": "hi"}],
    )
    assert result["error"] is None
    assert result["answer"]["summary"] == "ok"
    assert calls["retrieve_kwargs"] == {"rerank_enabled": False}
    assert calls["conversation"] == [{"role": "user", "content": "hi"}]


def test_json_default_serializes_dates() -> None:
    import datetime

    from apps.frontend.api import json_default

    assert json_default(datetime.date(2026, 5, 1)) == "2026-05-01"
