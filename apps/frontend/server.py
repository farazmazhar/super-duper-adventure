"""App 7 — Starlette FE server (replaces the Streamlit app).

Run:  uvicorn apps.frontend.server:app --port 8000

Routes:
  /                    -> chat UI
  /dashboard           -> exec dashboard
  /customer/<id>       -> customer drill-down
  /admin               -> system status
  /api/ask             -> POST {question, rerank_enabled, moderation_enabled, conversation} -> SSE stream
  /api/dashboard       -> JSON KPI + chart data
  /api/customer/<id>   -> JSON profile/tickets/feedback/usage
  /api/admin           -> JSON app status

Chat answers stream over SSE: a single `answer` event carries the full
structured answer + render_hint + trace (the agent runs synchronously; the
stream keeps the door open for token streaming later).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402
from starlette.templating import Jinja2Templates  # noqa: E402

from apps.frontend import api  # noqa: E402

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))


def _json(v: Any) -> str:
    return json.dumps(v, default=api.json_default)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
async def page_chat(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "chat.html", {"active": "chat"})


async def page_dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "dashboard.html", {"active": "dashboard"})


async def page_customer(request: Request) -> HTMLResponse:
    customer_id = request.path_params.get("customer_id", "CUST-0001")
    return templates.TemplateResponse(request, "customer.html", {"active": "customer", "customer_id": customer_id})


async def page_admin(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "admin.html", {"active": "admin"})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
async def api_ask(request: Request) -> StreamingResponse:
    """SSE: run the agent question, emit one `answer` event with the full payload."""
    body = await request.json()
    question = (body.get("question") or "").strip()
    rerank = body.get("rerank_enabled")
    moderation = body.get("moderation_enabled")
    conversation = body.get("conversation") or []

    async def gen():
        if not question:
            yield f"event: error\ndata: {_json({'message': 'Empty question.'})}\n\n"
            return
        try:
            # The agent runs synchronously in a thread (it spawns MCP over stdio).
            result = await asyncio.to_thread(
                api.run_chat, question, rerank, moderation, conversation
            )
            yield f"event: answer\ndata: {_json(result)}\n\n"
        except Exception as exc:  # noqa: BLE001 - never crash the stream
            yield f"event: error\ndata: {_json({'message': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


async def api_dashboard(request: Request) -> JSONResponse:
    return JSONResponse(json.loads(_json(api.dashboard_payload())))


async def api_customer(request: Request) -> JSONResponse:
    customer_id = request.path_params.get("customer_id", "")
    return JSONResponse(json.loads(_json(api.customer_payload(customer_id))))


async def api_admin(request: Request) -> JSONResponse:
    return JSONResponse(json.loads(_json(api.admin_payload())))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
routes = [
    Route("/", page_chat),
    Route("/dashboard", page_dashboard),
    Route("/customer/{customer_id}", page_customer),
    Route("/admin", page_admin),
    Route("/api/ask", api_ask, methods=["POST"]),
    Route("/api/dashboard", api_dashboard),
    Route("/api/customer/{customer_id}", api_customer),
    Route("/api/admin", api_admin),
    Mount("/static", StaticFiles(directory=str(HERE / "static")), name="static"),
]

app = Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
