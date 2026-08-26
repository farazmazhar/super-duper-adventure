"""App 6 — evaluator runner.

Runs the golden set through the real agent (via the MCP client, same as
production), computes Layer-1 deterministic metrics, and writes
`data/eval/report.json` + `data/eval/report.md`.

Usage:
    .venv/bin/python -m apps.eval.run_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.agent.agent import build_reason_agent  # noqa: E402
from apps.agent.graph import AgentGraph  # noqa: E402
from apps.agent.mcp_client import McpClient  # noqa: E402
from apps.agent.runner import run_question  # noqa: E402
from apps.agent.tracing import TraceContext  # noqa: E402
from apps.common.config import settings  # noqa: E402
from apps.eval import judge
from apps.eval.golden_set import get_golden_set
from apps.eval.metrics import evaluate_question

EVAL_DIR = REPO_ROOT / "data" / "eval"


def _run_one(graph: AgentGraph, question: str) -> dict[str, Any]:
    trace = TraceContext(question=question)
    # Moderation off for eval: the golden set tests agent reasoning quality,
    # not the llama-guard pre-check (which can false-positive on benign
    # business questions). Guardrail behavior is tested in App 5's tests.
    state = run_question(graph, question, trace=trace, moderation_enabled=False)
    answer = state.get("answer")
    answer_dict = answer.model_dump() if hasattr(answer, "model_dump") else (answer or {})
    tr = state.get("trace")
    if not isinstance(tr, dict):
        tr = trace.to_dict()
    return {"answer": answer_dict, "trace": tr}


def run_eval(limit: int | None = None) -> dict[str, Any]:
    golden = get_golden_set()
    if limit:
        golden = golden[:limit]

    with McpClient() as mcp:
        reason_agent = None if settings.openai_api_key is None else build_reason_agent(mcp=mcp)
        graph = AgentGraph(mcp, reason_agent=reason_agent)

        results: list[dict[str, Any]] = []
        for g in golden:
            try:
                run = _run_one(graph, g.question)
                metric_result = evaluate_question(g, run["trace"], run["answer"])
                j = judge.judge_question(g, run["answer"], run["trace"]) if judge.judge_enabled() else {"skipped": True}
                results.append({
                    **metric_result,
                    "question": g.question,
                    "judge": j,
                    # store the trace + answer so the report is debuggable
                    "trace": run["trace"],
                    "answer": run["answer"],
                })
                print(f"  {g.id:10} {g.question[:60]:60} {'PASS' if metric_result['passed'] else 'FAIL'}")
            except Exception as exc:  # noqa: BLE001 - never abort the whole eval
                print(f"  {g.id:10} {g.question[:60]:60} ERROR {type(exc).__name__}: {exc}")
                results.append({
                    "question_id": g.id, "question": g.question,
                    "passed_metrics": 0, "total_metrics": 0, "passed": False,
                    "error": f"{type(exc).__name__}: {exc}", "judge": {"skipped": True},
                })

    overall = _summarize(results)
    report = {
        "evaluated_at": __import__("datetime").datetime.now().isoformat(),
        "judge_enabled": judge.judge_enabled(),
        "overall": overall,
        "results": results,
    }
    _write_report(report)
    return report


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    metric_ok = sum(r["passed_metrics"] for r in results)
    metric_total = sum(r["total_metrics"] for r in results)
    return {
        "questions": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": round(passed / n, 3) if n else 0.0,
        "metric_ok": metric_ok,
        "metric_total": metric_total,
        "metric_rate": round(metric_ok / metric_total, 3) if metric_total else 0.0,
    }


def _write_report(report: dict[str, Any]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (EVAL_DIR / "report.md").write_text(_render_markdown(report))
    print(f"\nReport written to {EVAL_DIR / 'report.json'} and {EVAL_DIR / 'report.md'}")


def _render_markdown(report: dict[str, Any]) -> str:
    o = report["overall"]
    lines = [
        "# Evaluation report",
        "",
        f"- Evaluated at: {report['evaluated_at']}",
        f"- LLM judge: {'enabled' if report['judge_enabled'] else 'skipped (Layer-1 deterministic only)'}",
        f"- Questions: {o['questions']} — passed {o['passed']}, failed {o['failed']} "
        f"(pass rate {o['pass_rate']:.0%})",
        f"- Metric rate: {o['metric_ok']}/{o['metric_total']} ({o['metric_rate']:.0%})",
        "",
        "| id | question | pass | metrics | routing | intent | entities | tools | retrieval | citations | render | rec | prio | irrelevant | confidence |",
        "|---|----------|------|---------|---------|--------|----------|-------|-----------|-----------|--------|-----|------|------------|------------|",
    ]
    for r in report["results"]:
        m = r.get("metrics", {})
        cells = [
            r["question_id"], r["question"][:40].replace("|", "/"),
            "✅" if r["passed"] else "❌",
            f"{r['passed_metrics']}/{r['total_metrics']}",
        ]
        for key in ("routing", "intent", "entities", "tools", "retrieval", "citations", "render", "recommendations", "prioritization", "irrelevant", "confidence"):
            mm = m.get(key, {})
            cells.append("✅" if mm.get("passed") else "❌")
        lines.append("| " + " | ".join(cells) + " |")
    if not report["judge_enabled"]:
        lines += ["", "> Layer 2 (LLM-as-judge) skipped by design — deterministic Layer-1 only."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_eval()
