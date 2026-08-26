"""App 3 — CLI entry point.

    python -m apps.agent.cli ask "which customers are at risk?"
    python -m apps.agent.cli customer CUST-0001
    python -m apps.agent.cli exec

Runs the graph end-to-end (spawning the MCP server as the data path), applies
input guardrails before the graph and output guardrails on the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.agent.agent import build_reason_agent  # noqa: E402
from apps.agent.graph import AgentGraph  # noqa: E402
from apps.agent.mcp_client import McpClient  # noqa: E402
from apps.agent.runner import run_question  # noqa: E402


def _run_question(graph: AgentGraph, question: str) -> dict:
    return run_question(graph, question)


def _print_answer(state: dict) -> None:
    from apps.agent.cli_render import render_answer_text

    answer = state.get("answer")
    if answer is None:
        print("(no answer)")
        return
    print("=" * 74)
    print(render_answer_text(answer))
    gr = state.get("guardrails")
    if gr:
        flags = [g for g in gr if g["severity"] in ("block", "flag")]
        if flags:
            print("\n-- guardrail flags --")
            for g in flags:
                print(f"  [{g['rule']}] {g['message']}")
    print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Customer-intelligence agent (App 3).")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a natural-language question.")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--no-llm", action="store_true", help="Use the deterministic reason fallback (no API key needed).")

    sub.add_parser("customer", help="Customer drill-down (prompts for id).")
    sub.add_parser("exec", help="Executive summary (risk + revenue at risk).")

    args = parser.parse_args()

    with McpClient() as mcp:
        reason_agent = None if getattr(args, "no_llm", False) else build_reason_agent(mcp=mcp)
        graph = AgentGraph(mcp, reason_agent=reason_agent)

        if args.command == "ask":
            question = " ".join(args.question)
            state = _run_question(graph, question)
            _print_answer(state)
        elif args.command == "customer":
            customer_id = input("Customer id (e.g. CUST-0001): ").strip()
            state = _run_question(graph, f"Show me everything about customer {customer_id}")
            _print_answer(state)
        elif args.command == "exec":
            state = _run_question(graph, "Give me an executive overview: which customers are at risk and what revenue is at risk?")
            _print_answer(state)


if __name__ == "__main__":
    main()
