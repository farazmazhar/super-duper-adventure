#!/usr/bin/env python3
"""Project orchestrator — one command to build and serve the demo.

    python run.py build   # one-time: App 1 cleansing + App 2 embedding → data/intelligence.duckdb
    python run.py serve   # runtime: start the FE (uvicorn) → http://localhost:8000
    python run.py eval    # dev/CI: run App 6 evaluator → data/eval/report.json + report.md
    python run.py all     # build → eval → serve (serve is the blocking last step)

Design (docs/internal/08-runner-integration.md):
- Pure subprocess + path checks — no heavy imports at module load.
- `serve` runs uvicorn (Starlette FE, App 7) as the only long-running process; the
  agent spawns the MCP server (App 4) as a stdio subprocess on first chat, and
  guardrails (App 5) run inside the agent loop. Everything the user sees flows
  through the one uvicorn process.
- `serve` fails fast if the DB is missing (hint to run `build` first).
- Subprocesses are cleaned up on exit so no port or MCP stdio process dangles.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "data" / "intelligence.duckdb"
FE_HOST = os.environ.get("FE_HOST", "0.0.0.0")
FE_PORT = os.environ.get("FE_PORT", "8000")

PYTHON = sys.executable


def _check_db() -> None:
    if not DB_PATH.is_file():
        print(f"[run] data/intelligence.duckdb not found — run `{PYTHON} run.py build` first.")
        sys.exit(1)


def build() -> None:
    """App 1 cleansing → App 2 embedding → data/intelligence.duckdb (idempotent).

    App 1 imports a bare `config` module from its own directory, so it runs
    with cwd=apps/cleansing (matching the documented invocation).
    """
    print("== build: App 1 (cleansing) ==")
    subprocess.run(
        [PYTHON, "run.py"],
        cwd=REPO_ROOT / "apps" / "cleansing",
        check=True,
    )
    print("\n== build: App 2 (embedding) ==")
    subprocess.run([PYTHON, "-m", "apps.embedding.run"], cwd=REPO_ROOT, check=True)
    print(f"\n[run] build complete -> {DB_PATH}")


def serve() -> None:
    """Start the FE (uvicorn, App 7) — the only long-running user process.

    The agent (App 3), its MCP server (App 4), and guardrails (App 5) all run
    inside/under this process: the FE imports the agent runner, the agent spawns
    the MCP stdio subprocess on first chat, and guardrails run in the agent loop.
    Ctrl-C tears it all down.
    """
    _check_db()
    uvicorn = shutil.which("uvicorn") or f"{PYTHON} -m uvicorn"
    cmd = [uvicorn, "apps.frontend.server:app", "--host", FE_HOST, "--port", FE_PORT]
    print(f"== serve: FE + agent + MCP + guardrails on http://localhost:{FE_PORT} (Ctrl-C to stop) ==")
    try:
        subprocess.run(cmd, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        print("\n[run] serve stopped.")


def eval_() -> None:
    """Run App 6 evaluator: golden set through the real agent -> data/eval/report."""
    _check_db()
    print("== eval: App 6 (golden set) ==")
    subprocess.run([PYTHON, "-m", "apps.eval.run_eval"], cwd=REPO_ROOT, check=True)


def all_() -> None:
    """build → eval → serve (serve blocks until Ctrl-C)."""
    build()
    eval_()
    serve()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "build":
        build()
    elif command == "serve":
        serve()
    elif command == "eval":
        eval_()
    elif command == "all":
        all_()
    else:
        print(__doc__)
        sys.exit(1 if command else 0)


if __name__ == "__main__":
    main()
