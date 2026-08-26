# AI CustIntel Agent

A prototype that helps Customer-Success, Product, and leadership teams turn customer
data into decisions. It answers natural-language questions over five synthetic data
sources (customers, support tickets, product usage, customer feedback, subscription
events): it identifies problems, finds affected customers, prioritizes by business
impact, surfaces trends as charts, and recommends actions — with citations, confidence,
and a visible trace of everything the agent did.

> **Assignment context:** technical-assignment prototype (7-hour budget, synthetic data
> only, single AI agent). Not production-ready by design — see
> [Scope & trade-offs](#scope--trade-offs).

---

## How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure the AI endpoint (OpenRouter or any OpenAI-compatible API)
cp .env.example .env          # set OPENAI_API_KEY; defaults already point at OpenRouter

# 3. Build the data (cleansing + embeddings) — idempotent; data/ is committed, so often a no-op
python run.py build

# 4. Run the demo (web UI + agent + MCP + guardrails) — everything in one command
python run.py serve           # http://localhost:8000

# 5. Evaluate the agent on the golden set (dev/CI — happy-path scenarios)
python run.py eval                # writes data/eval/report.{json,md}

# Or do it all in one go: build → eval → serve
python run.py all
```

**Test the agent without the UI** (same graph + MCP + guardrails, terminal output):

```bash
python -m apps.agent.cli ask "which enterprise customers are at risk, and why?"
python -m apps.agent.cli ask "what are the top feedback themes?" --no-llm   # deterministic, no key
python -m apps.agent.cli customer       # drill-down (prompts for CUST-xxxx)
python -m apps.agent.cli exec           # executive overview
```

---

## Data: where the synthetic CSVs go

Cleansing reads the raw CSVs from a single directory; embedding reads the cleaned
tables Cleansing produces. The default location is **repo-relative**:

```
_assignment/synthetic_customer_data/
├── customers.csv
├── support_tickets.csv
├── product_usage.csv
├── customer_feedback.csv
└── subscription_events.csv
```

- If the files are already there (the assignment's data), nothing to do — `run.py build`
  picks them up automatically.
- To use CSVs elsewhere, set `DATA_DIR` (env or `.env`) to the directory containing the
  5 files, and `DB_PATH` if you want the DuckDB output elsewhere (default
  `data/intelligence.duckdb`):

```bash
# .env
DATA_DIR=/absolute/path/to/my_synthetic_data
DB_PATH=/absolute/path/to/data/intelligence.duckdb
```

- Cleansing writes all derived tables into the DuckDB file; embedding then fills the
  `vector` schema from the cleaned `fact_*` tables. Both are idempotent and re-runnable.
- The built `data/intelligence.duckdb` is **committed** to the repo (small) so the demo
  starts warm — a fresh clone can skip the build step.

---

## What it does

Given a question, the agent:

1. **classifies** intent + entities (customer/ticket/feedback IDs, segment, dates),
2. **routes** to a specialized node (customer, themes, analytics, trend, rag, general) —
   and **blocks irrelevant** queries,
3. gathers evidence via **MCP tools**, including **RAG retrieval** over embeddings,
4. **reasons** with a single PydanticAI agent,
5. returns a **structured, visual answer** — Facts (cited) / Interpretation /
   Recommendations / Prioritized list / Confidence — plus a `render_hint` (chart, table,
   cards, QA, markdown) and a full **trace**.

Every answer separates **facts supported by data** (with record citations) from
**model interpretation**, **recommendations**, and **uncertainty**. Trend questions
always produce a chart; recommendation/prioritization questions always include those
sections; irrelevant questions get a short bounded reply.

### The six business outcomes

| Business need | Delivered by |
|---|---|
| Identify important customer problems | themes node (feedback themes + ticket breakdown + RAG) |
| Understand which customers are most affected | customer node + risk ranking (with drivers) |
| Prioritize issues by customer + business impact | revenue-at-risk + prioritized list in the answer |
| Explore trends and patterns | trend node → charts |
| Receive practical recommendations | Recommendations section, cited |
| Ask NL questions about the data | the whole classify → route → answer flow |

---

## Architecture at a glance

```
_sources/ (raw CSVs)
   │  Cleansing (DuckDB SQL, build-time)
   ▼
data/intelligence.duckdb   (main / vector / agent schemas)
   │  Embedding (build-time: chunk + embed feedback/tickets)
   ▼
Agent (LangGraph + PydanticAI, RAG, memory, tracing, CLI)
   │  every data action over stdio
   ▼
MCP server (owns the single runtime read-write DuckDB connection; tools + semantic layer + retrieval)
Guardrails (runtime: input moderation + output validation)
Evaluator (dev/CI: 14-question golden set, deterministic metrics; LLM-judge deferred)
Frontend (Starlette web UI: chat, exec, customer, admin, behind-the-scenes; read-only DB)
```

**One-line summary of the key decision:** the agent has *no direct database connection* —
every data action goes through an MCP server over stdio, which owns the single runtime
read-write DuckDB connection. Cleansing + embedding are build-time only. This makes the
data path auditable and the components independently testable.

---

## Component by component

### 1. Cleansing — `apps/cleansing/`

**Stack:** DuckDB SQL (`pipeline.sql`), thin `run.py`, pytest.

**What it does:** reads the 5 raw CSVs into `data/intelligence.duckdb` and produces
clean, normalized, feature-engineered tables: a customer dimension, fact tables
(ticket/feedback/usage/subscription), `aggregate_customer_features` (one row per customer
— the risk-score inputs), `aggregate_segment_metrics`, `aggregate_theme`, and a
`quality_report` audit table. Every cleaning decision is logged (rows dropped, values
normalized, NULLs preserved).

**Why DuckDB SQL:** the cleaning is exactly what SQL is good at, it's idempotent
(`CREATE OR REPLACE`), auditable line-by-line in review, and fast. The pipeline is
data-agnostic (driven by a `pipeline_rules` catalogue, derived trend windows) so it
generalizes beyond this dataset.

**Data-quality decisions (no invented data):** conflicting duplicate customers are
deduped keeping the last row on the assumption it is the latest (logged); missing
revenue / satisfaction / resolution are **kept NULL, never imputed** (revenue-at-risk
flags NULL revenue as unknown); impossible values (rating >5, negative resolution time)
are normalized/NULLed and logged.

### 2. Embedding — `apps/embedding/`

**Stack:** `openai` SDK → OpenRouter `/embeddings` (`voyageai/voyage-4-lite`, 1024-dim),
DuckDB `vector` schema, pytest (mock client).

**What it does:** chunks feedback + tickets **per-record** (one chunk each — they're
~20–40 tokens, so no splitting), embeds them, and stores vectors + metadata in
`vector.embeddings` in the same DuckDB file. Incremental: only new/changed records are
re-embedded (source-hash cache). Also seeds `aggregate_theme` with a rule-based theme +
sentiment baseline.

**Why per-record chunks:** every feedback item / ticket is a short, self-contained unit;
splitting would fragment meaning, and cross-record concatenation would blur which record
a claim came from (hurting citations). Retrieval stays record-level; the agent aggregates
across records via tools.

### 3. Agent — `apps/agent/`

**Stack:** LangGraph (orchestration) + PydanticAI (single agent) + `openai` SDK →
OpenRouter (`deepseek/deepseek-v4-flash-0731`), DuckDB-backed tracing + memory, CLI +
shared runner.

**What it does:** a query-routed LangGraph: `classify` (intent + entities) → `route`
(→ customer/themes/analytics/trend/rag/general, and **blocks irrelevant**) → the routed
node gathers evidence via MCP tools + RAG → `reason` (single PydanticAI agent synthesizes)
→ `answer` (structured, visual, traced). A dedicated `error_handler` node catches
failures, retries once for transient errors, and always returns a safe message + trace.
Long-term memory is **always written** (rule-based hooks) and **retrieved only when
enabled** (`LTM_ENABLED`).

**Why PydanticAI:** a clean single-agent tool-calling loop with strongly-typed tools
(`@agent.tool` + Pydantic schemas) and easy OpenAI-compatible wiring. Every tool returns
structured `{data, source_refs, warnings}`, which guardrails validate and the evaluator
scores. Lighter than a full agent framework; fits "one agent, no multi-agent choreography".

**Why LangGraph:** the flow is genuinely stateful — classify → route → gather → reason →
answer, with a dedicated error handler and tracing hooks on every node. LangGraph makes
the graph explicit, inspectable, and resumable (state + checkpointing) — exactly what the
"Behind the scenes" tracing needs. The agent stays a single PydanticAI agent inside the
graph; LangGraph owns orchestration, not reasoning.

### 4. MCP server — `apps/mcp/`

**Stack:** mcp SDK **v2** (`MCPServer` — FastMCP was renamed in v2), stdio transport,
DuckDB (single runtime read-write connection), `openai` SDK for retrieval.

**What it does:** owns **the** runtime DuckDB connection and exposes every data action as
an MCP tool: fixed tools (profile, risk rank, revenue-at-risk, tickets, feedback, themes,
usage, segments, list, memory), **RAG retrieval** (`retrieve_sources`: embed → cosine →
optional rerank), and a **semantic layer** (`semantic_query`).

**Why MCPServer v2 over FastMCP:** FastMCP was the mcp SDK v1 ergonomic wrapper; in v2 it
was renamed `MCPServer` (same API, `@server.tool()`, `server.run(transport="stdio")`). We
use the current SDK's native name rather than pinning an older version or a deprecated
alias.

**Why the semantic layer lives here (and why not raw SQL):** when no fixed tool fits, the
agent emits a structured `SemanticQuery` (metric/dimensions/filters from a curated
catalog — 7 entities, 8 metrics, 13 dimensions). The MCP server validates it against the
catalog (unknown → reject), translates to parameterized SQL, and executes read-only. The
agent **never writes raw SQL**. It lives inside MCP because MCP owns the only runtime DB
connection — colocating the layer avoids a second connection and keeps the
"agent has no DB" rule. (Also a time-constraint decision: no separate query service.)

**Why single connection / MCP at all:** a standard, language-agnostic tool interface that
makes the single-connection rule hold by construction — the agent has tools, not a DB
handle; the frontend reads read-only.

### 5. Guardrails — `apps/guardrails/`

**Stack:** pure Python (deterministic) + one togglable LLM moderation call
(`meta-llama/llama-guard-4-12b` via OpenRouter), pytest.

**What it does:** input guardrails (prompt-injection heuristics, length cap, input
schema, and togglable llama-guard moderation) run before the graph; output guardrails
(answer-schema validation, confidence required, citation check, PII scan, tool-name +
**tool-argument** validation) run before the answer is shown. Every result is traced and
visible in the UI. Evaluate-all-then-aggregate: `block`s stop, `flag`s annotate.

**Why deterministic + one exception:** guardrails must work even if the LLM is down, so
they're deterministic and cheap; the single moderation call is the documented exception
(togglable, fail-open). Guardrails *block/flag*; the evaluator *scores* — two different
jobs, two components.

### 6. Evaluator — `apps/eval/`

**Stack:** pytest-style runner over the golden set, DuckDB trace read, deterministic
metrics.

**What it does:** scores the agent on **14 curated questions** covering the business
outcomes (customer/ticket/feedback queries, themes, analytics, trend-must-chart,
recommendations, prioritization, irrelevant-blocking, no-data). **Deterministic metrics
only** (Layer 1): routing correctness, entity extraction, tool-call correctness,
retrieval recall@k, citation coverage, render-hint correctness, recommendation/
prioritization presence, confidence honesty. Runs the agent through the same MCP path as
production and writes `data/eval/report.{json,md}`.

**What's deferred:** the **LLM-judge layer** (Layer 2 — faithfulness + usefulness scoring
via OpenRouter, `EVAL_JUDGE_ENABLED`, fail-open) is designed and the env toggle is wired,
but not yet implemented. The deterministic metrics provide the core quality signal; the
judge would add subjective scoring on top.

**Known issues:** the evaluator is implemented and runs, but has known issues that are
being triaged. The design (golden-set schema, metric definitions, two-layer judge) is
captured in the internal docs and ready to extend.

### 7. Frontend — `apps/frontend/`

**Stack:** Starlette (ASGI) + Jinja2 templates + vanilla JS, SSE for streaming chat,
read-only DuckDB access, no build step.

**What it does:** a chat-first web UI with: the conversation (rendering tables, charts,
cards, QA from the agent's `render_hint`), a collapsible **"Behind the scenes"** panel
(agent journey: routing, RAG, MCP calls, LLM tokens, guardrails — from the in-band
trace), customer drill-down, exec dashboard, and an **Admin** panel (system status:
pipeline quality report, embedding meta, tool registry, guardrail events, eval report).
**All DuckDB reads are read-only.**

**Why Starlette + Jinja2 + vanilla JS instead of Streamlit:** the chat UI needs rich,
structured rendering (charts, tables, cards) with a pinned composer and a collapsible
trace panel — a custom thin web layer gives full control with no framework lock-in, no
Streamlit runtime quirks, and a lightweight demo that's easy to review. The agent
`render_hint` contract keeps the UI a dumb renderer: the agent decides structure, the FE
renders it.

---

## Configuration

All config via env vars / `.env` (see `.env.example`):

| Var | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenRouter key (or any OpenAI-compatible). **Never committed** |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `deepseek/deepseek-v4-flash-0731` | chat model |
| `OPENAI_EMBEDDING_MODEL` | `voyageai/voyage-4-lite` | embeddings (1024-dim, Matryoshka) |
| `OPENAI_RERANK_MODEL` | `voyageai/rerank-2.5-lite` | rerank (togglable) |
| `RERANK_ENABLED` | `true` | rerank on/off |
| `MODERATION_ENABLED` | `true` | llama-guard prompt moderation on/off |
| `LTM_ENABLED` | `false` | long-term memory: always written, retrieved only when enabled |
| `EVAL_JUDGE_ENABLED` | `true` | evaluator LLM-judge layer (skipped if off / no key) |

---

## Testing

```bash
python -m pytest                  # unit tests (no network; default excludes integration)
python -m pytest -m integration   # end-to-end tests that spawn the real MCP server
```

---

## Scope & trade-offs

**Deliberately not built:** real-time ingestion, multi-tenant auth, a production
database, rate limiting / abuse control, a hosted tracing backend, hybrid (BM25+vector)
search, and any multi-agent orchestration. Each is a documented decision. The evaluator's
**LLM-judge layer** is designed but deferred (see §6).

**Online-only:** the demo expects `OPENAI_API_KEY` (all LLM/embedding/rerank steps go
through the endpoint). Failures are surfaced + traced, not silently substituted. The
`--no-llm` CLI flag and deterministic guardrails work without a key.

**Security / privacy:** synthetic data only; no API keys committed (`.env` gitignored);
the agent never writes raw SQL (semantic layer is catalog-bounded); the frontend is
read-only; all LLM-bound data is synthetic.

---

## Time budget

Tracked in `time_log.csv` (7-hour assignment budget).
