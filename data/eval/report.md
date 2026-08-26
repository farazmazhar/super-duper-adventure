# Evaluation report

- Evaluated at: 2026-08-26T22:39:18.832169
- LLM judge: skipped (Layer-1 deterministic only)
- Questions: 3 — passed 1, failed 2 (pass rate 33%)
- Metric rate: 30/33 (91%)

| id | question | pass | metrics | routing | intent | entities | tools | retrieval | citations | render | rec | prio | irrelevant | confidence |
|---|----------|------|---------|---------|--------|----------|-------|-----------|-----------|--------|-----|------|------------|------------|
| eval-001 | Who is CUST-0001 and what is their accou | ✅ | 11/11 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| eval-002 | Show me everything about customer CUST-0 | ❌ | 10/11 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| eval-003 | Which customers are most at risk, and wh | ❌ | 9/11 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |

> Layer 2 (LLM-as-judge) skipped by design — deterministic Layer-1 only.
