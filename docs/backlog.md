# Phase 1 backlog

- Date: 2026-11-22
- Source: development plan (Phase 1 tasks), updated with Phase 0 spike results
- Full task cards (objective, files, approach, tests, acceptance, risks) are in the development plan document.

## Changes from the plan after Phase 0

| Task | Change | Reason |
| --- | --- | --- |
| T1.01 | Pin DuckDB 2.0.1, SQLGlot 30.33.0, pandas 3.1.1, Streamlit 1.67.0 (or newer patch releases re-tested) | Versions validated in spikes |
| T1.05 | Written against pandas 3 string dtype behaviour | pandas 3 default |
| T1.06 | Add a catalog integrity check after every query; mark the session corrupted and ask for a reload if the table changed | Spike note A1: DuckDB has no read-only in-memory mode |
| T1.08 | Function allowlist keyed on SQLGlot expression classes; rule V4 rejects path-like or schema-qualified table names; default-deny node allowlist; executed text regenerated from the validated tree; dangerous functions rejected; new category `unsupported_syntax`; single-row join refinement (security.md, implementation notes) | Spike notes B1 and B2; probes during T1.08 |
| T1.09 | Call `SessionDatabase.verify_catalog()` after every query, inside `db.lock`; map `OutOfMemoryException` to `out_of_memory`; run each query with a timer calling `db.interrupt()`; execute only `ValidationResult.safe_sql`, and confirm on the session connection with `extract_statements` that it is exactly one `SELECT` | T1.06: spilling is disabled, so an over-cap query fails instead of spilling to disk. T1.08: the validator already makes the same check on a parse-only connection |
| T1.10 | Use the Responses API (not Chat Completions); add `LLM_PRICES` so cost is computed from configured prices, since model names and prices are unknown until spike T0.4 (c); count calls with a per-question `CallBudget`; carry `usage` on `LLMError` so billed calls with unusable answers are still counted; the `correct` stage uses the generate model | ADR 0004 leaves the API and price source open; R9 asks for configurable models and cost logging |
| T1.11 | The schema card is already rendered by `analyst/ingestion/schema_card.py` (names, types, ranges, frequent values, 80-character cut, budget fitting). The context builder adds sample rows, region rules and the untrusted-data block, and should check the card's token estimate (3 characters per token) against real token counts from the API | T1.07: no tokenizer is available offline |
| T1.12 | Starts only after spike T0.4 (c) confirms or replaces ADR 0006 | Model comparison pending |
| T1.14 | Split in two PRs because the explainer call needs the LLM client from T1.10. Part 1 is the grounding check, caveats for truncation and missing values, the empty-result answer and the template answer. Part 2 is the explainer prompt, its input (all rows up to 50, else 20 rows plus summary statistics, fenced as untrusted data) and the one regeneration. Rounding is strict (half a unit of the last digit, trailing zeros not treated as rounding) until the benchmark's fallback rate says otherwise | ADR 0017 |
| T1.18 | Progress indicator during Excel parsing | Spike (d): up to about 14 s at the size limit |
| T1.20 | Start on a 2 GB Railway instance; memory alert at 75% | Spike (d) sizing |

## Backlog

| ID | Task | Size | Depends on | Milestone | Status |
| --- | --- | --- | --- | --- | --- |
| T1.01 | Project scaffold, settings, logging, CI | S | Repository created | M1 | Done |
| T1.02 | Core models and error types | S | T1.01 | M1 | Done |
| T1.03 | CSV loader | M | T1.02 | M1 | Done |
| T1.04 | Excel loader | M | T1.02 | M1 | Done |
| T1.05 | Type inference and column mapping | M | T1.03 | M1 | Done |
| T1.06 | Session DuckDB, lockdown, catalog integrity check | M | T1.02 | M1 | Done |
| T1.07 | Profiler and schema card | M | T1.05, T1.06 | M1 | Done |
| T1.08 | SQL validator | L | T1.06 | M2 | Done |
| T1.09 | Query executor | M | T1.08 | M2 | Done |
| T1.10 | LLM client and fake | M | T1.02 | M3 | In review |
| T1.11 | Prompts, region rules, context builder | M | T1.07 | M3 | Ready |
| T1.12 | Intent and SQL generator | M | T1.10, T1.11, spike (c) | M3 | Blocked on OpenAI key |
| T1.13 | Corrector and retry loop | M | T1.09, T1.12 | M3 | Waiting |
| T1.14 | Explainer and grounding check | M | T1.09, T1.10 | M3 | In progress: grounding and result notes in review, explainer call waits for T1.10 to merge |
| T1.15 | Chart recommender and builder | M | T1.02 | M4 | Ready |
| T1.16 | Conversation memory | S | T1.02 | M3 | Ready |
| T1.17 | Orchestrator | M | T1.12 to T1.16 | M3 | Waiting |
| T1.18 | Streamlit UI | L | T1.04, T1.07, T1.17 | M4 | Waiting |
| T1.19 | Benchmark harness and datasets | M | T0.5, T1.17 | M4 | Waiting |
| T1.20 | Railway deployment | S | T1.18 | M5 | Waiting (needs Railway account) |
| T1.21 | Alpha hardening and review | M | T1.19, T1.20 | M5 | Waiting |

## External dependencies on Fahad

| Needed for | What |
| --- | --- |
| T1.01 | Empty GitHub repository `ai-data-analyst` |
| Spike (c), T1.12 onward | `OPENAI_API_KEY` in the project's cloud environment, from a dedicated OpenAI project with a spend cap |
| T1.20 | Railway account and project; `OPENAI_API_KEY` and `APP_PASSWORD` set as Railway variables |
