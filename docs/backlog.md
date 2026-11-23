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
| T1.11 | The schema card is already rendered by `analyst/ingestion/schema_card.py` (names, types, ranges, frequent values, 80-character cut, budget fitting). The context builder adds sample rows, region rules and the untrusted-data block. The builder returns its token estimate (3 characters per token) so that T1.12 can log it beside the API's real `input_tokens`; the comparison itself needs the key. The prompt tells the model to refuse requests to change data by returning no SQL and no clarification, which the generator (T1.12) must turn into a failed answer with the model's explanation | T1.07: no tokenizer is available offline. Benchmark case I02 expects `failed` |
| T1.16 | Memory returns the last three turns and the carried assumptions. T1.17 must pass `memory.recent()` as `history` to the context builder and add a parameter there for the carried assumptions (the builder is `analyst/sql/context.py`, T1.11), and make the builder's `HISTORY_TURNS` and memory's `WINDOW_TURNS` one constant | ADR 0011: confirmed assumptions carry forward beyond the window |
| T1.12 | Planned to start after spike T0.4 (c). Built ahead of it against the scripted fake, because `Generator` returns the same things (`kind`, `sql`, `message`, `unknown_columns`) whether it makes one call or two, so a switch away from ADR 0006 changes only its inside. Only the quality of real answers waits for the key. A refusal (no SQL, no question) is a `refused` generation that T1.17 turns into a `failed` answer carrying the model's explanation (case I02). T1.17 sends `unknown_columns` to the corrector before validation, passes the same `ContextBuilder` to the generator and the explainer, and keeps one call in the budget for the explainer | Model comparison pending |
| T1.13 | The corrector and the retry loop are two modules: `Corrector` makes one correction call on the generator's conversation, and `QueryLoop` runs validate, execute and correct within the limits (2 corrections, one call kept for the explainer). What is retried follows the validator's `status`, so a syntax error (V12) ends the turn although the older architecture table said it was retried; the loop needs no change if V12 is relaxed. A timeout or out-of-memory failure is corrected once. The intent check (`unknown_columns`) is a soft first correction. The database's message goes to the model inside a fenced `PROBLEM` block, and schema-only mode removes quoted text from it and withholds the messages that can echo a cell. T1.17 passes the generation, the profile, the question's `CallBudget` and the schema-only flag to `QueryLoop.run`, and maps the `RunOutcome` status to `AnalystResponse` | Architecture section 7; FR-4.4 |
| T1.17 | `Analyst` (`analyst/pipeline/analyst.py`) wires the stages as in architecture section 6, one per dataset. The context builder takes the carried assumptions (listed in the `HISTORY` block before the turns), the builder's window and memory's are one constant, and `clean_question` is public so the orchestrator can refuse a bad question without a model call. Rate limiting (FR-10.2) stays with the interface (T1.18), which owns the session's question timestamps; `RateLimitedError` is its to raise. Wall-clock latency, not the sum of call latencies, goes in `Usage.latency_ms` (ADR 0013). A bug in a stage is a plain `failed` answer, not a stack trace | Architecture sections 3 and 6; the notes on T1.12, T1.13, T1.14 and T1.16 |
| T1.14 | Split in two PRs because the explainer call needs the LLM client from T1.10. Part 1 is the grounding check, caveats for truncation and missing values, the empty-result answer and the template answer. Part 2 is the explainer prompt, its input (all rows up to 50, else 20 rows plus summary statistics, fenced as untrusted data) and the one regeneration. The explainer also refuses numbers spelled out from eleven up (the check cannot read them), strips links, images and addresses from the model's text, and falls back to the template on any failed call or exhausted budget. Rounding is strict (half a unit of the last digit, trailing zeros not treated as rounding) until the benchmark's fallback rate says otherwise. T1.17 wires the explainer: one `ContextBuilder` per session shared with generation, one call left in the question's budget for it, and `Explanation` mapped to `AnalystResponse` (`message`, `caveats`, `fallback_used`, `usage`, `grounded` true) | ADR 0017 |
| T1.18 | The page is split in two: `analyst/session/` holds everything that is not drawing (`AnalystSession`, the file loading order, the rate limiter, idle removal, and the text and tables the page shows), so it is tested without a browser and counted in coverage; `analyst/ui/` only draws and acts on clicks. A file is *staged* and read only when the user presses *Load file*, so the sheet, separator and header row can be chosen first (FR-1.2, FR-1.4); replacing a dataset with a conversation asks first (FR-7.4). Excel parsing shows a spinner that says it can take up to 15 seconds (spike (d)); the orchestrator gains an optional `on_progress` callback (understanding, running, explaining) for the progress line while a question runs (FR-9.2), and a callback that raises never fails the question. The rate limit (FR-10.2) is in the session and is not reset by a new file, a new conversation or idle removal. Idle removal happens on the user's return and by a sweeper thread holding weak references only. After a load the uploaded bytes are deleted from Streamlit's buffer. The privacy notice said schema-only mode sends "no sample rows or values", but the rows of each result are still sent so they can be explained (privacy.md section 3); the notice, `privacy.md` and the switch's help text now say so, and a test keeps the code and the document equal. Downloads defuse spreadsheet formulas. No cost or token count is shown to the user. Without an OpenAI key the page says questions are switched off and still shows the data | Spike (d): up to about 14 s at the size limit; FR-1.2, FR-1.4, FR-7.4, FR-9.2, FR-10.2, FR-10.3; security T8, T9, T12 |
| T1.19 | The six datasets are generated by `eval/generate_datasets.py` from a fixed seed, in the messy forms of `eval/README.md`, and all load through the real ingestion with exactly the planned SQL names and types, so no gold query changed. The runner loads each under the name of its table, so the gold SQL reads `FROM sales`. Gold queries run through the real validator and executor, which makes the ADR 0013 line "gold SQL wrongly blocked" measurable. Scoring ignores column names and allows extra columns, accepts any tied row at a LIMIT cut-off, and treats no rows and a single NULL as the same answer. A run with `--oracle` answers with the gold SQL to prove the harness without a key; reports from it are named for it. Held-out cases need `--milestone`, and `--max-cost-usd` needs prices for both models. The seed set's held-out count was 12, not 13. 39 cases were added (80 in all, 25 held out). The first model run is `eval/reports/2026-11-23-4711510-dev.md` (dev split, 55 cases, US$0.05): it is below the ADR 0013 thresholds on accuracy, clarification and follow-ups, and the manual review of 30 answers is still to be done by a person | ADR 0013, ADR 0017; eval/README.md |
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
| T1.10 | LLM client and fake | M | T1.02 | M3 | Done |
| T1.11 | Prompts, region rules, context builder | M | T1.07 | M3 | Done |
| T1.12 | Intent and SQL generator | M | T1.10, T1.11, spike (c) | M3 | Done (built on the fake; real-model quality check needs the key) |
| T1.13 | Corrector and retry loop | M | T1.09, T1.12 | M3 | Done |
| T1.14 | Explainer and grounding check | M | T1.09, T1.10 | M3 | Done |
| T1.15 | Chart recommender and builder | M | T1.02 | M4 | Done |
| T1.16 | Conversation memory | S | T1.02 | M3 | Done |
| T1.17 | Orchestrator | M | T1.12 to T1.16 | M3 | Done |
| T1.18 | Streamlit UI | L | T1.04, T1.07, T1.17 | M4 | Done |
| T1.19 | Benchmark harness and datasets | M | T0.5, T1.17 | M4 | Done |
| T1.20 | Railway deployment | S | T1.18 | M5 | Waiting (needs Railway account) |
| T1.21 | Alpha hardening and review | M | T1.19, T1.20 | M5 | Waiting |

## External dependencies on Fahad

| Needed for | What |
| --- | --- |
| T1.01 | Empty GitHub repository `ai-data-analyst` |
| Spike (c), T1.12 onward | `OPENAI_API_KEY` in the project's cloud environment, from a dedicated OpenAI project with a spend cap |
| T1.20 | Railway account and project; `OPENAI_API_KEY` and `APP_PASSWORD` set as Railway variables |
