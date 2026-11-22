# Testing strategy

- Version: 1.0
- Date: 2026-11-22
- Related: [benchmark](../eval/README.md), [ADR 0013](adr/0013-mvp-acceptance-thresholds.md), [security](security.md)

## 1. Principles

1. Code is not done until its tests pass. Writing it is not evidence that it works.
2. Expected numbers come from an independent calculation (hand-written SQL, pandas, or a hand-computed fixture), never from the code under test.
3. Expected results are never changed to make a test pass. A wrong expectation is fixed in its own commit with the reason.
4. Tests are never skipped or weakened to get a green build.
5. The default test run needs no network and no API key.

## 2. Layers

| Layer | Folder | Network | Runs | Purpose |
| --- | --- | --- | --- | --- |
| Unit | `tests/unit/` | No | Every push (CI) | One module at a time |
| Integration | `tests/integration/` | No (fake LLM) | Every push (CI) | Pipeline paths through `ask()` |
| UI | `tests/e2e/` | No (fake LLM) | Every push (CI) | Streamlit AppTest flows |
| Live benchmark | `eval/` | Yes (OpenAI) | On demand, with a cost cap | Accuracy, latency, cost |
| Deployment smoke | `tests/smoke/` | Yes | After each deploy | The deployed app answers one question |

Markers: `@pytest.mark.slow` for tests over 5 s (large files); `@pytest.mark.live` for anything calling OpenAI, excluded by default.

## 3. Coverage by module

| Module | Key tests | Fixtures |
| --- | --- | --- |
| CSV loader | Encodings (UTF-8, UTF-8 BOM, Latin-1, Windows-1252); delimiters (comma, semicolon, tab, pipe); quoted newlines; empty file; header only; duplicate and blank headers; 50 MB timing | `fixtures/csv/` |
| Excel loader | Multi-sheet; title rows; merged cells; Excel date serials; formulas with cached values; `.xls`; oversized; zip bomb | `fixtures/excel/` |
| Normaliser | Table-driven: each tricky column with its expected type, flags and identifier; ambiguous day/month; numeric text with `$`, `,`, `%`; censored `<0.5`, `ND`; reserved-word and duplicate identifiers; non-ASCII headers | Inline tables |
| Session store | Every lockdown attack in security.md section 5 errors; two sessions isolated; memory cap error is clean; catalog integrity check detects a dropped table | In-memory |
| Profiler | Every value matches pandas on fixtures; schema card under budget for 150 columns | Fixtures |
| Validator | 60+ adversarial queries rejected with the right category; 40+ valid queries (including all benchmark gold SQL) accepted; one test per allowlisted function | `tests/unit/sql_cases.yaml` |
| Executor | Timeout fires within limit + 1 s; truncation flag; each error category; forged or unvalidated results never run; catalog check after every query; all validator corpora and allowlisted functions run end to end on the locked-down session | In-memory |
| LLM client | Request shape (endpoint, strict schema, `store=false`, model per stage); retry on 429, 5xx and connection errors with `Retry-After`; no retry on 4xx, timeout or exhausted quota; refusal, truncation and misfit are `invalid_output` and carry their usage; not configured fails before any request; budget charged once per call; cost from configured prices; no prompt, answer or key in logs | Mocked HTTP transport, real SDK |
| Context builder | Snapshots of the system prompt and of the user message in full and schema-only mode (`tests/unit/snapshots/`, rewritten with `UPDATE_SNAPSHOTS=1`); canary values prove schema-only mode sends no cell value; file text cannot forge a marker line; at most three sample rows, three turns and one result preview; every construct the prompt recommends passes the validator; the function list equals the allowlist | Fixtures and the synthetic sales table |
| Generator | Each kind (sql, clarification, refused); the clarification gate (asked with and without SQL, asked twice); inconsistent and unusable answers asked for again once and then raised with the cost of both calls; failed calls raised at once; spent budget; links and images removed from every text for the user and not from the SQL; intent columns against the table (misspelt, original header, capitalisation, `*`, sort aliases, suggestions limited to three); logs carry counts and the token estimate, never text | Fake LLM |
| Corrector | What each validator and executor verdict becomes (reason, database message cut and flattened, column types for type problems, whole-word and case-insensitive column match, 12-column cap); a rejected or accepted verdict can never become a problem; schema-only mode removes quoted text and withholds the messages that can echo a cell; the call continues the conversation (assistant answer, then the problem block, then the request) and a second correction continues from the first; charged to the budget; failures raised with their cost; text cleaned and SQL untouched; no-SQL answers give a message; a forged marker in the database's message cannot close the block; logs carry no text | Fake LLM; real validator verdicts |
| Query loop | Good query makes no call; each fixable category corrected and revalidated; the limit of 2 (and 0, 1, from settings); every rejected category ends the turn and is never sent, including a correction that is itself rejected; real conversion and binder failures with the database's own message, and schema-only mode sending no cell value; timeout and memory corrected once, together, without using the other correction; a call kept for the explainer; failed, unusable and empty correction answers; the intent problem (corrected, shared limit, soft in each way it can fail); a corrupted session not caught; logs without text | Fake LLM; real validator, executor and session database; stubs for failures a small table cannot produce |
| Grounding | Number formats (1,234.5; 12%; -3.2; 1e3; 1.2 million); identifiers, ordinals and list numbering ignored; rounding half-unit boundaries; percent against fractions; numbers from the question, assumptions, dates, text values and column names allowed; invented and derived numbers caught; key numbers checked against their column; result caveats, empty-result and template answers | Inline, with expected values written by hand |
| Explainer | Grounded answer returned as written; one regeneration that names the bad numbers and shows the model its own answer; a second failure gives the template; an empty result makes no call; failed, unusable, empty and budget-limited calls fall back with the right reason and keep their cost; spelled-out numbers refused; links, images and addresses removed so data cannot leave in an image address; logs carry no text | Fake LLM |
| Result digest | Rows shown (50 and 51 rows, 20 sampled, 30 columns); each statistic against a hand calculation; non-finite values; dates, text, boolean and unsummarised columns; the numbers handed to the grounding check | Inline |
| Explanation prompt | Snapshots of the system prompt and of the user message for a short and a long result; marker forging in the result, assumptions and SQL; statistics and row limits; truncation and hidden-column notes | Fixtures |
| Charts | Result shape matrix to chart type; model suggestion used, refused or `none`; every refusal reason; slot order, legend, end label, value label, panel and gap rules at their boundaries; markup in labels escaped; dark mode and texture; theme contrast | Inline |
| Memory | Window of 3 and turn limit; carried assumptions (answered turns only, no repeats, latest word last, outlive the window, capped); reset clears everything; sessions do not share memory | Inline |
| Orchestrator | Every path in the question flow diagram: answered, clarification, refusal, rejected, corrected, unfixed, a correction that would take the explainer's call, all five calls used, template fallback, empty result, spent budget, row cap; failed, unusable and unconfigured model calls with their cost; a damaged session, a bug (plain message, type-only log) and a failing chart; bad questions make no call and leave no turn; the prompt carries turns and carried assumptions, which outlive the window and are not set by failed or clarification turns; schema-only mode (no sample rows, no cell value in a correction, result still explained); one marker code across stages and a different one per analyst; wall-clock latency; logs without text | Fake LLM; real validator, executor, corrector, explainer, recommender and session database |
| Rate limiter | The boundary of the window (an event exactly one window old is gone); refused events are not counted; the wait shown is until the oldest event leaves; whole seconds under a minute, minutes above; bad limits refused; a clock that stands still | Fake clock |
| Display | Every cell of the profile tables is text and the tables agree with the profile; share, number and date text at their edges (never 0% or 100% for a partial share); all-midnight timestamps shown as dates; filters and periods in words; `escape_markdown` on hostile text (images, links, headings, emoji, colour directives, line breaks); CSV downloads (byte order mark, formula starts defused in cells and headings, numbers untouched, duplicate column names) | Inline; hostile strings |
| Dataset loading | Read options (header counted from 1 for the user and from 0 for the loaders); what was found, in one sentence; a file that cannot be read leaves the database alone; a failure after loading closes it; the log carries counts only | Fixtures |
| Session | A file refused leaves the old dataset; a load that lost the table forgets what used it; a bug in loading is a plain message and a logged type; replacing starts a new conversation and a new client and keeps the limit; schema-only carries over; a question over the limit is an entry with a notice, no model call and no memory; progress is passed on; idle expiry at the boundary, activity resets it, no dataset means nothing to expire; two sessions share nothing | Fake LLM, fake clock |
| Eviction | The sweeper removes only idle sessions, survives a session that fails, keeps nothing alive (weak references), starts once and stops | Fake clock |
| UI | AppTest flows with a scripted model and the real validator, executor, explainer and recommender. `test_app_gate`: password form and comparison, wrong password waits, nothing typed is logged. `test_app_flows`: first page, staged file, preview, columns, quality, answer, inspector, filters, download, clarification, refusal, failures, notes, new conversation, progress, question length. `test_app_files`: sheets, separator, header row, no header, a file that fails stays staged, replace (at once, confirm, cancel), release of the uploaded bytes. `test_app_limits`: the rate limit, idle removal (on return and by the sweeper), two sessions isolated. `test_app_privacy`: schema-only mode sends no cell value, what the conversation has already sent, the notice. `test_app_hostile`: HTML, links, images and bare addresses in a file name, column names, values, model text and the question are shown as text, and again when earlier layers are bypassed. A real-browser pass (Playwright, scripted model) checked what AppTest cannot: no request or link to the hostile address, no script dialog, no console error | AppTest |
| UI rules | From the source: no `unsafe_allow_html`, no `st.cache_data` or `st.cache_resource`, nothing written to disk, Streamlit imported only by the interface, the interface never imports the database, SQL checks or model parts, no module of the interface holds a session, table or bytes | Source scan |

## 4. Numeric correctness

For every analytical path that produces numbers (profiler, executor results used in tests, grounding), tests use fixtures whose answers are computed independently:

- small fixtures: values computed by hand and written into the test;
- larger fixtures: values computed with pandas in the test itself, never by calling the module under test.

## 5. Fake LLM

`analyst/llm/fake.py` takes a script with one item per expected call: a Pydantic instance, a dict or raw JSON string (validated against the requested model, so a misfit is `invalid_output`), an exception to raise, or a function of the call. Wrapping an item as `(stage, item)` makes the test fail if the call is for a different stage. The fake charges the call budget like the real client, records every call (`calls`, `stages`), fails on a call past the end of the script, and `assert_exhausted()` fails if scripted answers were never used. This makes budget and retry behaviour testable without the network.

## 6. CI

GitHub Actions on every push and pull request:

1. `uv sync --frozen`
2. `ruff check` and `ruff format --check`
3. `pytest -m "not live and not slow"` with coverage; fails under 80% on `analyst/` excluding `analyst/ui/`, which the AppTest flows cover instead (what the page draws cannot be measured as lines)
4. Secret scan

`slow` tests run nightly or before a milestone.

## 7. Milestone gates

| Milestone | Required |
| --- | --- |
| M1 Data foundation | All ingestion, normaliser, session store and profiler tests pass |
| M2 Safe query core | Validator: 100% adversarial blocked, 0 valid blocked; executor limits pass |
| M3 First answers | All integration paths pass; seed benchmark run with a report |
| M4 Usable app | AppTest flows pass; Fahad completes the unguided UI test; about 80 cases scored |
| M5 Alpha live | Deployment smoke test passes; held-out report produced |
| MVP complete | ADR 0013 thresholds met on the full benchmark including held-out cases |

## 8. When tests fail

Re-run only the failing tests while fixing, then the full offline suite once before proposing a checkpoint. A flaky test is fixed or reported, never retried until it passes.
