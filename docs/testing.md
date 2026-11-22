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
| Context builder | Prompt snapshots; injection text stays inside the untrusted block; schema-only mode omits values | Fixtures |
| Generator | Each structured output shape; intent column check; clarification gate | Fake LLM |
| Corrector | Each category; budget exhaustion; Reject never retried | Fake LLM |
| Grounding | Number formats (1,234.5; 12%; -3.2; 1e3); rounding; numbers from the question and assumptions allowed; invented number caught | Inline |
| Charts | Result shape matrix to chart type | Inline |
| Memory | Window of 3; reset; replace; carried assumptions | Inline |
| Orchestrator | Every path in the question flow diagram, including API failure | Fake LLM |
| UI | Password gate; upload then ask; clarification; failure message; reset; replace; rate limit; two-session isolation; HTML in data is not rendered | AppTest |

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
3. `pytest -m "not live and not slow"` with coverage; fails under 80% on `analyst/` excluding `analyst/ui/`
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
