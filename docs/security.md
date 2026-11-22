# Security

- Version: 1.0 (MVP)
- Date: 2026-11-22
- Related: [ADR 0005](adr/0005-layered-sql-safety.md), [ADR 0012](adr/0012-alpha-access-control.md), [privacy](privacy.md)

## 1. Principles

1. LLM output is untrusted. It is parsed and checked by code before anything acts on it.
2. Uploaded data is untrusted. Its contents never become instructions.
3. Two independent layers protect every query, so one failing alone causes no harm.
4. Fail closed: anything that cannot be fully checked is rejected.
5. Nothing sensitive in logs or Git.

## 2. Threat model

| ID | Threat | Source | Impact |
| --- | --- | --- | --- |
| T1 | Destructive SQL (DROP, DELETE, UPDATE) | Hallucination or manipulation | Session data corrupted |
| T2 | File or network access through SQL (`read_csv('/etc/passwd')`, `ATTACH`, `COPY TO`, `httpfs`) | Same | Server files or secrets exposed |
| T3 | Configuration change through SQL (`SET`, `PRAGMA`, `INSTALL`, `LOAD`) | Same | Lockdown removed |
| T4 | Resource exhaustion (huge joins, recursive CTEs, `range(1e12)`) | Same | Process crash for all users |
| T5 | Prompt injection in the question | User | Model steered to unsafe SQL |
| T6 | Prompt injection in data (cell values, headers, sheet names) | File author | Same as T5 |
| T7 | Malicious uploads (zip bombs, XML attacks, huge files) | User | Crash or memory exhaustion |
| T8 | Cross-session data access | Bug | Confidential data exposed |
| T9 | Budget abuse on the public URL | Anyone with the link | OpenAI cost |
| T10 | Sensitive data in logs | Bug | Data exposed to anyone with log access |
| T11 | Secrets committed to Git | Mistake | Key compromise |
| T12 | HTML or script in rendered answers | Data or model | Phishing or script execution in the browser |

## 3. Controls

| Threat | Control | Layer | Test |
| --- | --- | --- | --- |
| T1 to T3 | SQL validator rules (section 4) | Validator | Adversarial suite |
| T1 to T3 | DuckDB lockdown (section 5) | Database | Lockdown tests |
| T4 | Cost rules in validator; timeout via `interrupt()`; memory cap; row cap | Validator, executor | Executor limit tests |
| T5, T6 | Model output is only SQL plus text, both checked; data placed in a marked untrusted block, cut to 80 characters; identifiers normalised; model has no tools or secrets | Prompt, validator | Injection fixtures and benchmark cases |
| T7 | Size check before parse; zip uncompressed-size check; row and column caps; defusedxml installed | Ingestion | Oversized and zip-bomb fixtures |
| T8 | Per-session DuckDB in `st.session_state`; no user data in globals or Streamlit caches | Session | Two-session AppTest |
| T9 | Shared password; OpenAI project spend cap; 20 questions per 10 minutes per session | UI, account | Gate and limiter tests |
| T10 | Logs carry metadata only; SQL text behind a debug flag | Logging | Log assertion tests |
| T11 | Key from environment only; `.env` git-ignored; `.env.example` has names only; secret scanning in CI | Repository | CI |
| T12 | Markdown rendered with HTML disabled (Streamlit default); no `unsafe_allow_html` anywhere | UI | UI test with HTML in data |

## 4. SQL validator rules

Implemented in `analyst/sql/validator.py`. Parsing: `sqlglot.parse(sql, read="duckdb")`.

| # | Rule | Result on failure |
| --- | --- | --- |
| V1 | Exactly one statement | Reject |
| V2 | Root is SELECT, UNION, INTERSECT, EXCEPT, or WITH ending in one of those | Reject |
| V3 | No INSERT, UPDATE, DELETE, MERGE, DROP, CREATE, ALTER, TRUNCATE, COPY, ATTACH, DETACH, PRAGMA, SET, RESET, INSTALL, LOAD, EXPORT, IMPORT, CALL, or unparsed `Command` nodes anywhere in the tree | Reject |
| V4 | Every table reference is a session table or a CTE defined in the same query. SQLGlot's `qualify` does not catch unknown tables, so this explicit check is required (spike note B1) | Correctable for an unknown plain name; Reject for any name containing `/`, `.`, `:` or a schema qualifier |
| V5 | No system catalogs: `information_schema`, `pg_catalog`, `duckdb_*` functions and views, `sqlite_master` | Reject |
| V6 | No table functions in FROM: `read_csv*`, `read_parquet`, `read_json*`, `read_text`, `read_blob`, `glob`, `range`, `generate_series`, `parquet_scan` and any other function used as a table | Reject |
| V7 | Every scalar, aggregate and window function is on the allowlist | Correctable |
| V8 | Columns qualify against the session schema (`sqlglot.optimizer.qualify` with the catalog) | Correctable |
| V9 | No `WITH RECURSIVE` | Correctable |
| V10 | At most 4 joins; no join without a condition unless both inputs are CTEs or subqueries of 1,000 rows or fewer by LIMIT | Correctable |
| V11 | Final query wrapped as `SELECT * FROM (<query>) LIMIT <max_rows + 1>` | Applied |
| V12 | Anything SQLGlot cannot parse completely | Reject |

"Reject" ends the turn with a safety message and a warning log. "Correctable" goes to the corrector.

### Function allowlist (initial)

Grown deliberately, one test per addition.

- Aggregates: `count`, `sum`, `avg`, `mean`, `min`, `max`, `median`, `mode`, `stddev`, `stddev_samp`, `stddev_pop`, `variance`, `var_samp`, `var_pop`, `quantile_cont`, `quantile_disc`, `approx_count_distinct`, `count_if`, `bool_and`, `bool_or`, `first`, `last`, `arg_min`, `arg_max`, `corr`, `covar_pop`, `covar_samp`, `string_agg`, `list` (result display only)
- Window: `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`, `lag`, `lead`, `first_value`, `last_value`, `nth_value`
- Math: `abs`, `round`, `floor`, `ceil`, `sqrt`, `power`, `ln`, `log`, `log10`, `exp`, `sign`, `greatest`, `least`
- Conditional and null: `coalesce`, `nullif`, `ifnull`, `if`, `case` (syntax)
- Casting: `cast`, `try_cast`
- Text: `lower`, `upper`, `trim`, `ltrim`, `rtrim`, `length`, `substring`, `replace`, `concat`, `starts_with`, `ends_with`, `contains`, `regexp_matches`, `split_part`, `left`, `right`
- Date and time: `date_trunc`, `date_part`, `extract`, `year`, `quarter`, `month`, `week`, `day`, `dayofweek`, `dayofyear`, `hour`, `strftime`, `strptime`, `make_date`, `date_diff`, `date_add`, `date_sub`, `age`, `current_date` (for display only; relative dates use the dataset anchor), `epoch`, `last_day`

## 5. DuckDB lockdown

Applied in `analyst/database/session_db.py` after the table is loaded:

```sql
SET enable_external_access = false;
SET autoinstall_known_extensions = false;
SET autoload_known_extensions = false;
SET memory_limit = '1GB';
SET threads = 2;
SET lock_configuration = true;
```

Verification tests (each must raise an error after lockdown, on the pinned DuckDB version):

- `SELECT * FROM read_csv('/etc/passwd')`
- `ATTACH 'x.db'`
- `COPY (SELECT 1) TO 'out.csv'`
- `INSTALL httpfs`
- `LOAD httpfs`
- `SET memory_limit = '10GB'`
- `SET enable_external_access = true`

These are confirmed in spike T0.4 (a) before T1.06 builds on them.

## 6. Upload safety

| Check | When | Limit |
| --- | --- | --- |
| Extension allowlist (`.csv`, `.txt`, `.xlsx`, `.xls`) | Before reading | Reject others |
| File size | Before parsing | 50 MB |
| XLSX total uncompressed size | Before parsing | 10 times the upload limit |
| Rows and columns | During parsing | 2,000,000 rows, 300 columns |
| Cell text in prompts | Context building | 80 characters |

## 7. Secrets

| Secret | Where it lives | Never |
| --- | --- | --- |
| `OPENAI_API_KEY` | Railway variables; local `.env`; cloud environment settings | In Git, logs, chat or prompts |
| `APP_PASSWORD` | Railway variables; local `.env` | In Git or logs |

## 8. Review checklist for every pull request

- [ ] No new SQL execution path bypasses the validator and executor
- [ ] No user data in globals, `st.cache_data` or `st.cache_resource`
- [ ] No dataset values or answer text in logs
- [ ] No `unsafe_allow_html`
- [ ] No secrets or real datasets in the diff
- [ ] New allowlisted functions have a test
