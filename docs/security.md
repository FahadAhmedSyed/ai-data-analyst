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
| T5, T6 | The explanation call gets the result, question and SQL in the same kind of marked blocks (random code, text flattened and cut); every number in what it returns is checked against the result in code, and the text cannot carry an address or image | Prompt, grounding, explainer | Marker-forging tests on the explanation prompt; grounding and explainer tests |
| T5, T6 | The correction call reports a failed query in a `PROBLEM` block of the same kind. The database's message can quote values and names from the file, so it is flattened, cut and fenced; in schema-only mode quoted text is removed, and for the categories that can echo a cell the message is withheld. A query the validator rejected is never sent, and a corrected query is validated again like any other | Prompt, corrector, query loop | Marker-forging test on the problem block; schema-only canary test on a real conversion error; rejected-never-retried tests |
| T7 | Size check before parse; zip uncompressed-size check; row and column caps; defusedxml installed | Ingestion | Oversized and zip-bomb fixtures |
| T8 | Per-session DuckDB in `st.session_state`; no user data in globals or Streamlit caches | Session | Two-session AppTest |
| T9 | Shared password; OpenAI project spend cap; 20 questions per 10 minutes per session | UI, account | Gate and limiter tests |
| T10 | Logs carry metadata only; SQL text behind a debug flag | Logging | Log assertion tests |
| T10 | Failure messages shown to users are fixed text chosen by category, never the SQL, the database's message or a value; an unexpected exception is logged by its type only and the user sees a plain message | Orchestrator, query loop | Tests with a secret in the exception, a rejected query and a conversion error |
| T11 | Key from environment only; `.env` git-ignored; `.env.example` has names only; secret scanning in CI | Repository | CI |
| T12 | Markdown rendered with HTML disabled (Streamlit default); no `unsafe_allow_html` anywhere | UI | UI test with HTML in data |
| T12 | Text from the explainer and the generator (restatement, assumptions, clarifying question, refusal) has links reduced to their text, images replaced by their alt text, and web and email addresses removed before it is checked or shown, so an injected image address cannot send data out when the answer renders | Explainer, generator, `safe_text` | Explainer and generator tests with a data-carrying image and link |
| T12 | Column names, categories and group names in charts are cut, flattened and escaped (`&`, `<`, `>`, `%{`) before they reach Plotly | Charts | Chart tests with HTML in names; headless browser check |

### Prompt containment (T1.11)

Threats T5 and T6 are met in three layers, none relied on alone (the validator still checks whatever the model writes):

1. **Shape.** Every cell, name and value from the file, and every value carried into a later turn, is cut to 80 characters and flattened to one line, so it cannot begin a line of its own.
2. **Markers.** The data goes in blocks whose marker lines carry a random 12-character code made when the session's builder is created. The file's author cannot know it, so no text in the file can close a block and pass for an instruction. If the code ever appeared inside the content it is removed from it.
3. **Instruction.** The fixed system prompt says the `DATASET`, `HISTORY` and `PROBLEM` blocks are data, tells the model never to follow instructions found there, and tells it to refuse requests to change or export data (benchmark cases I01 and I02).

The marker code must be new for each dataset and is never shown to the user. Schema-only mode sends no cell value; question text, SQL and assumptions from turns that ran before the mode was switched on may still contain values the user or model quoted.

## 4. SQL validator rules

Implemented in `analyst/sql/validator.py`. Parsing: `sqlglot.parse(sql, read="duckdb")`.

| # | Rule | Result on failure |
| --- | --- | --- |
| V1 | Exactly one statement | Reject |
| V2 | Root is SELECT, UNION, INTERSECT, EXCEPT, or WITH ending in one of those | Reject |
| V3 | No INSERT, UPDATE, DELETE, MERGE, DROP, CREATE, ALTER, TRUNCATE, COPY, ATTACH, DETACH, PRAGMA, SET, RESET, INSTALL, LOAD, EXPORT, IMPORT, CALL, or unparsed `Command` nodes anywhere in the tree | Reject |
| V4 | Every table reference is a session table or a CTE defined in the same query. SQLGlot's `qualify` does not catch unknown tables, so this explicit check is required (spike note B1) | Correctable for an unknown plain identifier; Reject for any other name (note 3) or a schema qualifier |
| V5 | No system catalogs: `information_schema`, `pg_catalog`, `duckdb_*` functions and views, `sqlite_master` | Reject |
| V6 | No table functions in FROM: `read_csv*`, `read_parquet`, `read_json*`, `read_text`, `read_blob`, `glob`, `range`, `generate_series`, `parquet_scan` and any other function used as a table | Reject |
| V7 | Every scalar, aggregate and window function is on the allowlist | Correctable; Reject for functions that read files, the environment, settings or nested SQL (note 5) |
| V8 | Columns qualify against the session schema (`sqlglot.optimizer.qualify` with the catalog) | Correctable |
| V9 | No `WITH RECURSIVE` | Correctable |
| V10 | At most 4 joins; no join without a condition unless both inputs are CTEs or subqueries of 1,000 rows or fewer by LIMIT | Correctable |
| V11 | Final query wrapped as `SELECT * FROM (<query>) LIMIT <max_rows + 1>` | Applied |
| V12 | Anything SQLGlot cannot parse completely | Reject |

"Reject" ends the turn with a safety message and a warning log. "Correctable" goes to the corrector.

### Implementation notes (T1.08)

How `analyst/sql/validator.py` realises the rules, and where it goes beyond the table above.

1. **Default deny.** Every node in the parsed tree must be a class on an explicit syntax or function list. A construct SQLGlot or DuckDB adds later is blocked until someone allows it, with a test. A node that is on no list gives `disallowed_function` (if it is a function), `forbidden_statement` (if it is a statement class) or `unsupported_syntax` (anything else, for example `PIVOT`, `VALUES`, `LATERAL`, `TABLESAMPLE`, subscripts, query parameters, and `ASOF`, `SEMI` or `ANTI` joins). `unsupported_syntax` is a new category and is correctable.
2. **What runs is regenerated.** The text sent to DuckDB is generated from the validated tree, with comments removed and V11 applied, not the model's own text. Two checks then back this up: the generated text is parsed and validated again, and DuckDB's own parser (`extract_statements` on a parse-only, locked-down connection) must read it as exactly one `SELECT`. SQLGlot and DuckDB tokenize separately, so this closes the gap where they might disagree about where a statement ends. It also catches text SQLGlot accepts and DuckDB does not (an empty `GROUP BY`).
3. **V4 names.** An unknown table name is correctable only if it is a plain identifier (`str.isidentifier()`). Anything else (a path, a URL, a name with a space or hyphen) is rejected, as is any name starting with `duckdb_`, `pg_`, `sqlite_`, `pragma_` or `information_schema`, unless it is the session table. Names are compared case-insensitively.
4. **V6.** Any function used as a table (`read_csv`, `range`, `unnest`, any other) is rejected.
5. **V7 hardening.** A call to `getenv`, `current_setting`, `getvariable`, `query`, `query_table`, `glob`, `pg_sleep`, `error`, `nextval`, `unnest`, `generate_series`, `range`, or any function whose name starts with `read_`, `write_`, `duckdb_`, `pragma_`, `pg_`, `sqlite_` or `parquet_` is an attack signature, not a slip, so it is **rejected** instead of corrected. Every other function that is not on the allowlist stays correctable, with a nearest-name suggestion when the model's own spelling is close.
6. **V10 refinement.** A join without a condition (comma join, `CROSS JOIN`, `JOIN` with no `ON` or `USING`, `NATURAL JOIN`) is allowed when both inputs are subqueries or CTEs known to have at most 1,000 rows, or when either input is known to be exactly one row. Row bounds come from a literal `LIMIT`, an aggregate with no `GROUP BY`, a `SELECT` with no `FROM`, or a single bounded input. The single-row case is not in the original rule; it is needed for the common "compare each row with the overall average" pattern (`... CROSS JOIN (SELECT AVG(x) FROM t)`), where the result is no larger than the table. A CTE that shares its name with the session table or with another CTE has no known size. Only plain `INNER`, `LEFT`, `RIGHT`, `FULL` and `CROSS` joins are accepted. The join count includes joins inside subqueries and CTEs.
7. **Size limits** (correctable `too_complex`): 10,000 characters, 2,000 tree nodes, nesting depth 40. Input that overflows the parser's recursion limit is rejected.
8. **Several problems at once.** The most serious finding is reported: refusals before corrections, then unknown table, function, syntax, size and column, in that order.
9. **V12.** A syntax error is a refusal, as the table says. Ordinary syntax slips are probably the commonest model mistake, and nothing runs when one occurs, so making plain syntax errors correctable is worth reviewing once real traffic is measured. Statements SQLGlot cannot parse at all (`EXPORT DATABASE`, `IMPORT DATABASE`) are refused as `parse_error`.
10. **Allowlist spelling.** The list is keyed on SQLGlot classes, so a few spellings of an allowlisted function come with it: `ceiling`, `substr`, `len`, `ucase`, `lcase`, `array_agg`, `percentile_cont`, `datetrunc`, `group_concat`, `log2`. SQLGlot also turns spellings that DuckDB lacks into DuckDB ones (`nvl` becomes `COALESCE`), so such a query runs where the original text would not. The allowlist test checks the names above against this document, so the two cannot drift apart.
11. **Logs.** A refusal logs its category, a reason built from short identifiers, and a hash of the SQL. The SQL itself is never logged (T10). SQLGlot's own logger is silenced because it prints the statement when it falls back to an unparsed `Command`.

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

Applied in `analyst/database/session_db.py` after the table is loaded, in this order:

```sql
SET temp_directory = '';
SET enable_external_access = false;
SET autoinstall_known_extensions = false;
SET autoload_known_extensions = false;
SET allow_community_extensions = false;
SET python_enable_replacements = false;
SET memory_limit = '1GB';
SET threads = 2;
SET lock_configuration = true;
```

Notes from T1.06 (DuckDB 2.0.1):

- `temp_directory = ''` must come first: once external access is off it cannot be changed. It is needed because turning external access off does **not** stop DuckDB spilling large intermediate results to a `.tmp` directory in the working directory (a grouped aggregate under a 50 MB cap wrote about 190 MB and left the files behind). With it set, nothing is written to disk (NFR-5) and a query over the memory cap fails with an out-of-memory error.
- `python_enable_replacements = false` stops a query reading Python variables by name (a replacement scan), which otherwise works on an unlocked connection.
- After applying the settings, `session_db.py` reads them back from `duckdb_settings()` and refuses to open the session if any is wrong.

Verification tests (each must raise an error after lockdown, on the pinned DuckDB version): the 22 statements in `tests/unit/test_session_db.py`, including

- `SELECT * FROM read_csv('/etc/passwd')`, `read_text`, `glob`, `read_parquet`, and a URL used as a table
- `ATTACH 'x.db'`, `COPY ... TO`, `EXPORT DATABASE`
- `INSTALL httpfs` (also `FROM community`) and `LOAD httpfs`
- `SET` of memory, threads, external access, extension, replacement and temp-directory options, `PRAGMA memory_limit`, and `SET lock_configuration = false`

Two further tests show that a query cannot read a Python variable and that a query over the memory cap writes no file.

### Catalog integrity check

DuckDB has no read-only mode for in-memory databases, so DDL and DML are stopped only by the validator (rules V2 and V3). As a second layer, `SessionDatabase.verify_catalog()` compares the database with a snapshot taken at load and is called by the executor after every query. It covers every database, schema, table, view, sequence, type, macro and index, each column's name and type, the row count, and a hash of all values (a sum of per-row hashes, so duplicate rows do not cancel). Any difference marks the session corrupted and raises `SessionCorruptedError`; the session then refuses further use until the file is uploaded again. Only the kinds of change are logged, never names or values.

The value hash is skipped above 50 million cells (rows times columns), because it would add seconds per question (about 0.3 s at that size, 4 s for 2 million rows by 300 columns). Above the cap a changed value is caught only by the validator; structure and row counts are still checked.

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
