# Phase 0 spike results (T0.4)

- Date: 2026-11-22
- Environment: Python 3.12.3, DuckDB 2.0.1, SQLGlot 30.33.0, pandas 3.1.1, python-calamine (latest), Streamlit 1.67.0, Linux x86-64 cloud container
- Spike scripts were throwaway and are not part of the repository.

## Summary

| Spike | Question | Result |
| --- | --- | --- |
| (a) | Does the DuckDB lockdown block file, network, extension and settings access, and can a query be interrupted? | **Pass.** All 12 access attempts blocked; `interrupt()` stops a query at 1.00 s; memory cap raises a clean error |
| (b) | Can SQLGlot detect every unsafe construct the validator needs? | **Pass, with two design notes** (unknown tables and function names) |
| (c) | Which model, one call or two, at what cost? | **Waiting for the OpenAI API key** |
| (d) | Ingestion speed and memory per session | **Pass.** 51.5 MB CSV parsed and typed in 2.5 s; 1 M rows use about 70 MiB in DuckDB; calamine 7x faster than OpenPyXL |

No finding changes an accepted ADR. Two refinements to the validator design and one to memory sizing follow.

## (a) DuckDB lockdown

After loading a table and applying the six lockdown settings from [security.md](security.md#5-duckdb-lockdown):

| Attempt | Outcome |
| --- | --- |
| `read_csv('/etc/passwd')` | Blocked: PermissionException, file system operations disabled |
| `read_text('/etc/passwd')` | Blocked: same |
| `glob('/*')` | Blocked: same |
| `ATTACH '/tmp/x.db'` | Blocked: same |
| `COPY (SELECT 1) TO '/tmp/out.csv'` | Blocked: same |
| `SELECT * FROM 'https://example.com/x.csv'` | Blocked: same |
| `INSTALL httpfs` | Blocked: cannot access extension directory |
| `LOAD httpfs` | Blocked: loading external extensions disabled |
| `SET memory_limit = '10GB'` | Blocked: configuration locked |
| `PRAGMA memory_limit = '10GB'` | Blocked: configuration locked |
| `SET enable_external_access = true` | Blocked: configuration locked |
| `SET lock_configuration = false` | Blocked: configuration locked |
| `DROP TABLE rainfall` | **Not blocked by DuckDB** (expected) |

- Timeout: a `threading.Timer` calling `connection.interrupt()` stopped a `range(1e11)` aggregate after exactly 1.00 s with `InterruptException`.
- Memory: with `memory_limit = '100MB'`, an oversized query raised `OutOfMemoryException` without crashing the process.

**Design note A1.** DuckDB has no read-only mode for in-memory databases, so DML and DDL are stopped only by the validator (rules V2 and V3). Impact is limited to the user's own session. Added safeguard for T1.06: after each query the executor checks that the catalog (table names and row count) is unchanged; if not, the session is marked corrupted, an error is logged, and the user is asked to reload the file.

## (b) SQLGlot validation signals

Twenty test queries were parsed with `sqlglot.parse(sql, read="duckdb")`.

| Need | How SQLGlot exposes it | Works |
| --- | --- | --- |
| Stacked statements, including `SELECT 1 /* ; */ ; -- DELETE ...` | `len(parse(...)) > 1` | Yes |
| DDL and DML (`DROP`, `ATTACH`, `COPY`, `PRAGMA`, `INSTALL`, `SET`) | Root node type (`Drop`, `Attach`, `Copy`, `Pragma`, `Install`, `Set`) | Yes |
| Table functions (`read_csv`, `range`, `duckdb_settings()`) | `exp.Table` whose `this` is a function node | Yes |
| Recursive CTEs | `exp.With` with `recursive = True` | Yes |
| Unknown functions (`getenv`) | `exp.Anonymous` nodes | Yes |
| Unknown columns | `qualify(..., schema, validate_qualify_columns=True)` raises | Yes |
| Unknown tables (`weather`) | **`qualify` does not raise** | Needs explicit check |
| File path as table (`FROM '/etc/passwd'`) | Parsed as a table named `/etc/passwd` | Caught by the table allowlist |
| System catalogs (`information_schema.tables`) | Table with a `db` qualifier | Caught by the table allowlist |

**Design note B1.** Rule V4 (table allowlist) is essential, not just a convenience: it is the only check that catches unknown tables, file-path tables and system catalogs. Any table name containing `/`, `.`, `:` or a schema qualifier is a Reject, not Correctable.

**Design note B2.** SQLGlot normalises function names into its own expression classes: `AVG` becomes `Avg`, `quantile_cont` becomes `PercentileCont`, `date_trunc` becomes `TimestampTrunc`, `strftime` becomes `TimeToStr`. The function allowlist in T1.08 is therefore keyed on SQLGlot expression classes, with `Anonymous` functions checked by name against a short list. Each allowlisted function gets a round-trip test (DuckDB SQL in, same semantics out).

## (c) Model comparison

Not run yet. Needs `OPENAI_API_KEY` in the cloud environment. Plan when available:

- 20 questions from the seed benchmark across categories, on the rainfall and sales datasets.
- Two candidate current OpenAI models, chosen from OpenAI's model list at run time.
- Single-call (ADR 0006) versus two-call design.
- Measures: result accuracy, schema linking, clarification behaviour, p50 and p95 latency, tokens and cost per question.
- Output: model choice, confirmation or supersession of ADR 0006, and the cost threshold for ADR 0013.

## (d) Ingestion speed and memory

Test file: 1,000,000 rows, 7 columns (day-first timestamps, site, rainfall, temperature, flow, quality flag, notes), 51.5 MB as CSV.

| Step | Time | Memory |
| --- | --- | --- |
| Delimiter sniff and parse as text (pandas, C engine) | 0.7 s | |
| Type conversion (day-first timestamps, 3 numeric columns) | 1.8 s | +131 MB process RSS at peak (text plus typed frames) |
| Load into DuckDB | 0.2 s | |
| DuckDB storage for the table | | 70 MiB |
| Process RSS after dropping the dataframes | | +137 MB over baseline |
| `SUMMARIZE` over 1 M rows | 0.25 s | |
| Typical seasonal `GROUP BY` | 8 ms | |

Excel, 200,000 rows by 7 columns (7.1 MB file, 70 MB uncompressed):

| Engine | Read time |
| --- | --- |
| calamine | 2.0 s |
| OpenPyXL | 13.7 s |

Findings:

- NFR-6 (50 MB in under 20 s, 1 M-row profile in under 5 s) is met with a wide margin for CSV.
- A 50 MB XLSX holds roughly 1.4 M rows of this shape and would take about 14 s with calamine. That is within target but slower than CSV; the UI shows progress during Excel parsing.
- XLSX compresses about 10 to 1, so the zip-bomb check limit of 10 times the upload size (500 MB uncompressed) is correct and leaves no headroom to tighten.
- ADR 0008 (calamine) confirmed.

**Memory sizing for Railway.** A session with a maximum-size dataset costs about 150 MB steady and about 300 MB at peak during parsing. The 1 GB DuckDB cap is a ceiling for query working memory, not a reservation. Recommendation: start with a 2 GB instance, which supports about 8 concurrent large sessions or many more small ones, and set the instance memory alert at 75%. Streamlit's own baseline is measured in T1.20.

## Version pins for T1.01

| Package | Version tested |
| --- | --- |
| duckdb | 2.0.1 |
| sqlglot | 30.33.0 |
| pandas | 3.1.1 |
| streamlit | 1.67.0 |

pandas 3 uses its string dtype by default for `dtype=str`; the normaliser in T1.05 is written against that behaviour.
