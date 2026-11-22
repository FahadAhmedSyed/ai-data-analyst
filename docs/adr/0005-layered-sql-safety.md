# ADR 0005: Layered SQL safety: AST allowlist plus a locked-down DuckDB connection

- Status: Accepted
- Date: 2026-11-22

## Context

The LLM writes SQL that runs against user data. A hallucinated or manipulated query could try to modify data, read local files, attach databases, install extensions or exhaust resources. Keyword filtering is easy to bypass.

## Decision

Two independent layers, so a failure of either alone cannot cause harm:

1. Validator (`analyst/sql/validator.py`): parse with `sqlglot.parse(sql, read="duckdb")`; require exactly one statement whose root is a SELECT, UNION or WITH ending in SELECT; reject any DML, DDL, COPY, ATTACH, PRAGMA, SET, INSTALL, LOAD or unparsed command anywhere in the tree; allow only session tables and query-defined CTEs; reject table functions in FROM (`read_csv`, `read_parquet`, `glob`, `range` and similar) and system catalogs; allow only allowlisted functions; qualify columns against the session schema; reject recursive CTEs, unconditioned joins on large tables and more than 4 joins; wrap the query as `SELECT * FROM (...) LIMIT cap + 1`. Anything SQLGlot cannot fully parse is rejected (fail closed).
2. Connection lockdown (`analyst/database/session_db.py`), applied after data load:
   `enable_external_access = false`, `autoinstall_known_extensions = false`, `autoload_known_extensions = false`, `memory_limit`, `threads = 2`, then `lock_configuration = true`.
   The executor enforces a timeout with `connection.interrupt()` from a timer thread.

Destructive or file-reading queries end the turn with a safety message and are not sent to the corrector.

## Alternatives considered

- Keyword blocklist: bypassable with comments, casing and functions.
- Read-only database file: DuckDB in-memory databases cannot be opened read-only; lockdown achieves the same effect.

## Consequences

- An adversarial test suite of 60+ queries must be 100% blocked, with zero valid queries blocked.
- Every lockdown setting is verified against the pinned DuckDB version (spike T0.4 (a), task T1.06).
- The function allowlist grows deliberately, through tests.
