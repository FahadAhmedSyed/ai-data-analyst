# ADR 0002: One in-memory DuckDB database per browser session

- Status: Accepted
- Date: 2026-11-22

## Context

Users upload their own files and must never see each other's data. The MVP has no accounts and no persistent storage. DuckDB is fast for analytical queries, runs in-process and can load pandas dataframes directly.

## Decision

Each browser session gets its own in-memory DuckDB connection, stored in `st.session_state` and created by `analyst/database/session_db.py`. Uploaded data is loaded into it with `CREATE TABLE ... AS`, then the connection is locked down (ADR 0005). Connections are dropped when the session ends or after 60 idle minutes. User data is never stored in `st.cache_data` or `st.cache_resource`, because both are shared across sessions.

## Alternatives considered

- One shared DuckDB file with a table prefix per session: weaker isolation and file cleanup problems.
- PostgreSQL: an extra service with slower analytical queries; deferred to Phase 5 for metadata and history.

## Consequences

- Strong isolation by construction.
- A Railway restart clears all sessions; the UI shows a clean "session expired" state.
- Memory use scales with concurrent sessions, so per-session caps are required (ADR 0014).
- Persistence is a Phase 5 concern.
