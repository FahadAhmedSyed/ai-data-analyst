# ADR 0014: Resource limits

- Status: Accepted; Railway memory sizing confirmed in spike T0.4 (d)
- Date: 2026-11-22
- Covers decision: D9

## Decision

All limits are settings in `analyst/config/settings.py`, overridable by environment variables.

| Limit | Default |
| --- | --- |
| Upload size | 50 MB |
| Rows per dataset | 2,000,000 |
| Columns per dataset | 300 (column selection for prompts above about 150 arrives in Phase 2) |
| Query timeout | 10 s |
| Rows returned per query | 10,000 |
| DuckDB memory per session | 1 GB |
| DuckDB threads per session | 2 |
| Correction attempts per question | 2 |
| LLM calls per question | 5 |
| Idle session eviction | 60 minutes |
| Questions per session | 20 per 10 minutes |

## Consequences

- Predictable memory and cost per session.
- Users hitting a limit get a specific message naming the limit.
