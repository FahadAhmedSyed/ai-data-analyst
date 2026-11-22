# ADR 0007: Pandas as the only dataframe library

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D2

## Context

The spec lists "Pandas / Polars". Streamlit, Plotly and the Excel readers all work natively with pandas. DuckDB does the heavy analytical work.

## Decision

Use pandas for ingestion and as the interchange format between DuckDB, Streamlit and Plotly. Do not add Polars in the MVP.

## Alternatives considered

- Polars for ingestion: faster on very large files, but adds a second API and conversion code with no MVP benefit given the 50 MB upload limit.

## Consequences

- One dataframe API to learn, test and maintain.
- Polars can be reconsidered if profiling shows pandas is the bottleneck.
