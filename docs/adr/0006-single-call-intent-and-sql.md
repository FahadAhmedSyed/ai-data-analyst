# ADR 0006: One structured LLM call returns intent, then SQL

- Status: Accepted, pending confirmation by spike T0.4 (c)
- Date: 2026-11-22
- Covers decision: D3

## Context

The spec describes question understanding, intent representation and SQL generation as stages A to C. Each separate LLM call adds roughly 1 to 4 seconds and its own cost.

## Decision

A single structured call returns, in field order: resolved question, intent (operation, measures, dimensions, filters, time range, sort, output), assumptions, ambiguities, `needs_clarification`, clarification question, SQL, explanation and chart suggestion. Field order makes the model state the intent before writing SQL. Code then checks every intent column exists. If clarification is needed, no SQL runs.

## Alternatives considered

- Two calls (intent, then SQL): easier to inspect, about 1.5x the cost and latency. Benchmarked against the single call in T0.4 (c) and again in Phase 2.

## Consequences

- Lower latency and cost per question.
- If the spike shows the two-call design is clearly more accurate, this ADR is superseded before T1.12.
- T1.12 was written ahead of the spike, against the scripted fake. `Generator` returns the same things (kind, SQL, message, unknown columns) whether it makes one call or two, so a change to two calls would rework its inside and not its callers.
