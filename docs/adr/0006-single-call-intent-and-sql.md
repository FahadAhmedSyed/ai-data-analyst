# ADR 0006: One structured LLM call returns intent, then SQL

- Status: Accepted, pending confirmation by spike T0.4 (c)
- Date: 2026-11-22
- Covers decision: D3

## Context

The spec describes question understanding, intent representation and SQL generation as stages A to C. Each separate LLM call adds roughly 1 to 4 seconds and its own cost.

## Decision

A single structured call returns, in field order: resolved question, ambiguities, `needs_clarification`, clarification question, intent (operation, measures, dimensions, filters, time range, sort, output), assumptions, SQL, explanation and chart suggestion. Field order makes the model say what is unclear, and whether to ask, before it writes a reading of the question down, and state the intent before writing SQL. Code then checks every intent column exists. If clarification is needed, no SQL runs.

## Alternatives considered

- Two calls (intent, then SQL): easier to inspect, about 1.5x the cost and latency. Benchmarked against the single call in T0.4 (c) and again in Phase 2.

## Consequences

- Lower latency and cost per question.
- If the spike shows the two-call design is clearly more accurate, this ADR is superseded before T1.12.
- T1.12 was written ahead of the spike, against the scripted fake. `Generator` returns the same things (kind, SQL, message, unknown columns) whether it makes one call or two, so a change to two calls would rework its inside and not its callers.

## Amendment, 2026-11-23: ambiguities and the decision to ask come before the intent

The first model run (`eval/reports/2026-11-23-4711510-dev.md`) asked for clarification in 0 of 5 ambiguous dev cases. The model listed no ambiguities at all: by the time it reached that field it had already written a reading into the intent (a filter value, a date range) and recorded the choice as an assumption. The fields `ambiguities`, `needs_clarification` and `clarification_question` were therefore moved ahead of `intent`, and the prompt was reworded to match (ADR 0011 is unchanged). Intent still comes before SQL. Measured on the dev cases, repeating each question: clarification was asked for 8 of 15 samples with the reworded prompt alone, and for 20 of 25 once the fields were also moved. In the same checks, clear questions were asked about 1 time in 92.
