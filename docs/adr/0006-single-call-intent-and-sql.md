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

The first model run (`eval/reports/2026-11-23-4711510-dev.md`) asked for clarification in 0 of 5 ambiguous dev cases. The model listed no ambiguities at all: by the time it reached that field it had already written a reading into the intent (a filter value, a date range) and recorded the choice as an assumption. The fields `ambiguities`, `needs_clarification` and `clarification_question` were therefore moved ahead of `intent`, and the prompt was reworded to match (ADR 0011 is unchanged). Intent still comes before SQL. Measured on the dev split in three full runs each (`eval/reports/2026-11-23-5a404bf-dev.md` is the first of the new ones): with the earlier order and wording the model asked in 0 of 5 ambiguous cases every time. With the new wording and order it asked in 4 of 5 every time, and asked about none of the 48 clear questions. The other cases did not change beyond run-to-run variation (right in 37 to 39 of 50 before, 38 to 40 after). While developing it, repeating each question showed the new wording alone asked in 8 of 15 samples, and the new order raised that to 20 of 25. The case that is still missed is A02 ("What is the average score?"), which the model never asks about. The held-out ambiguous cases (A03, A05, A08) have not been run, and the prompt's example was chosen not to resemble them.
