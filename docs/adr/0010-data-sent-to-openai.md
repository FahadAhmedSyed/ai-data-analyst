# ADR 0010: What dataset content is sent to OpenAI

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D4

## Context

Accurate SQL needs real values: a filter on `site = 'Site A'` fails if the model guesses `'site a'`. But datasets can be confidential, and sending all rows is neither needed nor affordable.

## Decision

By default each prompt contains: the schema card (names, types, null rates, distinct counts, min and max, date ranges), up to 20 most frequent values of text columns with 50 or fewer distinct values, 3 sample rows, and the question. Results sent for explanation are capped at 50 rows, or 20 rows plus summary statistics. A schema-only mode removes sample rows and frequent values. The UI shows a short notice on upload. Cell text is cut to 80 characters and marked as untrusted data. OpenAI's current API data-use terms are confirmed in T0.3 and summarised in `docs/privacy.md`.

## Alternatives considered

- Schema-only by default: more private, lower accuracy on filters.
- Sending full data: costly, slow, and unnecessary.

## Consequences

- Good filter accuracy by default with a clear opt-out.
- The benchmark measures accuracy in both modes.
