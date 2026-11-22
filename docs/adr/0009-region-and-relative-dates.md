# ADR 0009: New Zealand region defaults; relative dates anchored to the dataset

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D5

## Context

Dates such as 03/04/2025 mean 3 April in New Zealand and 4 March in the US. Seasons differ by hemisphere. Terms such as "last year" or "recent" need an anchor, and historical datasets often end long before today.

## Decision

- Default region: New Zealand. Dates parse day-first; summer is December to February, autumn March to May, winter June to August, spring September to November.
- Region rules live in `analyst/config/domain.py` and are configurable.
- Date columns where day and month cannot be told apart are flagged and shown to the user for confirmation.
- Relative terms ("last year", "last 12 months", "recent") are anchored to the latest date in the dataset, and the answer states the exact range used.

## Alternatives considered

- Anchor to today's date: matches everyday speech but often returns empty results on historical data.
- US month-first default: wrong for the primary user base.

## Consequences

- Correct defaults for NZ users; others change one setting.
- Every relative-date answer states its range, so a wrong anchor is visible.
