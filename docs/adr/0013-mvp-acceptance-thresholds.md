# ADR 0013: MVP acceptance thresholds

- Status: Accepted; cost threshold set after spike T0.4 (c)
- Date: 2026-11-22
- Covers decision: D8

## Context

The MVP is not complete just because the app launches. It must show reliable end-to-end analysis on a benchmark of about 150 questions, including a 30% held-out set.

## Decision

| Metric | Threshold |
| --- | --- |
| Adversarial SQL blocked | 100% |
| Gold SQL wrongly blocked by the validator | 0 |
| Execution success, non-ambiguous cases | 95% or more |
| Result accuracy, all non-ambiguous cases | 85% or more |
| Result accuracy, simple aggregation and filter cases | 95% or more |
| Clarification recall on ambiguous cases | 80% or more |
| Clarification requested on clear cases | 10% or less |
| Multi-turn sequences fully correct | 80% or more |
| Answers shown with an ungrounded number | 0 |
| Profile values matching pandas | 100% |
| Question latency, p95 | 15 s or less |
| Median cost per question | Set after spike T0.4 (c); target under US$0.02 |

## Consequences

- MVP complete (end of Phase 2) means all thresholds met on the full benchmark, including the held-out set.
- Phase 3 does not start until then.
- Thresholds change only by a new ADR, never to make a run pass.
