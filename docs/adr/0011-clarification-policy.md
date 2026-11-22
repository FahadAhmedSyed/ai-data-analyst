# ADR 0011: When to ask for clarification

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D6

## Context

Asking too often annoys users; assuming too often gives confident wrong answers.

## Decision

Ask a clarifying question only when two or more plausible readings would produce different results and no configured default (ADR 0009) or earlier choice in the conversation settles it. Otherwise, proceed with the most likely reading and state the assumption in the answer. Assumptions the user confirms carry forward for the rest of the session.

## Alternatives considered

- Always ask on any ambiguous term: safest, but slow and frustrating.
- Never ask: fastest, but silently wrong on genuinely ambiguous questions.

## Consequences

- Measured in the benchmark: clarification recall of 80% or more on ambiguous cases, and clarification requested on 10% or fewer clear cases (ADR 0013).
