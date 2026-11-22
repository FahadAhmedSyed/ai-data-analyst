# ADR 0018: Standing approval to push finished task branches

- Status: Accepted
- Date: 2026-11-22
- Amends: ADR 0016 (the "nothing committed or pushed without approval" rule)

## Context

ADR 0016 required Fahad's explicit approval before any commit or push. In practice every task already goes through a pull request that Fahad reviews and merges, so asking before each branch push only added waiting.

## Decision

Claude commits and pushes each finished task to its own feature branch, and opens a pull request, without asking first, provided the task's tests, lint and format checks pass locally. Merging into `main` stays with Fahad. Nothing is ever pushed directly to `main`, and history on shared branches is never rewritten.

## Consequences

- Tasks move from done to reviewable without a round trip.
- Fahad's review gate is the pull request merge.
