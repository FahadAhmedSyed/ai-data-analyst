# ADR 0016: Repository and change workflow

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D11

## Decision

- A new GitHub repository, `ai-data-analyst`, created by Fahad.
- Work happens on feature branches, one per task or small group of tasks, and reaches `main` through pull requests that Fahad reviews.
- Nothing is committed or pushed without Fahad's explicit approval.
- Secrets never enter the repository; `.env` is git-ignored and `.env.example` lists variable names only.

## Consequences

- Every change has a reviewable checkpoint.
- Phase 0 documents are drafted in the project's shared files and move into the repository's `docs/` folder in the first pull request.
