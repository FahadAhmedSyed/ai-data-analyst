# ADR 0012: Protecting the deployed alpha

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D7

## Context

The MVP has no accounts but deploys to a public Railway URL. Anyone with the link could upload data and spend the OpenAI budget.

## Decision

- A shared password (`APP_PASSWORD` environment variable) gates the app.
- The OpenAI API key belongs to a dedicated OpenAI project with a monthly spend cap.
- Each session is limited to 20 questions per 10 minutes.

## Alternatives considered

- Run locally only until Phase 5: safest, but nobody else can try the MVP.
- Full authentication now: Phase 5 work brought forward.

## Consequences

- Low-effort protection adequate for a small group of testers.
- Proper accounts replace the password in Phase 5.
