# ADR 0004: OpenAI behind a single thin client; model chosen by measurement

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D12

## Context

Several stages call an LLM (intent and SQL, correction, explanation). Each needs structured output, retries, timeouts and cost tracking. Model prices and capabilities change often.

## Decision

All model calls go through `analyst/llm/client.py`, which uses OpenAI structured outputs parsed into Pydantic models, a 30 s timeout, two retries with backoff on rate limits and server errors, and logs model, tokens, cost and latency per call. The model is configurable per stage. The initial model is chosen in spike T0.4 (c) on benchmark accuracy, latency and cost. Fahad sets a monthly spend cap on the OpenAI project. `analyst/llm/fake.py` replays scripted outputs for tests.

## Alternatives considered

- Calling the OpenAI SDK directly from each stage: duplicated retry and cost code, harder provider switch.
- Fixing a model now: risks paying more, or getting less accuracy, than necessary.

## Consequences

- One place to change provider, model or pricing.
- All pipeline tests run without network access.
- Cost per question is measurable from day one.
