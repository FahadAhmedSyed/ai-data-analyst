# ADR 0003: Plain Python orchestration, no LangChain or LangGraph in the MVP

- Status: Accepted
- Date: 2026-11-22

## Context

The MVP pipeline is a fixed sequence: build context, generate intent and SQL, validate, execute, correct with a bounded retry, explain, chart. It has no branching between agents or long-running state.

## Decision

Implement orchestration as ordinary Python functions in `analyst/pipeline/analyst.py`, with explicit budgets for LLM calls and retries. No LangChain or LangGraph dependency in the MVP.

## Alternatives considered

- LangChain: adds abstraction layers and frequent API churn for no MVP benefit.
- LangGraph now: useful for stateful multi-agent workflows, which the MVP does not have.

## Consequences

- Fewer dependencies; every step is readable and testable.
- LangGraph is evaluated at Phase 4 entry in its own ADR, judged on actual workflow needs (conditional routing, multi-agent coordination, resumable state).
