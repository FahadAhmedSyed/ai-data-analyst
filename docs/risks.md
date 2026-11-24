# Risk register

- Updated: 2026-11-24 (T1.21 review)
- Review: at every milestone

Sorted by severity. Likelihood and impact: High, Medium, Low.

| ID | Risk | Likelihood | Impact | Mitigation | Owner task | Status |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Valid SQL that answers the wrong question | High | High | Benchmark on result accuracy; resolved question and assumptions shown; clarification policy (ADR 0011) | T1.12, T1.19 | Open |
| R5 | Day and month swapped in dates | High | High | NZ day-first default; ambiguity flag; user confirmation (ADR 0009) | T1.05 | Open |
| R7 | Public URL drains the OpenAI budget | High | Medium | Password, spend cap, rate limit (ADR 0012). The rate limit is per browser session, so the spend cap is the real limit (security.md, known limits). The password is closed-by-default in a deployment (ADR 0019) | T1.18, T1.20, T1.21 | Open |
| R2 | Explanation misstates a number | Medium | High | Grounding check with template fallback (ADR 0017) | T1.14 | Open |
| R8 | Data leaks between sessions through Streamlit caches | Medium | High | No user data in shared caches; two-session test; PR checklist | T1.06, T1.18 | Open |
| R10 | Users unwilling to send data to OpenAI | Medium | High | Notice, schema-only mode, documented terms (privacy.md) | T0.3, T1.18 | Open |
| R15 | Censored environmental values mishandled | Medium | High | Detect and flag; never coerce silently; benchmark case A03 | T1.05 | Open |
| R6 | Railway runs out of memory with several large sessions | Medium | High | Caps (ADR 0014), idle eviction. Spike: about 150 MB steady and 300 MB peak per maximum-size session; start with a 2 GB instance. Measured in the container (T1.20): about 140 MB for the server and libraries, about 80 MB more for a 120,000-row CSV. Hobby has no memory alert (Pro only): watch the Metrics graph | T1.06, T1.20 | Reduced by spike and measurement |
| R20 | Memory used up before sign-in through Streamlit's upload route, which does not check the password (security T13) | Medium | Medium | A ceiling of 8 maximum-size uploads held over all sessions, refused with 413 above it; restart policy brings the app back | T1.21 | Reduced |
| R17 | DML or DDL reaches DuckDB if the validator has a bug (DuckDB has no read-only in-memory mode) | Low | Medium | Validator rules V2 and V3; catalog integrity check after each query; damage limited to the user's own session | T1.06, T1.08 | New from spike |
| R3 | File, network or settings access through SQL | Low | High | Validator plus lockdown; spike confirmed all 12 access attempts blocked on DuckDB 2.0.1 | T1.06, T1.08 | Reduced by spike |
| R4 | Instructions hidden in uploaded data | Medium | Medium | Untrusted data block; output limited to validated SQL; cases I01 and I02 | T1.11 | Open |
| R9 | OpenAI price, latency or API changes | Medium | Medium | Thin client, configurable models, cost logging | T1.10 | Open |
| R11 | Too many or too few clarification questions | Medium | Medium | Benchmark category with thresholds | T1.19 | Open |
| R12 | Prompts overfit to the benchmark | Medium | Medium | 32% held out; prompts never use benchmark data | T1.19 | Open |
| R14 | Wide tables exceed the prompt budget | Medium | Medium | Card budget now; column selection in Phase 2 | T1.07 | Open |
| R16 | Advanced features started before the MVP is reliable | High | Medium | Phase 3 gated on MVP thresholds | Milestones | Open |
| R18 | Model comparison spike delayed, blocking the generator design | Medium | Medium | T1.01 to T1.11 do not depend on it; spike runs as soon as the key is available | T0.4 (c) | New |
| R13 | Streamlit limits at production scale | Medium | Medium | Pipeline independent of UI; revisit at Phase 5 | Architecture | Open |
| R19 | Excel parsing is slow near the 50 MB limit (estimated 14 s) | Medium | Low | Progress indicator; calamine engine | T1.04, T1.18 | New from spike |
