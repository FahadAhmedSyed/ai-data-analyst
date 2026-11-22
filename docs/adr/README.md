# Architecture Decision Records

0001 to 0017 were accepted on 2026-11-22 when the development plan was approved.

| ADR | Decision | Plan ref |
| --- | --- | --- |
| [0001](0001-python-streamlit-monolith.md) | Python monolith, Streamlit UI, UI-independent pipeline | Stack |
| [0002](0002-per-session-duckdb.md) | One in-memory DuckDB per browser session | Stack |
| [0003](0003-plain-python-orchestration.md) | Plain Python orchestration; no LangChain or LangGraph in the MVP | Stack |
| [0004](0004-openai-thin-client.md) | OpenAI behind one thin client; model chosen by measurement | D12 |
| [0005](0005-layered-sql-safety.md) | AST allowlist plus locked-down DuckDB | Security |
| [0006](0006-single-call-intent-and-sql.md) | One structured call returns intent then SQL | D3 |
| [0007](0007-pandas-only.md) | Pandas only | D2 |
| [0008](0008-excel-engine.md) | Calamine for reading Excel; OpenPyXL for writing | D1 |
| [0009](0009-region-and-relative-dates.md) | NZ defaults; relative dates anchored to the dataset | D5 |
| [0010](0010-data-sent-to-openai.md) | What dataset content goes to OpenAI | D4 |
| [0011](0011-clarification-policy.md) | When to ask for clarification | D6 |
| [0012](0012-alpha-access-control.md) | Password, spend cap and rate limit for the alpha | D7 |
| [0013](0013-mvp-acceptance-thresholds.md) | MVP acceptance thresholds | D8 |
| [0014](0014-resource-limits.md) | Resource limits | D9 |
| [0015](0015-tooling.md) | uv, ruff, pytest, GitHub Actions | D10 |
| [0016](0016-repository-and-workflow.md) | New repository, branches and reviewed pull requests | D11 |
| [0017](0017-grounded-numbers.md) | Every number in an answer is grounded in results | Pipeline |
| [0018](0018-standing-push-approval.md) | Finished task branches are pushed and opened as PRs without asking; merging stays with Fahad (amends 0016) | 2026-11-22 |

Format: context, decision, alternatives considered, consequences. A changed decision gets a new ADR that supersedes the old one; accepted ADRs are not edited in place.
