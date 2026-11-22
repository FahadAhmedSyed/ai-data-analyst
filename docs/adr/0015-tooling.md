# ADR 0015: Tooling: uv, ruff, pytest, GitHub Actions

- Status: Accepted
- Date: 2026-11-22
- Covers decision: D10

## Decision

- Dependencies declared in `pyproject.toml` and locked with uv (`uv.lock`). A `requirements.txt` is exported for Railway.
- ruff for linting and formatting; pytest for tests; type hints throughout.
- GitHub Actions runs ruff and the offline test suite (unit, fake-LLM integration, Streamlit AppTest) on every push. The live benchmark runs on demand with a cost cap.

## Alternatives considered

- pip with `requirements.txt` only: no lockfile, less reproducible builds.

## Consequences

- Reproducible environments locally, in CI and on Railway.
