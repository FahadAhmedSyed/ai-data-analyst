# AI Data Analyst

Upload a CSV or Excel file and ask questions about it in plain English. The app turns each question into SQL, checks the SQL, runs it on your data with DuckDB, and answers with an explanation, a table and a chart. Every number in an answer comes from an executed query, never from the language model.

> Status: Phase 1 in progress. The app works end to end with an OpenAI key; deployment and the accuracy benchmark are still to come. See [docs/backlog.md](docs/backlog.md).

## How it works

1. Your file is parsed in memory and loaded into a private DuckDB database for your browser session.
2. A language model turns your question into a structured intent and a SQL query.
3. The query is checked by a SQL parser and an allowlist before it runs on a locked-down database that cannot reach files, the network or extensions.
4. The result is explained in plain language, and every number in the explanation is checked against the result.

Details: [architecture](docs/architecture.md), [security](docs/security.md), [privacy](docs/privacy.md).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key (only to ask questions; tests don't need one)

## Run locally

```bash
uv sync
cp .env.example .env        # then fill in OPENAI_API_KEY and, optionally, APP_PASSWORD
uv run streamlit run app.py
```

The app opens at http://localhost:8501. Choose a CSV or Excel file in the sidebar, check how it will be read, press **Load file**, then ask questions. If `APP_PASSWORD` is set, the page asks for it first. Without `OPENAI_API_KEY` you can still load a file and look at it; questions are switched off.

## Configuration

All settings are environment variables, listed with defaults in [.env.example](.env.example) and defined in [analyst/config/settings.py](analyst/config/settings.py). Limits follow [ADR 0014](docs/adr/0014-resource-limits.md).

## Tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov
```

The default run needs no network or API key. Tests marked `slow` or `live` are excluded; run them with `uv run pytest -m slow` or `-m live`. The accuracy benchmark is described in [eval/README.md](eval/README.md).

## Project layout

```text
app.py              Streamlit entry point (thin)
analyst/            Application package
  config/           Settings, prompts, region rules
  core/             Shared models, errors, logging
  session/          One browser session: loading a file, rate limit, idle removal, display text
  ui/               The Streamlit page: gate, sidebar, data panel, chat
docs/               PRD, architecture, security, privacy, testing, ADRs
eval/               Accuracy benchmark
tests/              unit/, integration/, e2e/
```

More packages are added as Phase 1 tasks land; see [docs/architecture.md](docs/architecture.md) for the full layout.

## Documentation

- [Product requirements](docs/prd.md)
- [Architecture](docs/architecture.md)
- [Architecture decision records](docs/adr/README.md)
- [Security](docs/security.md) and [privacy](docs/privacy.md)
- [Testing strategy](docs/testing.md) and [benchmark](eval/README.md)
- [Risk register](docs/risks.md) and [Phase 1 backlog](docs/backlog.md)
- [Phase 0 spike results](docs/spikes.md)
