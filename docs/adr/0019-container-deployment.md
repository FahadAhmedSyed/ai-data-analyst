# ADR 0019: Deploy on Railway as a container that starts closed

- Status: Accepted
- Date: 2026-11-24
- Covers decision: T1.20 (builds on ADR 0001, 0012 and 0015)

## Context

Architecture section 12 planned one Railway service with a start command, a health check and `railway.json`. Three facts changed the details:

- Railway's documentation says Config as Code files (`railway.json`, `railway.toml`) are deprecated and keep working for existing services until 2026-12-01. A forum answer says that after that date the file is not read. A start command kept only in `railway.json` could stop applying one week after this task.
- Railway always builds a `Dockerfile` when the repository has one, and the same image can be built and run on a laptop, which a Railpack build cannot.
- ADR 0012 gates the app with `APP_PASSWORD`, but the gate was open when the variable was missing. A deployment that forgot the variable would have been open to everyone with the address, spending the OpenAI budget.

The sample repository suggested for this task is a Flask application. Its deployment ideas apply here (a health check, a restart policy, a stdlib check of the live address, refusing to start with a missing secret); its page design does not, because this app is Streamlit (ADR 0001).

## Decision

- A `Dockerfile` builds the image: Python 3.12 slim, the packages from `requirements.txt` (ADR 0015), only `app.py`, `.streamlit/` and `analyst/` copied in, run by a user that is not root. The start command is its `CMD`: `streamlit run app.py` on `$PORT` and all addresses. A `.dockerignore` keeps `.env`, data, reports and tests out of the build.
- `railway.json` selects the Dockerfile and sets the health check (`/_stcore/health`, Streamlit's own route), its timeout and the restart policy. It holds nothing the app needs to start. `docs/deployment.md` lists the same values for the dashboard.
- One replica, because every session lives in the process (ADR 0001, 0002).
- The image sets `REQUIRE_GATE=true`. With that set and no `APP_PASSWORD`, the page shows a closed message and nothing else, logs an error, and the health check still answers. Running without a password stays possible locally, where the variable is not set.
- `scripts/check_deployment.py` checks a live address with the standard library only.
- The plan is 2 GB (spike (d)). Railway's Free (0.5 GB) and Trial (1 GB) plans cannot hold it, so Hobby is the smallest that can. The memory alert is a Pro-plan dashboard feature; on Hobby the Metrics graph is watched by hand.

## Alternatives considered

- **Railpack build with a start command in `railway.json`:** fewer files, but it depends on the deprecated file for the start command, cannot be tried on a laptop, and picks its own Python and install method.
- **Start command typed into the dashboard:** works, but lives outside the repository and is lost if the service is recreated.
- **Move to Flask as in the sample repository:** a rewrite of a working interface for a visual style, against ADR 0001. Not done.
- **Leave the gate open when no password is set:** simpler, but one forgotten variable exposes the budget.
- **A second replica for safety:** sessions would split between copies.

## Consequences

- A deploy is reproducible and has been built and run here; only the Railway-side steps (build, health check, domain, alert) wait for the first deploy.
- A forgotten `APP_PASSWORD` shows a closed page rather than an open app. The cost is a visible, fixable message.
- The image is large (about 750 MB, mostly pandas, DuckDB, Plotly and Streamlit).
- Updating packages stays a two-step job (`uv lock`, then `uv export`); CI already fails when `requirements.txt` is out of step.
