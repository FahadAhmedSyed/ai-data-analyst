# ADR 0001: Python monolith with a Streamlit UI and a UI-independent pipeline

- Status: Accepted
- Date: 2026-11-22
- Related decisions: stack review in the development plan

## Context

The MVP needs a web UI for non-technical users, fast iteration, and the Python data ecosystem (DuckDB, pandas, Plotly, SQLGlot). Streamlit gives a usable UI quickly, but its rerun model, websocket sessions and limited auth make it a weak foundation for a multi-user production service.

## Decision

Build the MVP as one Python 3.12 process. Streamlit is the UI. All application logic lives in an importable package (`analyst/`), and the pipeline never imports Streamlit. `app.py` and `analyst/ui/` are thin layers that call `analyst.pipeline.analyst.ask(question, session)`.

## Alternatives considered

- FastAPI backend plus a React frontend: better long-term scaling and auth, but two codebases and weeks of extra work before the first answer.
- Gradio or Dash: no clear advantage over Streamlit for a chat-plus-data app.

## Consequences

- Fast route to a working MVP.
- The pipeline can be tested without a browser and later served behind an API.
- Streamlit must run as a single replica on Railway (no sticky sessions needed).
- The UI choice is revisited at Phase 5.
