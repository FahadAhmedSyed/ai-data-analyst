"""The conversation: questions, answers, tables, charts and the SQL inspector.

What an answer shows (PRD journeys J2 to J5, FR-7.2, FR-8.1 to FR-8.3, FR-9.1, FR-9.2):

* how the question was read (``Interpreted as``) and the assumptions made, always visible;
* the answer, then any notes (rows cut, missing values) and whether the written explanation
  could be checked or a plain summary is shown instead;
* the result as a table, a chart where one fits, and a CSV download;
* an expander with the SQL that ran, whether it ran, the row count, the filters and period,
  and what the question cost in model calls and time.

A clarifying question and a failure are shown in their own way, never as an answer. Text from
the file or the model is escaped so it cannot become a link or an image; the cleaned
explanation is the one thing shown as markdown (``$`` is still escaped, or two prices in a
sentence would be read as a formula).
"""

from __future__ import annotations

from typing import get_args

import streamlit as st

from analyst.core.logging import get_logger
from analyst.core.models import AnalystResponse
from analyst.pipeline.analyst import ProgressStage
from analyst.session.display import (
    describe_intent,
    display_frame,
    escape_dollars,
    escape_markdown,
)
from analyst.session.state import AnalystSession, ChatEntry
from analyst.sql.context import MAX_QUESTION_CHARS
from analyst.visualization.charts import build_figure

log = get_logger(__name__)

STAGE_TEXT: dict[str, str] = {
    "understanding": "Understanding your question…",
    "running": "Running the query…",
    "explaining": "Writing the answer…",
}
assert set(STAGE_TEXT) == set(get_args(ProgressStage))  # noqa: S101 - fails at import if they drift

NO_DATASET_PROMPT = "Upload a file in the sidebar to ask questions"
ASK_PROMPT = "Ask a question about your data"


def chat_box(session: AnalystSession) -> str | None:
    """Draw the chat box and return the question sent, if one was.

    It is drawn first, though it appears at the bottom of the page, so that the page can
    already fold away the data panel on the run that answers a question.
    """
    if not session.has_dataset:
        st.chat_input(NO_DATASET_PROMPT, disabled=True)
        return None
    return st.chat_input(ASK_PROMPT, max_chars=MAX_QUESTION_CHARS) or None


def render(session: AnalystSession, question: str | None) -> None:
    """Draw the conversation so far, then run ``question`` and draw its answer."""
    dataset = session.dataset
    if dataset is None:
        return
    names = dataset.profile.display_names
    for index, entry in enumerate(session.history):
        _entry(index, entry, names)
    if question:
        _ask(session, question, names)


def _ask(session: AnalystSession, question: str, names: dict[str, str]) -> None:
    index = len(session.history)
    with st.chat_message("user"):
        st.markdown(escape_markdown(question))
    with st.chat_message("assistant"):
        slot = st.empty()
        slot.status(STAGE_TEXT["understanding"], state="running")

        def progress(stage: str) -> None:
            slot.status(STAGE_TEXT[stage], state="running")

        entry = session.ask(question, progress)
        slot.empty()
        _response_or_notice(index, entry, names)


def _entry(index: int, entry: ChatEntry, names: dict[str, str]) -> None:
    with st.chat_message("user"):
        st.markdown(escape_markdown(entry.question))
    with st.chat_message("assistant"):
        _response_or_notice(index, entry, names)


def _response_or_notice(index: int, entry: ChatEntry, names: dict[str, str]) -> None:
    if entry.response is None:
        st.warning(entry.notice)
    else:
        _response(index, entry, entry.response, names)


# -- one answer -----------------------------------------------------------------------------------


def _response(
    index: int, entry: ChatEntry, response: AnalystResponse, names: dict[str, str]
) -> None:
    if response.resolved_question:
        st.caption(f"Interpreted as: {escape_markdown(response.resolved_question)}")
    if response.assumptions:
        st.caption("Assumptions: " + "; ".join(escape_markdown(a) for a in response.assumptions))
    if response.status == "failed":
        st.warning(escape_dollars(response.message))
    elif response.status == "clarification":
        st.info(escape_dollars(response.message))
    else:
        st.markdown(escape_dollars(response.message))
        if response.fallback_used:
            st.caption(
                "The written explanation could not be checked against the data, "
                "so this is a plain summary. The table below is the answer."
            )
        for caveat in response.caveats:
            st.caption(f"Note: {escape_markdown(caveat)}")
        _result(index, entry, response)
    if response.sql:
        _inspector(response, names)


def _result(index: int, entry: ChatEntry, response: AnalystResponse) -> None:
    result = response.result
    if result is None or result.is_empty:
        return
    table = entry.drawn.get("table")
    if table is None:
        table = entry.drawn["table"] = display_frame(result.dataframe)
    st.dataframe(table, hide_index=True)
    _chart(index, entry, response)
    download = entry.download()
    if download is not None:
        st.download_button(
            "Download results (CSV)",
            download,
            file_name=f"results-{index + 1}.csv",
            mime="text/csv",
            key=f"download-{index}",
            on_click="ignore",
        )


def is_dark() -> bool:
    """Whether the browser is showing the dark theme (charts have a colour set for each)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _chart(index: int, entry: ChatEntry, response: AnalystResponse) -> None:
    spec, result = response.chart, response.result
    if spec is None or spec.type == "none" or result is None:
        return
    dark = is_dark()
    cache_key = ("figure", dark)
    figure = entry.drawn.get(cache_key)
    if figure is None:
        try:
            figure = build_figure(result, spec, dark=dark)
        except Exception as error:  # a chart is decoration: the table is already shown
            log.error("chart_not_drawn", extra={"error_type": type(error).__name__})
            return
        entry.drawn[cache_key] = figure
    # theme=None: the chart carries its own validated colours, which Streamlit would replace.
    st.plotly_chart(figure, theme=None, width="stretch", key=f"chart-{index}")


def _inspector(response: AnalystResponse, names: dict[str, str]) -> None:
    result = response.result
    label = "How this was answered" if result is not None else "The query that was tried"
    with st.expander(label):
        st.code(response.sql, language="sql", wrap_lines=True)
        facts: list[tuple[str, str]] = []
        if result is not None:
            rows = f"{result.row_count:,}"
            facts.append(("Status", "ran successfully"))
            facts.append(
                ("Rows returned", rows + (" (cut at the row limit)" if result.truncated else ""))
            )
            facts.append(("Run time", f"{result.duration_ms:,.0f} ms"))
        else:
            facts.append(("Status", "did not produce a result"))
        usage = response.usage
        facts.append(("Model calls", f"{usage.llm_calls}, {usage.latency_ms / 1000:.1f} s in all"))
        lines = [f"- **{name}:** {escape_markdown(value)}" for name, value in facts]
        lines += _filter_lines(response, names)
        st.markdown("\n".join(lines))


def _filter_lines(response: AnalystResponse, names: dict[str, str]) -> list[str]:
    filters = describe_intent(response.turn.intent, names)
    if not filters:
        return ["- **Filters and period:** none"]
    return ["- **Filters and period:**"] + [f"    - {escape_markdown(f)}" for f in filters]
