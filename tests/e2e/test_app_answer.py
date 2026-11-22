"""How an answer is drawn: the table, the chart, the download and the inspector (FR-8, FR-9.1)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from analyst.core.models import Usage
from analyst.ui import chat
from tests.e2e.driver import Page
from tests.e2e.test_app_flows import ASK, HAPPY


def log_events(capsys) -> list[dict]:
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


# -- the table and the download --------------------------------------------------------------------


def test_a_table_is_shown_without_its_row_numbers() -> None:
    table = Page(HAPPY).open_file().ask(ASK).answer().dataframe[0]
    assert json.loads(table.proto.columns) == {"_index": {"hidden": True}}


def test_each_download_is_a_csv_file_named_for_its_place_in_the_conversation(monkeypatch) -> None:
    offered: list[dict] = []
    real = chat.st.download_button

    def recording(label, data, **kwargs):
        offered.append({"label": label, "data": data, **kwargs})
        return real(label, data, **kwargs)

    monkeypatch.setattr(chat.st, "download_button", recording)
    Page([*HAPPY, *HAPPY]).open_file().ask(ASK).ask(ASK)
    names = [o["file_name"] for o in offered if o["file_name"].endswith(".csv")]
    assert names[-2:] == ["results-1.csv", "results-2.csv"]  # the last run drew both answers
    assert {o["mime"] for o in offered} == {"text/csv"}
    assert {o["on_click"] for o in offered} == {"ignore"}  # a download does not rerun the page
    assert all(o["data"].startswith(b"\xef\xbb\xbf") for o in offered)


# -- the chart -------------------------------------------------------------------------------------


def themed(monkeypatch, theme) -> list[bool]:
    """Set the theme the browser reports, and record whether each chart was built for dark."""
    monkeypatch.setattr(chat.st, "context", SimpleNamespace(theme=theme))
    built: list[bool] = []
    real = chat.build_figure

    def recording(result, spec, *, dark):
        built.append(dark)
        return real(result, spec, dark=dark)

    monkeypatch.setattr(chat, "build_figure", recording)
    return built


def test_the_chart_is_built_for_the_theme_the_browser_is_showing(monkeypatch) -> None:
    built = themed(monkeypatch, SimpleNamespace(type="dark"))
    Page(HAPPY).open_file().ask(ASK)
    assert built == [True]
    built = themed(monkeypatch, SimpleNamespace(type="light"))
    Page(HAPPY).open_file().ask(ASK)
    assert built == [False]


def test_a_theme_that_cannot_be_read_gives_the_light_chart(monkeypatch) -> None:
    class Unreadable:
        @property
        def type(self) -> str:
            raise RuntimeError("no theme here")

    built = themed(monkeypatch, Unreadable())
    Page(HAPPY).open_file().ask(ASK)
    assert built == [False]
    assert chat.is_dark() is False


def test_a_chart_that_is_drawn_again_is_not_rebuilt_for_the_same_theme(monkeypatch) -> None:
    built = themed(monkeypatch, SimpleNamespace(type="light"))
    page = Page(HAPPY).open_file().ask(ASK)
    page.at.run()
    page.at.run()
    assert built == [False]
    assert len(page.answer().get("plotly_chart")) == 1


def test_a_chart_that_cannot_be_built_is_logged_by_its_type_only(monkeypatch, capsys) -> None:
    def broken(*args, **kwargs):
        raise ValueError("SECRET-CELL-IN-THE-ERROR")

    monkeypatch.setattr(chat, "build_figure", broken)
    Page(HAPPY).open_file().ask(ASK)
    events = log_events(capsys)
    (failed,) = [e for e in events if e["event"] == "chart_not_drawn"]
    assert failed["error_type"] == "ValueError"
    assert "SECRET-CELL-IN-THE-ERROR" not in json.dumps(events)


def test_an_answer_without_a_chart_suggestion_draws_none_and_logs_no_failure(capsys) -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    entry = page.session.history[0]
    entry.response = entry.response.model_copy(update={"chart": None})
    entry.drawn.clear()
    capsys.readouterr()
    page.at.run()
    assert not page.at.exception
    assert len(page.answer().get("plotly_chart")) == 0
    assert [e for e in log_events(capsys) if e["event"] == "chart_not_drawn"] == []


def test_a_suggestion_of_no_chart_is_not_a_chart_that_failed(capsys) -> None:
    from analyst.core.models import ChartSpec

    page = Page(HAPPY).open_file().ask(ASK)
    entry = page.session.history[0]
    nothing = ChartSpec(type="none", x=None, y=[], color=None)
    entry.response = entry.response.model_copy(update={"chart": nothing})
    capsys.readouterr()
    page.at.run()
    assert not page.at.exception
    assert len(page.answer().get("plotly_chart")) == 0
    assert [e for e in log_events(capsys) if e["event"] == "chart_not_drawn"] == []


# -- the inspector ---------------------------------------------------------------------------------


def test_the_query_is_shown_as_sql_with_long_lines_wrapped() -> None:
    code = Page(HAPPY).open_file().ask(ASK).answer().expander[0].code[0]
    assert code.language == "sql"
    assert code.proto.wrap_lines is True


def test_a_result_cut_at_the_row_limit_says_so_in_the_inspector(monkeypatch) -> None:
    monkeypatch.setenv("MAX_RESULT_ROWS", "2")
    page = Page(HAPPY).open_file().ask(ASK)
    facts = page.answer().expander[0].markdown[0].value
    assert "**Rows returned:** 2 \\(cut at the row limit\\)" in facts


def test_a_result_that_was_not_cut_does_not_say_so() -> None:
    facts = Page(HAPPY).open_file().ask(ASK).answer().expander[0].markdown[0].value
    assert "cut at the row limit" not in facts


def test_the_inspector_gives_the_model_calls_and_the_time_they_took() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    entry = page.session.history[0]
    usage = Usage(llm_calls=3, latency_ms=1_000_000)
    entry.response = entry.response.model_copy(update={"usage": usage})
    page.at.run()
    assert "**Model calls:** 3\\, 1000\\.0 s in all" in page.answer().expander[0].markdown[0].value
