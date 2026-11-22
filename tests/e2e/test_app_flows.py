"""The app from the user's side (T1.18): AppTest flows with a scripted model.

The page is the real one; the model is a fake that answers from a script, so each test says
what the model returns and checks what the user is shown. The validator, executor,
explainer and chart recommender behind it are the real ones.
"""

from __future__ import annotations

from analyst.core.models import ChartSpec
from tests.e2e.driver import SALES, Page
from tests.unit.test_analyst import explains
from tests.unit.test_generator import answer, asks, refuses

SAID = "Revenue is shown for each region."
ASK = "Total revenue by region?"
HAPPY = [("generate", answer()), ("explain", explains(SAID))]


# -- before a file --------------------------------------------------------------------------------


def test_the_first_page_explains_the_app_and_asks_for_a_file() -> None:
    page = Page()
    assert page.at.title[0].value == "AI Data Analyst"
    assert "Upload" in page.at.markdown[0].value
    assert len(page.at.file_uploader) == 1
    assert page.at.file_uploader[0].label == "Upload a CSV or Excel file"
    assert page.at.sidebar.header[0].value == "Your data"
    assert [b.label for b in page.at.button] == ["New conversation"]
    assert page.at.button[0].disabled


def test_the_chat_box_is_there_but_switched_off_until_there_is_data() -> None:
    page = Page()
    assert len(page.at.chat_input) == 1
    assert page.at.chat_input[0].disabled


def test_the_privacy_notice_is_shown_before_anything_is_uploaded() -> None:
    page = Page()
    assert any("sends OpenAI the column names" in t for t in page.texts("caption"))
    assert any("Schema-only mode leaves out" in t for t in page.texts("caption"))


def test_the_uploader_takes_the_supported_types_up_to_the_configured_size() -> None:
    page = Page()
    uploader = page.at.file_uploader[0]
    assert sorted(uploader.allowed_type) == [".csv", ".txt", ".xls", ".xlsx"]


# -- a staged file --------------------------------------------------------------------------------


def test_choosing_a_file_offers_how_to_read_it_and_a_button_but_does_not_load_it() -> None:
    page = Page().upload("sales.csv", SALES)
    assert page.at.button[0].label == "Load file"
    assert not page.session.has_dataset
    assert page.at.selectbox[0].label == "Separator"
    assert page.at.selectbox[0].value is None
    assert page.at.radio[0].label == "Header row"
    assert page.at.radio[0].value == "auto"


def test_loading_a_file_shows_the_data_and_empties_the_uploader() -> None:
    page = Page(HAPPY).open_file()
    assert page.session.has_dataset
    assert page.at.file_uploader[0].value is None
    assert page.at.file_uploader[0].label == "Replace the file"
    assert [t.label for t in page.at.tabs] == ["Preview", "Columns", "Data quality"]
    assert not page.at.chat_input[0].disabled
    assert not page.at.error


def test_the_file_name_and_size_are_shown_in_the_sidebar_once_loaded() -> None:
    page = Page(HAPPY).open_file()
    sidebar = [m.value for m in page.at.sidebar.markdown]
    assert "**sales\\.csv**" in sidebar
    assert "4 rows, 4 columns" in [c.value for c in page.at.sidebar.caption]


def test_the_preview_shows_the_rows_under_their_original_names() -> None:
    page = Page(HAPPY).open_file()
    preview = page.at.dataframe[0].value
    assert list(preview.columns) == ["Order ID", "Region", "Revenue", "Order Date"]
    assert preview["Region"].tolist() == ["North", "South", "North", "East"]
    assert any(t.startswith("All 4 rows, as the app reads them") for t in page.texts("caption"))
    assert any("separated by commas" in t for t in page.texts("caption"))


def test_the_columns_tab_has_the_overview_and_the_numbers() -> None:
    page = Page(HAPPY).open_file()
    tables = [d.value for d in page.at.dataframe]
    overview = next(t for t in tables if "Missing %" in t.columns)
    assert overview["Column"].tolist() == ["Order ID", "Region", "Revenue", "Order Date"]
    numbers = next(t for t in tables if "Std deviation" in t.columns)
    assert numbers["Column"].tolist() == ["Order ID", "Revenue"]
    assert numbers.set_index("Column").loc["Revenue", "Average"] == "25.12"  # (10.5+20+30+40)/4


def test_the_columns_tab_has_four_titled_tables_in_a_fixed_order() -> None:
    page = Page(HAPPY).open_file()
    titles = [m.value for m in page.at.markdown if m.value.startswith("**")]
    assert titles[:4] == ["**All columns**", "**Numbers**", "**Text and true/false**", "**Dates**"]
    tables = [d.value for d in page.at.dataframe][1:]  # the first is the preview
    assert [list(t.columns)[:2] for t in tables] == [
        ["Column", "Type"],
        ["Column", "Smallest"],
        ["Column", "Different values"],
        ["Column", "Earliest"],
    ]
    assert tables[2]["Column"].tolist() == ["Region"]
    assert tables[3]["Column"].tolist() == ["Order Date"]
    assert tables[3]["Earliest"].tolist() == ["2024-01-01"]


def test_a_dataset_with_no_dates_has_no_dates_table() -> None:
    page = Page(HAPPY).open_file("other.csv", b"Name,Score\nAnn,5\nBo,7\n")
    titles = [m.value for m in page.at.markdown if m.value.startswith("**")]
    assert "**Dates**" not in titles
    assert "**Numbers**" in titles


def test_the_tables_of_the_data_panel_are_shown_without_row_numbers() -> None:
    import json

    page = Page(HAPPY).open_file()
    assert len(page.at.dataframe) == 5
    for table in page.at.dataframe:
        assert json.loads(table.proto.columns) == {"_index": {"hidden": True}}


def test_the_quality_tab_says_so_when_there_is_nothing_to_report() -> None:
    page = Page(HAPPY).open_file()
    assert any("No missing values, duplicate rows" in s.value for s in page.at.success)
    assert "The app reports these notes and never changes your data." in page.texts("caption")


def test_the_quality_tab_lists_what_it_found() -> None:
    csv = b"Site,Rain\nA,1\nA,1\nB,\n"
    page = Page(HAPPY).open_file("rain.csv", csv)
    notes = next(m.value for m in page.at.markdown if "exact copies" in m.value)
    assert "Rain is missing in 1 rows" in notes.replace("\\", "")
    assert "2 rows" not in notes or "exact copies" in notes


# -- asking ---------------------------------------------------------------------------------------


def test_an_answer_shows_how_the_question_was_read_the_answer_the_table_and_the_chart() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    assert page.questions() == ["Total revenue by region\\?"]
    reply = page.answer()
    captions = [c.value for c in reply.caption]
    assert "Interpreted as: What is the total revenue for each region\\?" in captions
    assert "Assumptions: Revenue means the revenue column\\." in captions
    assert reply.markdown[0].value == SAID
    table = reply.dataframe[0].value
    assert table["region"].tolist() == ["North", "South", "East"]
    assert table["total_revenue"].tolist() == [40.5, 20.0, 40.0]
    assert len(reply.get("plotly_chart")) == 1
    assert [b.label for b in reply.get("download_button")] == ["Download results (CSV)"]
    page.clients[0].assert_exhausted()


def test_the_sql_inspector_shows_the_query_status_rows_filters_and_cost() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    inspector = page.answer().expander[0]
    assert inspector.label == "How this was answered"
    assert "SUM(revenue)" in inspector.code[0].value
    assert inspector.code[0].language == "sql"
    facts = inspector.markdown[0].value
    assert "**Status:** ran successfully" in facts
    assert "**Rows returned:** 3" in facts
    assert "**Run time:**" in facts
    assert "**Model calls:** 2" in facts
    assert "**Filters and period:** none" in facts


def test_the_filters_of_the_question_are_listed_in_the_inspector() -> None:
    from analyst.core.models import FilterCondition
    from tests.unit.test_context import INTENT

    intent = INTENT.model_copy(
        update={"filters": [FilterCondition(column="region", operator="=", values=["North"])]}
    )
    script = [("generate", answer(intent=intent)), ("explain", explains(SAID))]
    page = Page(script).open_file().ask(ASK)
    facts = page.answer().expander[0].markdown[0].value
    assert "**Filters and period:**" in facts
    assert "region is 'North'" in facts.replace("\\", "") or "Region is 'North'" in facts.replace(
        "\\", ""
    )


def test_a_dollar_amount_in_an_answer_is_not_read_as_a_formula() -> None:
    text = "North made $40.50, South $20 and East $40."
    page = Page([("generate", answer()), ("explain", explains(text))]).open_file().ask(ASK)
    assert page.answer().markdown[0].value == "North made \\$40.50, South \\$20 and East \\$40."


def test_the_download_is_the_result_as_a_csv_file() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    entry = page.session.history[0]
    data = entry.download()
    assert data is not None
    assert data.decode("utf-8-sig").splitlines() == [
        "region,total_revenue",
        "North,40.5",
        "South,20.0",
        "East,40.0",
    ]


def test_a_second_question_is_answered_below_the_first_and_the_first_stays() -> None:
    script = [*HAPPY, ("generate", answer(resolved_question="Again.")), ("explain", explains(SAID))]
    page = Page(script).open_file().ask(ASK).ask("And again?")
    assert page.questions() == ["Total revenue by region\\?", "And again\\?"]
    assert len(page.at.chat_message) == 4
    assert len(page.session.history) == 2
    assert page.answer(0).markdown[0].value == SAID
    assert "Interpreted as: Again\\." in [c.value for c in page.answer(1).caption]


def test_every_answer_has_its_own_download_button() -> None:
    script = [*HAPPY, *HAPPY]
    page = Page(script).open_file().ask(ASK).ask("Again?")
    keys = [b.key for b in page.at.get("download_button")]
    assert keys == ["download-0", "download-1"]


def test_the_data_panel_is_open_before_the_first_question_and_folded_after() -> None:
    page = Page(HAPPY).open_file()
    panel = next(e for e in page.at.expander if e.label.startswith("Your data"))
    assert panel.proto.expanded is True
    page.ask(ASK)
    panel = next(e for e in page.at.expander if e.label.startswith("Your data"))
    assert panel.proto.expanded is False


# -- not an answer --------------------------------------------------------------------------------


def test_a_clarifying_question_is_shown_as_a_question_and_nothing_runs() -> None:
    page = Page([("generate", asks())]).open_file().ask("Which is best?")
    reply = page.answer()
    assert reply.info[0].value == "Do you mean revenue or quantity?"
    assert len(reply.dataframe) == 0
    assert len(reply.get("download_button")) == 0
    assert len(reply.expander) == 0  # no SQL: nothing ran
    assert page.session.history[0].response.status == "clarification"  # type: ignore[union-attr]


def test_a_request_to_change_data_is_shown_as_a_failure_with_the_models_explanation() -> None:
    page = Page([("generate", refuses())]).open_file().ask("Delete all the rows")
    reply = page.answer()
    assert reply.warning[0].value == "This tool only reads data, so it cannot change it."
    assert len(reply.dataframe) == 0
    assert len(reply.expander) == 0


def test_a_query_that_is_refused_shows_a_safety_message_and_the_query_that_was_tried() -> None:
    page = Page([("generate", answer(sql="DROP TABLE sales"))]).open_file().ask("Drop it")
    reply = page.answer()
    assert reply.warning[0].value == (
        "This tool only reads data, so a query that changes data or settings was not run."
    )
    assert len(reply.dataframe) == 0
    inspector = reply.expander[0]
    assert inspector.label == "The query that was tried"
    assert inspector.code[0].value == "DROP TABLE sales"
    assert "did not produce a result" in inspector.markdown[0].value


def test_a_failed_model_call_is_a_plain_message_not_a_stack_trace() -> None:
    from analyst.core.errors import LLMError

    page = Page([("generate", LLMError("timeout", "slow"))]).open_file().ask(ASK)
    assert not page.at.exception
    assert page.answer().warning[0].value == (
        "The language model took too long to respond. Please try again."
    )


def test_an_unconfigured_model_says_so_in_the_conversation() -> None:
    from analyst.core.errors import LLMError

    page = Page([("generate", LLMError("not_configured", "no key"))]).open_file().ask(ASK)
    assert "isn't set up yet" in page.answer().warning[0].value


def test_a_bug_in_the_pipeline_is_a_plain_message(monkeypatch) -> None:
    from analyst.pipeline import analyst as analyst_module

    def broken(*args, **kwargs):
        raise RuntimeError("SECRET-BUG-TEXT")

    page = Page(HAPPY).open_file()
    monkeypatch.setattr(analyst_module.Analyst, "_answer", broken)
    page.ask(ASK)
    assert not page.at.exception
    assert page.answer().warning[0].value == "Something went wrong. Please try again."
    assert "SECRET-BUG-TEXT" not in page.page_text()


def test_a_chart_that_cannot_be_drawn_costs_only_the_chart(monkeypatch) -> None:
    from analyst.ui import chat

    def broken(*args, **kwargs):
        raise ValueError("no chart")

    monkeypatch.setattr(chat, "build_figure", broken)
    page = Page(HAPPY).open_file().ask(ASK)
    reply = page.answer()
    assert not page.at.exception
    assert reply.markdown[0].value == SAID
    assert len(reply.dataframe) == 1
    assert len(reply.get("plotly_chart")) == 0


def test_a_suggestion_of_no_chart_draws_none() -> None:
    script = [
        ("generate", answer(chart=ChartSpec(type="none", x=None, y=[], color=None))),
        HAPPY[1],
    ]
    page = Page(script).open_file().ask("List the regions")
    assert len(page.answer().get("plotly_chart")) == 0


def test_an_empty_result_has_no_table_chart_or_download_but_keeps_the_sql() -> None:
    sql = "SELECT region, revenue FROM sales WHERE region = 'Nowhere'"
    page = Page([("generate", answer(sql=sql, chart=None))]).open_file().ask("Sales in Nowhere?")
    reply = page.answer()
    assert len(reply.dataframe) == 0
    assert len(reply.get("download_button")) == 0
    assert len(reply.get("plotly_chart")) == 0
    assert "Nowhere" in reply.expander[0].code[0].value
    assert "no rows" in reply.markdown[0].value.lower() or "No rows" in reply.markdown[0].value


def test_an_explanation_that_could_not_be_checked_is_replaced_and_says_so() -> None:
    bad = explains("Revenue is up 99 percent.")
    script = [("generate", answer()), ("explain", bad), ("explain", bad)]
    page = Page(script).open_file().ask(ASK)
    reply = page.answer()
    assert reply.markdown[0].value.startswith("Here is the result of the query")
    assert any("could not be checked against the data" in c.value for c in reply.caption)
    assert len(reply.dataframe) == 1  # the table is the answer


def test_a_note_about_the_result_is_shown_under_the_answer() -> None:
    csv = b"Region,Revenue\nNorth,10\nSouth,\nNorth,30\n"
    script = [
        ("generate", answer(sql="SELECT region, revenue FROM sales")),
        ("explain", explains("Revenue is shown for each sale.")),
    ]
    page = Page([script[0], script[1]]).open_file("sales.csv", csv).ask("List sales")
    notes = [c.value for c in page.answer().caption if c.value.startswith("Note:")]
    assert any("missing" in n for n in notes)


# -- starting over --------------------------------------------------------------------------------


def test_a_new_conversation_clears_the_questions_and_keeps_the_data() -> None:
    page = Page([*HAPPY, *HAPPY]).open_file().ask(ASK)
    assert not page.at.button[-1].disabled
    page.at.button[-1].click().run()
    assert len(page.at.chat_message) == 0
    assert page.session.has_dataset
    assert page.session.history == []
    assert page.at.button[-1].disabled
    page.ask(ASK)
    assert len(page.at.chat_message) == 2


def test_a_new_conversation_forgets_what_was_assumed() -> None:
    script = [*HAPPY, ("generate", answer(assumptions=[])), ("explain", explains(SAID))]
    page = Page(script).open_file().ask(ASK)
    page.at.button[-1].click().run()
    page.ask(ASK)
    prompt = page.clients[0].calls[2].messages[1].content
    assert "Revenue means the revenue column" not in prompt
    assert "HISTORY" not in prompt


# -- the question box ------------------------------------------------------------------------------


def test_the_question_box_takes_up_to_the_longest_question_the_app_accepts() -> None:
    from analyst.sql.context import MAX_QUESTION_CHARS

    page = Page(HAPPY).open_file()
    assert page.at.chat_input[0].proto.max_chars == MAX_QUESTION_CHARS == 2_000


def test_a_question_over_the_longest_is_cut_by_the_box_before_it_reaches_the_app() -> None:
    page = Page(HAPPY).open_file().ask("x" * 5_000)
    assert len(page.session.history[0].question) == 2_000
    assert page.session.history[0].response is not None


def test_a_progress_line_is_given_to_the_question_and_gone_once_it_is_answered(monkeypatch) -> None:
    from analyst.session.state import AnalystSession

    stages: list[str] = []
    real = AnalystSession.ask

    def watching(self, question, on_progress=None):
        assert callable(on_progress)
        original = on_progress

        def spy(stage: str) -> None:
            stages.append(stage)
            original(stage)

        return real(self, question, spy)

    monkeypatch.setattr(AnalystSession, "ask", watching)
    page = Page(HAPPY).open_file().ask(ASK)
    assert stages == ["understanding", "running", "explaining"]
    assert len(page.at.status) == 0  # replaced by the answer
    assert not page.at.exception


# -- what is logged --------------------------------------------------------------------------------


def test_a_session_is_logged_once_by_its_label_and_nothing_about_the_user(capsys) -> None:
    import json

    page = Page(HAPPY).open_file().ask(ASK)
    page.at.run()
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    (started,) = [r for r in lines if r["event"] == "session_started"]
    assert started["session"] == page.session.id
    assert set(started) == {"ts", "level", "logger", "event", "session"}
    written = json.dumps(lines)
    for text in ("sales.csv", "North", ASK, "Revenue"):
        assert text not in written


def test_the_progress_line_names_each_stage_as_it_is_reached(monkeypatch) -> None:
    from streamlit.delta_generator import DeltaGenerator

    from analyst.ui.chat import STAGE_TEXT

    drawn: list[tuple[str, str]] = []
    real = DeltaGenerator.status

    def recording(self, label, *, state="running", **kwargs):
        drawn.append((label, state))
        return real(self, label, state=state, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "status", recording)
    Page(HAPPY).open_file().ask(ASK)
    said = [label for label, _ in drawn]
    # the line is there from the moment the question is sent, then follows the stages
    assert said == [STAGE_TEXT["understanding"], *STAGE_TEXT.values()]
    assert {state for _, state in drawn} == {"running"}
