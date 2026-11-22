"""What is sent to the model, and what the user is told about it (FR-10.3, privacy.md)."""

from __future__ import annotations

from analyst.ui.notices import privacy_notice
from tests.e2e.driver import OTHER, Page
from tests.e2e.test_app_flows import ASK, HAPPY

CELL_VALUES = ("North", "South", "East", "2024-01-03")
NOTE = "may already have included values"


def switch(page: Page, on: bool) -> Page:
    page.at.sidebar.toggle(key="schema_only").set_value(on).run()
    return page


def prompt(page: Page, call: int, client: int = 0) -> str:
    return "\n".join(m.content for m in page.clients[client].calls[call].messages)


def notes(page: Page) -> list[str]:
    return [t for t in page.texts("caption") if NOTE in t]


# -- the switch ---------------------------------------------------------------------------------


def test_schema_only_mode_is_off_until_it_is_switched_on() -> None:
    page = Page(HAPPY).open_file()
    toggle = page.at.sidebar.toggle(key="schema_only")
    assert toggle.label == "Schema-only mode"
    assert toggle.value is False
    assert page.session.schema_only is False


def test_with_the_switch_off_the_question_carries_sample_values_from_the_file() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    assert any(value in prompt(page, 0) for value in CELL_VALUES)


def test_with_the_switch_on_the_question_carries_no_values_from_the_file() -> None:
    page = switch(Page(HAPPY).open_file(), True).ask(ASK)
    assert page.session.schema_only is True
    first = prompt(page, 0)
    assert "region" in first  # the column names still go
    assert not [value for value in CELL_VALUES if value in first]


def test_the_rows_of_the_result_are_still_sent_for_the_explanation_and_the_notice_says_so() -> None:
    page = switch(Page(HAPPY).open_file(), True).ask(ASK)
    assert any(value in prompt(page, 1) for value in CELL_VALUES)
    switch_help = page.at.sidebar.toggle(key="schema_only").help
    assert "The rows of each answer are still sent" in switch_help


def test_the_switch_is_read_before_the_question_runs() -> None:
    """Switching on and asking in the same run (a click then a question) uses the new mode."""
    page = Page(HAPPY).open_file()
    page.at.sidebar.toggle(key="schema_only").set_value(True)
    page.at.chat_input[0].set_value(ASK).run()
    assert not [v for v in CELL_VALUES if v in prompt(page, 0)]


def test_the_choice_carries_over_to_the_next_dataset() -> None:
    page = switch(Page(HAPPY, []).open_file(), True)
    page.upload("other.csv", OTHER).load()
    assert page.session.dataset.profile.source_name == "other.csv"
    assert page.session.schema_only is True
    assert page.at.sidebar.toggle(key="schema_only").value is True


# -- what the conversation has already sent --------------------------------------------------------


def test_no_note_when_nothing_has_been_asked() -> None:
    page = switch(Page(HAPPY).open_file(), True)
    assert notes(page) == []


def test_turning_it_on_after_questions_were_asked_without_it_says_values_may_have_gone() -> None:
    page = Page([*HAPPY, *HAPPY]).open_file().ask(ASK)
    assert notes(page) == []  # the switch is off, so there is nothing to warn about
    switch(page, True)
    assert len(notes(page)) == 1
    assert "Start a new conversation" in notes(page)[0]


def test_no_note_when_the_mode_was_on_from_the_start() -> None:
    page = switch(Page([*HAPPY, *HAPPY]).open_file(), True).ask(ASK).ask(ASK)
    assert notes(page) == []


def test_the_note_stays_while_the_old_questions_stay_and_goes_with_a_new_conversation() -> None:
    page = Page([*HAPPY, *HAPPY]).open_file().ask(ASK)
    switch(page, True).ask(ASK)
    assert len(notes(page)) == 1
    page.at.sidebar.button[-1].click().run()
    assert notes(page) == []


def test_the_note_goes_when_the_switch_goes_off_again() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    assert len(notes(switch(page, True))) == 1
    assert notes(switch(page, False)) == []


def test_each_answer_remembers_the_mode_it_was_asked_in() -> None:
    page = Page([*HAPPY, *HAPPY]).open_file().ask(ASK)
    switch(page, True).ask(ASK)
    assert [e.schema_only for e in page.session.history] == [False, True]
    assert page.session.values_may_have_been_sent


# -- the notice ------------------------------------------------------------------------------------


def test_the_notice_is_in_the_sidebar_with_the_session_idle_time() -> None:
    page = Page(HAPPY).open_file()
    expander = [e for e in page.at.sidebar.expander if e.label == "What is sent to OpenAI"]
    assert [m.value for m in expander[0].markdown] == [privacy_notice(60)]


def test_the_notice_names_what_is_sent_and_what_is_kept() -> None:
    notice = privacy_notice(60)
    for part in ("column names", "sample rows", "rows of each result", "60 minutes", "OpenAI"):
        assert part in notice
