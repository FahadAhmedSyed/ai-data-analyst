"""The limits on a session: questions per window (FR-10.2), idle time, and separate sessions."""

from __future__ import annotations

from analyst.config.settings import Settings
from analyst.llm.fake import FakeLLMClient
from analyst.session import eviction
from analyst.session.display import escape_markdown
from analyst.session.state import AnalystSession
from tests.e2e.driver import OTHER, SALES, Page
from tests.e2e.test_app_flows import ASK, HAPPY


class Clock:
    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now


def seeded(*scripts, clock: Clock | None = None, **settings) -> AnalystSession:
    """A session with its own clock, which a test hands to the page before it runs."""
    queue = list(scripts)
    return AnalystSession(
        Settings(_env_file=None, **settings),
        client_factory=lambda: FakeLLMClient(queue.pop(0) if queue else []),
        clock=clock or Clock(),
    )


# -- questions per window --------------------------------------------------------------------------


def test_a_question_over_the_limit_is_not_run_and_says_when_to_try_again(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_QUESTIONS", "2")
    page = Page([*HAPPY, *HAPPY]).open_file().ask(ASK).ask(ASK)
    calls = len(page.clients[0].calls)
    page.ask("One more?")
    assert len(page.clients[0].calls) == calls  # nothing was sent to the model
    notice = page.answer().warning[0].value
    assert "limit of 2 questions in 10 minutes" in notice
    assert "try again in about" in notice
    assert page.questions()[-1] == escape_markdown("One more?")
    assert len(page.session.history) == 3
    assert page.session.history[-1].response is None


def test_a_refused_question_is_not_part_of_what_the_model_remembers(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_QUESTIONS", "1")
    page = Page(HAPPY).open_file().ask(ASK)
    turns = len(page.session.memory)
    page.ask("One more?")
    assert len(page.session.memory) == turns


def test_the_limit_ends_when_the_window_has_passed(monkeypatch) -> None:
    clock = Clock()
    session = seeded(HAPPY, [*HAPPY, *HAPPY], clock=clock, rate_limit_questions=1)
    page = Page(session=session).open_file().ask(ASK)
    page.ask("Too soon?")
    assert page.session.history[-1].response is None
    clock.now += 10 * 60
    page.ask(ASK)
    assert page.session.history[-1].response is not None


def test_replacing_the_dataset_does_not_give_a_fresh_allowance(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_QUESTIONS", "1")
    page = Page(HAPPY, HAPPY).open_file().ask(ASK)
    page.upload("other.csv", OTHER).load()
    page.at.sidebar.button(key="replace_yes").click().run()
    assert page.session.history == []
    page.ask(ASK)
    assert page.session.history[-1].response is None
    assert "limit of 1 question in 10 minutes" in page.answer().warning[0].value


def test_a_new_conversation_does_not_give_a_fresh_allowance_either(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_QUESTIONS", "1")
    page = Page(HAPPY).open_file().ask(ASK)
    page.at.sidebar.button[-1].click().run()
    page.ask(ASK)
    assert page.session.history[-1].response is None


def test_the_chat_stays_open_when_the_limit_is_reached(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_QUESTIONS", "1")
    page = Page(HAPPY).open_file().ask(ASK).ask(ASK)
    assert not page.at.chat_input[0].disabled
    assert not page.at.exception


# -- idle time -------------------------------------------------------------------------------------


def test_coming_back_after_the_idle_limit_finds_the_data_gone_and_says_so() -> None:
    clock = Clock()
    session = seeded(HAPPY, clock=clock, session_idle_minutes=30)
    page = Page(session=session).open_file().ask(ASK)
    clock.now += 31 * 60
    page.at.run()
    assert not page.session.has_dataset
    assert page.session.expired
    assert any("removed after 30 minutes without activity" in t for t in page.texts("info"))
    assert len(page.at.chat_message) == 0
    assert page.at.chat_input[0].disabled
    assert page.at.file_uploader[0].label == "Upload a CSV or Excel file"
    assert not page.session.db.is_loaded


def test_coming_back_inside_the_idle_limit_changes_nothing() -> None:
    clock = Clock()
    session = seeded(HAPPY, clock=clock, session_idle_minutes=30)
    page = Page(session=session).open_file().ask(ASK)
    clock.now += 29 * 60
    page.at.run()
    assert page.session.has_dataset
    assert len(page.at.chat_message) == 2
    assert not page.texts("info") or not any("removed after" in t for t in page.texts("info"))


def test_every_visit_restarts_the_idle_timer() -> None:
    clock = Clock()
    session = seeded(HAPPY, clock=clock, session_idle_minutes=30)
    page = Page(session=session).open_file()
    for _ in range(3):
        clock.now += 20 * 60
        page.at.run()
    assert page.session.has_dataset


def test_a_tab_left_open_is_emptied_by_the_sweeper_and_told_on_return() -> None:
    clock = Clock()
    session = seeded(HAPPY, clock=clock, session_idle_minutes=30)
    page = Page(session=session).open_file().ask(ASK)
    clock.now += 31 * 60
    assert eviction.sweep() == 1
    assert not session.db.is_loaded
    page.at.run()
    assert any("removed after 30 minutes" in t for t in page.texts("info"))
    assert len(page.at.chat_message) == 0


def test_the_notice_goes_when_a_new_file_is_loaded() -> None:
    clock = Clock()
    session = seeded(HAPPY, [], clock=clock, session_idle_minutes=30)
    page = Page(session=session).open_file()
    clock.now += 31 * 60
    page.at.run()
    page.open_file("sales.csv", SALES)
    assert page.session.has_dataset
    assert not any("removed after" in t for t in page.texts("info"))


def test_removing_the_data_does_not_hand_the_allowance_back() -> None:
    clock = Clock()
    session = seeded(
        HAPPY,
        HAPPY,
        clock=clock,
        session_idle_minutes=30,
        rate_limit_questions=1,
        rate_limit_window_s=3600,
    )
    page = Page(session=session).open_file().ask(ASK)
    clock.now += 31 * 60
    page.at.run()
    assert not page.session.has_dataset
    page.open_file().ask(ASK)
    assert page.session.history[-1].response is None  # the hour is not over


# -- separate sessions -----------------------------------------------------------------------------


def test_two_browser_sessions_share_nothing() -> None:
    one = Page(HAPPY).open_file("sales.csv", SALES)
    two = Page([]).open_file("other.csv", OTHER)
    one.ask(ASK)
    assert one.session.id != two.session.id
    assert one.session.db is not two.session.db
    assert one.session.dataset.profile.source_name == "sales.csv"
    assert two.session.dataset.profile.source_name == "other.csv"
    assert len(one.at.chat_message) == 2
    assert len(two.at.chat_message) == 0
    assert two.session.history == []
    assert "Revenue" not in two.page_text()
    two.at.run()
    assert len(two.at.chat_message) == 0


def test_one_sessions_limit_is_not_spent_by_another(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_QUESTIONS", "1")
    one = Page(HAPPY).open_file().ask(ASK)
    two = Page(HAPPY).open_file().ask(ASK)
    assert one.session.history[-1].response is not None
    assert two.session.history[-1].response is not None


def test_removing_one_sessions_data_leaves_the_other_alone() -> None:
    clock = Clock()
    gone = Page(session=seeded(HAPPY, clock=clock, session_idle_minutes=30)).open_file()
    kept = Page(session=seeded(HAPPY, clock=clock, session_idle_minutes=30)).open_file()
    clock.now += 20 * 60
    kept.at.run()  # a visit by one of the two
    clock.now += 15 * 60
    assert eviction.sweep() == 1
    assert not gone.session.db.is_loaded
    assert kept.session.db.is_loaded
