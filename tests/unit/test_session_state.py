"""One browser session (T1.18): loading, asking, rate limiting, resetting and idle expiry."""

from __future__ import annotations

import gc
import logging
import weakref

import pytest

from analyst.config.settings import Settings
from analyst.core.errors import IngestionError
from analyst.core.models import Usage
from analyst.llm.fake import FakeLLMClient
from analyst.session.dataset import ReadOptions
from analyst.session.state import AnalystSession, ChatEntry
from tests.unit.test_analyst import explains
from tests.unit.test_generator import answer, asks

SALES = b"Region,Revenue\nNorth,10\nSouth,20\nNorth,30\n"
OTHER = b"Name,Score\nAnn,5\nBo,7\n"
SAID = "Revenue is shown for each region."
HAPPY = (("generate", answer()), ("explain", explains(SAID)))


class Clock:
    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def make(*scripts, clock=None, **kw) -> tuple[AnalystSession, list[FakeLLMClient]]:
    """A session whose n-th dataset is answered from the n-th script."""
    queue = list(scripts)
    made: list[FakeLLMClient] = []

    def factory() -> FakeLLMClient:
        client = FakeLLMClient(queue.pop(0) if queue else [])
        made.append(client)
        return client

    session = AnalystSession(
        settings(**kw), client_factory=factory, **({"clock": clock} if clock else {})
    )
    return session, made


# -- loading --------------------------------------------------------------------------------------


def test_a_new_session_has_no_dataset() -> None:
    session, _ = make()
    assert not session.has_dataset
    assert session.dataset is None
    assert session.history == []
    assert session.load_failure is None
    assert session.schema_only is False
    assert session.expired is False


def test_loading_a_file_makes_a_dataset_and_a_client_for_it() -> None:
    session, made = make(list(HAPPY))
    assert session.load(SALES, "sales.csv", key="k1") is True
    assert session.has_dataset
    assert session.source_key == "k1"
    assert session.dataset is not None and session.dataset.profile.row_count == 3
    assert len(made) == 1
    assert session.load_failure is None


def test_the_options_are_passed_to_the_loader() -> None:
    session, _ = make()
    assert session.load(b"A|B\n1|2\n", "p.csv", ReadOptions(delimiter="|"))
    assert session.dataset.profile.column_count == 2  # type: ignore[union-attr]


def test_a_file_that_is_refused_says_why_and_leaves_no_dataset() -> None:
    session, made = make()
    assert session.load(b"", "empty.csv", key="bad") is False
    assert not session.has_dataset
    assert session.load_failure is not None
    assert session.load_failure.key == "bad"
    assert session.load_failure.message == IngestionError("empty").user_message
    assert made == []  # no model client for a dataset that does not exist


def test_a_refused_file_leaves_the_dataset_and_conversation_that_were_there() -> None:
    session, _ = make(list(HAPPY))
    session.load(SALES, "sales.csv", key="a")
    session.ask("Total revenue by region?")
    assert session.load(b"", "empty.csv", key="b") is False
    assert session.has_dataset
    assert session.source_key == "a"
    assert len(session.history) == 1
    assert session.load_failure is not None and session.load_failure.key == "b"
    assert session.ask("again?").response is not None  # still usable


def test_a_failed_load_that_lost_the_table_forgets_everything_that_used_it(monkeypatch) -> None:
    session, _ = make(list(HAPPY))
    session.load(SALES, "sales.csv", key="a")
    session.ask("Total revenue by region?")

    from analyst.session import dataset as dataset_module

    def broken(_db):
        raise RuntimeError("profiler bug with SECRET")

    monkeypatch.setattr(dataset_module, "profile_database", broken)
    assert session.load(OTHER, "other.csv", key="b") is False
    assert not session.has_dataset
    assert session.dataset is None
    assert session.history == []
    assert session.memory.turns == ()
    assert session.source_key is None
    assert session.load_failure is not None
    assert session.load_failure.message == "This file couldn't be loaded."


def test_a_bug_in_loading_shows_a_plain_message_and_logs_only_its_type(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    session, _ = make()
    from analyst.session import dataset as dataset_module

    def broken(*args, **kwargs):
        raise RuntimeError("SECRET-FILE-NAME confidential.csv")

    monkeypatch.setattr(dataset_module, "normalize", broken)
    assert session.load(SALES, "confidential.csv") is False
    assert session.load_failure.message == "This file couldn't be loaded."  # type: ignore[union-attr]
    (record,) = [r for r in caplog.records if r.getMessage() == "file_failed"]
    assert record.error_type == "RuntimeError"
    assert record.session == session.id
    text = " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)
    assert "SECRET" not in text and "confidential" not in text


def test_a_refused_file_logs_its_category_and_not_its_name(caplog) -> None:
    caplog.set_level(logging.INFO)
    session, _ = make()
    session.load(b"data", "payroll-2024.pdf")
    (record,) = [r for r in caplog.records if r.getMessage() == "file_rejected"]
    assert record.category == "unsupported_type"
    assert "payroll" not in " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)


def test_a_successful_load_clears_an_earlier_failure() -> None:
    session, _ = make(list(HAPPY))
    session.load(b"", "empty.csv", key="bad")
    session.load(SALES, "sales.csv", key="good")
    assert session.load_failure is None


# -- replacing ------------------------------------------------------------------------------------


def test_replacing_the_dataset_starts_a_new_conversation_with_a_new_client() -> None:
    session, made = make(list(HAPPY), list(HAPPY))
    session.load(SALES, "sales.csv", key="a")
    first_memory = session.memory
    session.ask("Total revenue by region?")
    session.load(OTHER, "other.csv", key="b")
    assert session.history == []
    assert session.memory is not first_memory
    assert session.memory.turns == ()
    assert session.source_key == "b"
    assert len(made) == 2
    assert session.dataset.profile.table_name == "other"  # type: ignore[union-attr]


def test_replacing_the_dataset_does_not_give_a_fresh_allowance_of_questions() -> None:
    session, _ = make(list(HAPPY), list(HAPPY), rate_limit_questions=1)
    session.load(SALES, "sales.csv")
    assert session.ask("Total revenue by region?").response is not None
    session.load(SALES, "sales.csv")
    assert session.ask("Total revenue by region?").response is None


def test_the_schema_only_choice_carries_over_to_the_next_dataset() -> None:
    session, made = make(list(HAPPY), list(HAPPY))
    session.schema_only = True
    session.load(SALES, "sales.csv")
    session.load(SALES, "sales.csv")
    session.ask("Total revenue by region?")
    prompt = made[1].calls[0].messages[1].content
    assert "region" in prompt  # the schema is there
    assert "North" not in prompt  # and no value from the file


# -- asking ---------------------------------------------------------------------------------------


def test_a_question_is_answered_and_added_to_the_conversation() -> None:
    session, made = make(list(HAPPY))
    session.load(SALES, "sales.csv")
    entry = session.ask("  Total revenue by region?  ")
    assert isinstance(entry, ChatEntry)
    assert entry.question == "Total revenue by region?"
    assert entry.notice == ""
    assert entry.response is not None and entry.response.status == "answered"
    assert entry.response.message == SAID
    assert session.history == [entry]
    assert len(session.memory) == 1
    made[0].assert_exhausted()


def test_progress_is_passed_on() -> None:
    session, _ = make(list(HAPPY))
    session.load(SALES, "sales.csv")
    told: list[str] = []
    session.ask("Total revenue by region?", told.append)
    assert told == ["understanding", "running", "explaining"]


def test_a_clarification_is_an_entry_like_any_other() -> None:
    session, _ = make([("generate", asks())])
    session.load(SALES, "sales.csv")
    entry = session.ask("Which is best?")
    assert entry.response is not None and entry.response.status == "clarification"


def test_asking_with_no_dataset_is_a_programming_error() -> None:
    session, _ = make()
    with pytest.raises(RuntimeError, match="No dataset"):
        session.ask("hello?")
    assert session.history == []


def test_the_schema_only_switch_applies_to_the_next_question_and_back() -> None:
    session, made = make([*HAPPY, *HAPPY])
    session.load(SALES, "sales.csv")
    session.ask("Total revenue by region?")  # sample rows are sent
    session.schema_only = True
    session.ask("Total revenue by region?")
    session.schema_only = False
    first, second = made[0].calls[0], made[0].calls[2]
    assert "North" in first.messages[1].content
    assert "North" not in second.messages[1].content.split("<<<QUESTION")[0].split("<<<HISTORY")[0]


def test_the_question_that_is_asked_is_logged_without_its_text(caplog) -> None:
    caplog.set_level(logging.INFO)
    session, _ = make(list(HAPPY))
    session.load(SALES, "sales.csv")
    session.ask("ZEBRA question about revenue?")
    (record,) = [r for r in caplog.records if r.getMessage() == "question_asked"]
    assert record.session == session.id
    assert record.status == "answered"
    assert "ZEBRA" not in " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)


# -- the rate limit -------------------------------------------------------------------------------


def test_a_question_over_the_limit_is_not_run_and_says_when_to_try_again() -> None:
    clock = Clock()
    session, made = make(list(HAPPY), clock=clock, rate_limit_questions=1)
    session.load(SALES, "sales.csv")
    session.ask("Total revenue by region?")
    entry = session.ask("And again?")
    assert entry.response is None
    assert "limit of 1 question in 10 minutes" in entry.notice
    assert entry.question == "And again?"
    assert session.history[-1] is entry
    assert len(made[0].calls) == 2  # the first question's two calls, none for the refused one
    assert len(session.memory) == 1  # a question that was not run is not part of the conversation


def test_the_limit_ends_when_the_window_has_passed() -> None:
    clock = Clock()
    session, _ = make([*HAPPY, *HAPPY], clock=clock, rate_limit_questions=1, rate_limit_window_s=60)
    session.load(SALES, "sales.csv")
    assert session.ask("Total revenue by region?").response is not None
    assert session.ask("Total revenue by region?").response is None
    clock.now += 60
    assert session.ask("Total revenue by region?").response is not None


def test_a_refused_question_is_logged_without_its_text(caplog) -> None:
    caplog.set_level(logging.INFO)
    session, _ = make(list(HAPPY), rate_limit_questions=1)
    session.load(SALES, "sales.csv")
    session.ask("first?")
    session.ask("ZEBRA second?")
    assert [r.getMessage() for r in caplog.records].count("rate_limited") == 1
    assert "ZEBRA" not in " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)


def test_the_limits_come_from_the_settings() -> None:
    session, _ = make(rate_limit_questions=7, rate_limit_window_s=120)
    assert session.limiter.remaining() == 7


# -- the download ---------------------------------------------------------------------------------


def test_the_download_is_the_result_as_csv_and_is_built_once() -> None:
    session, _ = make(list(HAPPY))
    session.load(SALES, "sales.csv")
    entry = session.ask("Total revenue by region?")
    data = entry.download()
    assert data is not None and data.startswith(b"\xef\xbb\xbf")
    assert data.decode("utf-8-sig").splitlines()[0] == "region,total_revenue"
    assert entry.download() is data


def test_there_is_nothing_to_download_without_a_result() -> None:
    session, _ = make([("generate", asks())], rate_limit_questions=1)
    session.load(SALES, "sales.csv")
    assert session.ask("Which is best?").download() is None  # a clarification
    assert session.ask("over the limit").download() is None  # not run


def test_what_an_entry_holds_does_not_appear_in_its_repr() -> None:
    """A repr reaches logs and error messages, so it must not carry rows from the file."""
    session, _ = make(list(HAPPY))
    session.load(SALES, "sales.csv")
    entry = session.ask("Total revenue by region?")
    entry.download()
    entry.drawn["table"] = "CELL-FROM-THE-FILE"
    text = repr(entry)
    for secret in ("North", "South", "40", "CELL-FROM-THE-FILE", "total_revenue", SAID):
        assert secret not in text


def test_an_entry_made_by_hand_has_no_response_and_was_not_schema_only() -> None:
    entry = ChatEntry("a question")
    assert entry.response is None
    assert entry.notice == ""
    assert entry.schema_only is False
    assert entry.drawn == {}
    assert entry.download() is None


# -- what the conversation may already have sent --------------------------------------------------


def test_values_may_have_been_sent_only_by_a_question_that_ran_with_the_mode_off() -> None:
    session, _ = make([*HAPPY, *HAPPY, *HAPPY], rate_limit_questions=3)
    session.load(SALES, "sales.csv")
    assert session.values_may_have_been_sent is False  # nothing asked

    session.schema_only = True
    session.ask("Total revenue by region?")
    assert session.values_may_have_been_sent is False  # asked with the mode on

    session.schema_only = False
    session.ask("Total revenue by region?")
    assert session.values_may_have_been_sent is True  # a question ran with it off

    session.reset_conversation()
    assert session.values_may_have_been_sent is False  # the history is gone


def test_a_question_that_was_refused_sent_nothing() -> None:
    session, _ = make([*HAPPY], rate_limit_questions=1)
    session.load(SALES, "sales.csv")
    session.schema_only = True
    session.ask("Total revenue by region?")
    session.schema_only = False
    entry = session.ask("one more?")  # over the limit: not run, so nothing was sent
    assert entry.response is None and entry.schema_only is False
    assert session.values_may_have_been_sent is False


# -- resetting ------------------------------------------------------------------------------------


def test_a_new_conversation_keeps_the_data_and_forgets_the_questions() -> None:
    session, _ = make([*HAPPY, *HAPPY])
    session.load(SALES, "sales.csv")
    session.ask("Total revenue by region?")
    memory = session.memory
    session.reset_conversation()
    assert session.history == []
    assert session.memory is memory  # the analyst keeps using the same object
    assert memory.turns == () and memory.carried_assumptions() == ()
    assert session.has_dataset
    assert session.ask("Total revenue by region?").response is not None


def test_a_new_conversation_does_not_reset_the_rate_limit() -> None:
    session, _ = make(list(HAPPY), rate_limit_questions=1)
    session.load(SALES, "sales.csv")
    session.ask("Total revenue by region?")
    session.reset_conversation()
    assert session.ask("Total revenue by region?").response is None


def test_clearing_removes_the_data_and_the_conversation_and_keeps_the_limit() -> None:
    session, _ = make(list(HAPPY), rate_limit_questions=2)
    session.load(SALES, "sales.csv", key="a")
    session.ask("Total revenue by region?")
    session.clear()
    assert not session.has_dataset
    assert not session.db.is_loaded
    assert session.history == []
    assert session.memory.turns == ()
    assert session.source_key is None
    assert session.limiter.remaining() == 1


# -- idle expiry ----------------------------------------------------------------------------------


def test_a_session_idle_for_longer_than_the_limit_loses_its_data() -> None:
    clock = Clock()
    session, _ = make(list(HAPPY), clock=clock, session_idle_minutes=60)
    session.load(SALES, "sales.csv")
    session.ask("Total revenue by region?")
    clock.now += 60 * 60 + 1
    assert session.expire_if_idle() is True
    assert session.expired is True
    assert not session.has_dataset
    assert not session.db.is_loaded
    assert session.history == []


def test_a_session_at_exactly_the_limit_is_kept() -> None:
    clock = Clock()
    session, _ = make(clock=clock, session_idle_minutes=60)
    session.load(SALES, "sales.csv")
    clock.now += 60 * 60
    assert session.expire_if_idle() is False
    assert session.has_dataset


def test_activity_resets_the_idle_timer() -> None:
    clock = Clock()
    session, _ = make(clock=clock, session_idle_minutes=60)
    session.load(SALES, "sales.csv")
    clock.now += 50 * 60
    session.touch()
    clock.now += 50 * 60
    assert session.expire_if_idle() is False


def test_asking_a_question_is_activity() -> None:
    clock = Clock()
    session, _ = make(list(HAPPY), clock=clock, session_idle_minutes=60)
    session.load(SALES, "sales.csv")
    clock.now += 50 * 60
    session.ask("Total revenue by region?")
    clock.now += 50 * 60
    assert session.expire_if_idle() is False


def test_a_session_with_no_dataset_has_nothing_to_expire() -> None:
    clock = Clock()
    session, _ = make(clock=clock, session_idle_minutes=60)
    clock.now += 10 * 60 * 60
    assert session.expire_if_idle() is False
    assert session.expired is False


def test_expiry_is_logged_and_a_new_file_ends_it(caplog) -> None:
    caplog.set_level(logging.INFO)
    clock = Clock()
    session, _ = make(clock=clock, session_idle_minutes=60)
    session.load(SALES, "sales.csv")
    clock.now += 61 * 60
    session.expire_if_idle()
    assert [r.getMessage() for r in caplog.records].count("session_expired") == 1
    assert session.load(SALES, "sales.csv")
    assert session.expired is False


# -- sessions are separate ------------------------------------------------------------------------


def test_two_sessions_share_nothing() -> None:
    one, _ = make(list(HAPPY), rate_limit_questions=1)
    two, _ = make(list(HAPPY), rate_limit_questions=1)
    one.load(SALES, "sales.csv")
    two.load(OTHER, "other.csv")
    one.ask("Total revenue by region?")
    assert one.id != two.id
    assert one.db is not two.db
    assert one.memory is not two.memory
    assert one.limiter is not two.limiter
    assert one.dataset.profile.table_name == "sales"  # type: ignore[union-attr]
    assert two.dataset.profile.table_name == "other"  # type: ignore[union-attr]
    assert two.history == [] and len(two.memory) == 0
    assert two.ask("Total revenue by region?").response is not None  # not charged to one


def test_the_session_label_is_short_and_random() -> None:
    labels = {make()[0].id for _ in range(20)}
    assert len(labels) == 20
    assert all(len(label) == 8 for label in labels)


def test_the_default_client_is_the_openai_one_and_is_not_made_until_a_dataset_is_loaded() -> None:
    session = AnalystSession(settings())
    assert not session.has_dataset
    session.load(SALES, "sales.csv")
    entry = session.ask("hello?")  # not configured: a failed answer, not an exception
    assert entry.response is not None and entry.response.status == "failed"
    assert "isn't set up yet" in entry.response.message
    assert entry.response.usage == Usage(latency_ms=entry.response.usage.latency_ms)


def test_a_session_does_not_keep_itself_alive_for_the_sweeper() -> None:
    from analyst.session import eviction

    session, _ = make()
    reference = weakref.ref(session)
    assert session in eviction._sessions
    del session
    gc.collect()
    assert reference() is None
