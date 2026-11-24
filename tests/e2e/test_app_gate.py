"""The shared password (FR-10.1, ADR 0012)."""

from __future__ import annotations

import json

import pytest

from analyst.config.settings import Settings, get_settings
from analyst.ui import gate
from tests.e2e.driver import Page

PASSWORD = "s3cret-Pässword"  # noqa: S105 - a test value


@pytest.fixture
def password(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("APP_PASSWORD", PASSWORD)
    get_settings.cache_clear()
    return PASSWORD


def enter(page: Page, text: str) -> Page:
    page.at.text_input[0].input(text)
    page.at.button[0].click().run()
    return page


# -- the form -------------------------------------------------------------------------------------


def test_with_a_password_set_the_first_page_is_only_the_password_form(password) -> None:
    page = Page()
    assert page.at.title[0].value == "AI Data Analyst"
    assert page.at.text_input[0].label == "Password"
    assert page.at.text_input[0].proto.type == 1  # a password field: the characters are hidden
    assert page.at.button[0].label == "Enter"
    assert len(page.at.file_uploader) == 0
    assert len(page.at.chat_input) == 0
    assert len(page.at.sidebar.button) == 0
    assert not page.at.markdown  # not even the welcome text


def test_a_wrong_password_is_refused_and_slowed(password, _no_login_delay) -> None:
    page = enter(Page(), "wrong")
    assert page.at.error[0].value == "That password isn't right."
    assert len(page.at.file_uploader) == 0
    assert "authenticated" not in page.at.session_state
    assert _no_login_delay == [gate.FAILED_LOGIN_DELAY_S]
    assert gate.FAILED_LOGIN_DELAY_S == 1.0


def test_an_empty_password_is_refused(password) -> None:
    page = enter(Page(), "")
    assert page.at.error[0].value == "That password isn't right."


def test_the_right_password_opens_the_app(password) -> None:
    page = enter(Page(), PASSWORD)
    assert len(page.at.file_uploader) == 1
    assert len(page.at.text_input) == 0
    assert page.at.session_state["authenticated"] is True
    assert not page.at.error


def test_the_session_stays_open_on_later_runs(password) -> None:
    page = enter(Page(), PASSWORD)
    page.at.run()
    page.at.run()
    assert len(page.at.file_uploader) == 1
    assert len(page.at.text_input) == 0


def test_a_correct_password_with_extra_characters_is_refused(password) -> None:
    assert enter(Page(), PASSWORD + " ").at.error
    assert enter(Page(), PASSWORD.upper()).at.error


def test_another_browser_session_is_not_let_in_by_the_first(password) -> None:
    first = enter(Page(), PASSWORD)
    second = Page()
    assert len(first.at.file_uploader) == 1
    assert len(second.at.file_uploader) == 0
    assert len(second.at.text_input) == 1


def log_lines(capsys) -> list[dict]:
    """The structured log records the app wrote to its stream (``configure_logging`` replaces
    the handlers on every run, so pytest's own capture never sees them)."""
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_the_password_is_never_shown_or_logged(password, capsys) -> None:
    page = enter(Page(), "guess-123")
    enter(page, PASSWORD)
    written = capsys.readouterr().out
    assert written  # the app did log something, so the check below looks at real output
    assert PASSWORD not in written
    assert "guess-123" not in written
    assert PASSWORD not in page.page_text()
    assert "guess-123" not in page.page_text()


def test_a_failed_attempt_is_logged_without_what_was_typed(password, capsys) -> None:
    enter(Page(), "guess-123")
    records = log_lines(capsys)
    (failed,) = [r for r in records if r["event"] == "login_failed"]
    assert failed["level"] == "WARNING"
    assert "guess-123" not in json.dumps(records)


def test_no_password_set_means_no_gate(monkeypatch) -> None:
    monkeypatch.setenv("APP_PASSWORD", "")
    get_settings.cache_clear()
    page = Page()
    assert len(page.at.file_uploader) == 1
    assert len(page.at.text_input) == 0


# -- a deployment with no password stays closed ---------------------------------------------------


@pytest.fixture
def deployed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUIRE_GATE", "true")
    get_settings.cache_clear()


def test_a_deployment_with_no_password_is_closed_not_open(deployed, monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    page = Page()
    assert page.at.error[0].value == gate.CLOSED_MESSAGE
    assert len(page.at.text_input) == 0  # no form to guess at: there is nothing to enter
    assert len(page.at.file_uploader) == 0
    assert len(page.at.chat_input) == 0
    assert not page.at.markdown


def test_an_empty_password_in_a_deployment_is_closed_too(deployed, monkeypatch) -> None:
    monkeypatch.setenv("APP_PASSWORD", "")
    page = Page()
    assert page.at.error[0].value == gate.CLOSED_MESSAGE
    assert len(page.at.file_uploader) == 0


def test_a_deployment_with_a_password_shows_the_form_as_before(deployed, password) -> None:
    page = Page()
    assert page.at.text_input[0].label == "Password"
    assert len(page.at.error) == 0
    enter(page, PASSWORD)
    assert len(page.at.file_uploader) == 1


def test_the_closed_message_asks_for_the_setting_and_gives_away_nothing_else() -> None:
    assert "APP_PASSWORD" in gate.CLOSED_MESSAGE
    assert "OPENAI" not in gate.CLOSED_MESSAGE


def test_the_closed_state_is_logged_as_an_error(deployed, monkeypatch, capsys) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    Page()
    (closed,) = [r for r in log_lines(capsys) if r["event"] == "password_required_but_not_set"]
    assert closed["level"] == "ERROR"


# -- the comparison -------------------------------------------------------------------------------


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.mark.parametrize(
    ("entered", "expected"),
    [
        (PASSWORD, True),
        ("", False),
        ("s3cret", False),
        (PASSWORD + "x", False),
        (PASSWORD.lower(), False),
        (" " + PASSWORD, False),
    ],
)
def test_only_the_exact_password_matches(entered: str, expected: bool) -> None:
    assert gate.password_matches(entered, settings(app_password=PASSWORD)) is expected


def test_nothing_matches_when_no_password_is_set() -> None:
    assert gate.password_matches("", settings()) is False
    assert gate.password_matches("anything", settings()) is False


def test_the_comparison_is_the_constant_time_one(monkeypatch) -> None:
    calls: list[tuple[bytes, bytes]] = []

    def recording(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return a == b

    monkeypatch.setattr(gate.hmac, "compare_digest", recording)
    configured = settings(app_password="y")  # noqa: S106 - a test value
    assert gate.password_matches("x", configured) is False
    assert calls == [(b"x", b"y")]


def test_non_ascii_passwords_work() -> None:
    configured = settings(app_password="pässwörd-日本")  # noqa: S106 - a test value
    assert gate.password_matches("pässwörd-日本", configured)
