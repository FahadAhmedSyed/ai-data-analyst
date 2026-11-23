"""The password gate's pieces that need no page (FR-10.1). The form is in ``tests/e2e``."""

from __future__ import annotations

from analyst.ui import gate


def test_a_wrong_guess_waits_before_it_is_answered(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(gate.time, "sleep", slept.append)
    gate._wait(gate.FAILED_LOGIN_DELAY_S)
    assert slept == [gate.FAILED_LOGIN_DELAY_S]


def test_the_wait_is_long_enough_to_slow_guessing_and_short_enough_not_to_annoy() -> None:
    assert 0.5 <= gate.FAILED_LOGIN_DELAY_S <= 3
