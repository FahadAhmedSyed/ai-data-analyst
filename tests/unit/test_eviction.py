"""Removing the data of idle sessions (ADR 0014): the registry, the sweep and the sweeper thread."""

from __future__ import annotations

import gc
import logging
import threading
import weakref

import pytest

from analyst.session import eviction


class FakeSession:
    def __init__(self, idle: bool = True, *, fails: bool = False) -> None:
        self.idle = idle
        self.fails = fails
        self.checked = 0
        self.expired = False

    def expire_if_idle(self) -> bool:
        self.checked += 1
        if self.fails:
            raise RuntimeError("SECRET-IN-SESSION")
        if self.idle:
            self.expired = True
        return self.idle


@pytest.fixture(autouse=True)
def _clean_registry():
    before = list(eviction._sessions)
    eviction._sessions.clear()
    yield
    eviction.stop_sweeper()
    eviction._sessions.clear()
    eviction._sessions.update(before)


def test_an_idle_session_is_expired_and_counted() -> None:
    idle, busy = FakeSession(True), FakeSession(False)
    eviction.register(idle)
    eviction.register(busy)
    assert eviction.sweep() == 1
    assert idle.expired and not busy.expired
    assert idle.checked == busy.checked == 1


def test_a_sweep_with_nothing_registered_expires_nothing() -> None:
    assert eviction.sweep() == 0


def test_a_session_that_fails_does_not_stop_the_others_and_is_logged_by_type(caplog) -> None:
    caplog.set_level(logging.INFO)
    bad, good = FakeSession(fails=True), FakeSession(True)
    eviction.register(bad)
    eviction.register(good)
    assert eviction.sweep() == 1
    assert good.expired
    (record,) = [r for r in caplog.records if r.getMessage() == "sweep_failed"]
    assert record.error_type == "RuntimeError"
    assert "SECRET" not in " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)


def test_a_sweep_that_expired_sessions_says_how_many_and_one_that_did_not_is_silent(caplog) -> None:
    caplog.set_level(logging.INFO)
    busy, idle = FakeSession(False), FakeSession(True)  # held here: the registry holds none
    eviction.register(busy)
    eviction.sweep()
    assert "sessions_swept" not in [r.getMessage() for r in caplog.records]
    eviction.register(idle)
    eviction.sweep()
    (record,) = [r for r in caplog.records if r.getMessage() == "sessions_swept"]
    assert record.expired == 1


def test_registering_does_not_keep_a_session_alive() -> None:
    session = FakeSession()
    reference = weakref.ref(session)
    eviction.register(session)
    del session
    gc.collect()
    assert reference() is None
    assert eviction.sweep() == 0


def test_registering_twice_checks_a_session_once() -> None:
    session = FakeSession(False)
    eviction.register(session)
    eviction.register(session)
    eviction.sweep()
    assert session.checked == 1


def test_the_sweeper_sweeps_on_its_interval() -> None:
    swept = threading.Event()

    class Counting(FakeSession):
        def expire_if_idle(self) -> bool:
            swept.set()
            return False

    session = Counting()
    eviction.register(session)
    thread = eviction.start_sweeper(interval_s=0.01)
    assert thread.daemon
    assert thread.is_alive()
    assert swept.wait(timeout=5)


def test_starting_it_again_does_not_start_a_second_thread() -> None:
    first = eviction.start_sweeper(interval_s=60)
    assert eviction.start_sweeper(interval_s=60) is first
    assert sum(t.name == "session-sweeper" for t in threading.enumerate()) == 1


def test_it_can_be_stopped_and_started_again() -> None:
    first = eviction.start_sweeper(interval_s=60)
    eviction.stop_sweeper()
    assert not first.is_alive()
    second = eviction.start_sweeper(interval_s=60)
    assert second is not first and second.is_alive()


def test_stopping_with_no_sweeper_is_harmless() -> None:
    eviction.stop_sweeper()
    eviction.stop_sweeper()


def test_the_default_interval_is_a_minute() -> None:
    assert eviction.SWEEP_INTERVAL_S == 60
