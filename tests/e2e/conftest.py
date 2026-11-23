"""Fixtures for the interface tests."""

from __future__ import annotations

import pytest

from analyst.config.settings import get_settings
from analyst.session.eviction import stop_sweeper


@pytest.fixture(autouse=True)
def _no_sweeper_left_running():
    """The app starts the idle-session sweeper; end it with each test."""
    yield
    stop_sweeper()


@pytest.fixture(autouse=True)
def _no_login_delay(monkeypatch: pytest.MonkeyPatch):
    """A wrong password waits a second in the real app; tests record the wait and skip it."""
    waits: list[float] = []
    monkeypatch.setattr("analyst.ui.gate._wait", waits.append)
    return waits


@pytest.fixture(autouse=True)
def _fresh_settings():
    """The settings are cached; a test that sets an environment variable must not leak it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
