"""Shared test configuration."""

from __future__ import annotations

import pytest

from analyst.config.settings import Settings, get_settings

_ENV_VARS = [name.upper() for name in Settings.model_fields]


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Keep the developer's environment and .env file out of every test."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in an empty temporary directory
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
