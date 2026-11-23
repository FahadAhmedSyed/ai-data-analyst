from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from analyst import __version__
from analyst.session import eviction
from tests.e2e.driver import Page

APP = str(Path(__file__).resolve().parents[2] / "app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "AI Data Analyst"
    assert "Upload" in at.markdown[0].value


def test_the_app_says_questions_are_off_when_no_key_is_set():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "Questions are switched off" in at.warning[0].value


def test_the_app_does_not_say_questions_are_off_when_the_key_and_models_are_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL_GENERATE", "model-a")
    monkeypatch.setenv("LLM_MODEL_EXPLAIN", "model-b")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert len(at.warning) == 0


def test_a_page_given_its_own_model_never_says_questions_are_off():
    """The tests (and anything that supplies a client) have no key, and questions work."""
    assert len(Page().at.warning) == 0


def test_the_page_is_wide_and_titled_and_shows_the_version(monkeypatch):
    calls: list[dict] = []
    real = st.set_page_config

    def recording(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(st, "set_page_config", recording)
    page = Page()
    assert calls[0]["page_title"] == "AI Data Analyst"
    assert calls[0]["layout"] == "wide"
    assert f"Version {__version__}" in [c.value for c in page.at.sidebar.caption]


def test_running_the_page_starts_the_idle_sweeper():
    eviction.stop_sweeper()
    Page()
    thread = eviction._sweeper
    assert thread is not None and thread.is_alive() and thread.daemon
