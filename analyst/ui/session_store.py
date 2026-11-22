"""Where a browser session's state lives: ``st.session_state`` (architecture section 3)."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from analyst.core.logging import get_logger
from analyst.llm.client import LLMClient
from analyst.session.state import AnalystSession

log = get_logger(__name__)

SESSION_KEY = "analyst_session"
UPLOAD_COUNTER_KEY = "upload_counter"


def current_session(client_factory: Callable[[], LLMClient] | None = None) -> AnalystSession:
    """This browser session's ``AnalystSession``, created on its first run."""
    session = st.session_state.get(SESSION_KEY)
    if session is None:
        session = AnalystSession(client_factory=client_factory)
        st.session_state[SESSION_KEY] = session
        log.info("session_started", extra={"session": session.id})
    return session


def upload_widget_key() -> str:
    """The key of the file uploader, which changes to empty it."""
    return f"upload-{st.session_state.get(UPLOAD_COUNTER_KEY, 0)}"


def empty_uploader() -> None:
    """Give the file uploader a new key, so the browser shows it empty on the next run."""
    st.session_state[UPLOAD_COUNTER_KEY] = st.session_state.get(UPLOAD_COUNTER_KEY, 0) + 1
