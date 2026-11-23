"""The shared password (FR-10.1, ADR 0012).

A small group of testers share one password, set in ``APP_PASSWORD``. With none set the gate
is open, which is for running locally; a deployment must set it. The comparison takes the
same time whatever is typed, and a wrong guess waits a second before it is answered, which
slows a script that tries many on one connection (it does not stop one that opens many; the
spend cap on the OpenAI project is for that).
"""

from __future__ import annotations

import hmac
import time

import streamlit as st

from analyst.config.settings import Settings
from analyst.core.logging import get_logger

log = get_logger(__name__)

AUTHENTICATED_KEY = "authenticated"
FAILED_LOGIN_DELAY_S = 1.0
LOGIN_FAILED_MESSAGE = "That password isn't right."


def _wait(seconds: float) -> None:
    """Wait after a wrong guess. A function of its own so tests can skip the wait."""
    time.sleep(seconds)


def password_matches(entered: str, settings: Settings) -> bool:
    """True when ``entered`` is the configured password."""
    if settings.app_password is None:
        return False
    expected = settings.app_password.get_secret_value().encode()
    return hmac.compare_digest(entered.encode(), expected)


def passed(settings: Settings) -> bool:
    """True when the session may use the app; otherwise draws the password form."""
    if not settings.password_required or st.session_state.get(AUTHENTICATED_KEY):
        return True
    st.title("AI Data Analyst")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter", type="primary")
    if submitted:
        if password_matches(entered, settings):
            st.session_state[AUTHENTICATED_KEY] = True
            st.rerun()
        log.warning("login_failed")
        _wait(FAILED_LOGIN_DELAY_S)
        st.error(LOGIN_FAILED_MESSAGE)
    return False
