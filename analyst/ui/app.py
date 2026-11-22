"""The page: ties the password gate, sidebar, dataset view and chat together (T1.18).

``run`` is what ``app.py`` calls on every Streamlit run, so it keeps no state of its own: the
session is in ``st.session_state`` and everything else is drawn from it.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from analyst import __version__
from analyst.config.settings import get_settings
from analyst.core.logging import configure_logging
from analyst.llm.client import LLMClient
from analyst.session.eviction import start_sweeper
from analyst.ui import chat, dataset_view, gate, sidebar
from analyst.ui.notices import WELCOME
from analyst.ui.session_store import current_session


def run(client_factory: Callable[[], LLMClient] | None = None) -> None:
    """Draw the app for this browser session.

    Args:
        client_factory: Makes the model client for a dataset. ``None`` uses OpenAI with the
            application settings; tests pass a factory that returns a scripted fake.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    st.set_page_config(
        page_title="AI Data Analyst", layout="wide", initial_sidebar_state="expanded"
    )
    if not gate.passed(settings):
        st.stop()

    session = current_session(client_factory)
    start_sweeper()
    session.expire_if_idle()  # before touching: coming back after the limit finds the data gone
    session.touch()

    session.schema_only = sidebar.schema_only_chosen()
    sidebar.render_data(session)
    st.title("AI Data Analyst")
    if client_factory is None and not settings.llm_configured:
        st.warning(
            "Questions are switched off: the OpenAI key or the model names are not set. "
            "You can still upload a file and look at it."
        )
    if session.expired:
        st.info(
            f"Your data was removed after {session.settings.session_idle_minutes} minutes without "
            "activity. Please upload the file again."
        )
    question = chat.chat_box(session)
    if session.has_dataset:
        dataset_view.render(session, expanded=not (session.history or question))
        chat.render(session, question)
    else:
        st.markdown(WELCOME)
    sidebar.render_controls(session)
    st.sidebar.caption(f"Version {__version__}")
