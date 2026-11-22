"""Streamlit entry point. Kept thin: all logic lives in the ``analyst`` package."""

from __future__ import annotations

import streamlit as st

from analyst import __version__
from analyst.config.settings import get_settings
from analyst.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("app")

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst")
st.caption(f"Version {__version__}")
st.info(
    "This is the project skeleton. Uploading data and asking questions arrive in the next tasks."
)
log.info("page_rendered", extra={"version": __version__})
