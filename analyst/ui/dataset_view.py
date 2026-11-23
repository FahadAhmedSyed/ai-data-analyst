"""The loaded dataset: preview, columns and data quality (FR-3.1 to FR-3.7).

All of it is drawn from the ``LoadedDataset`` made when the file was loaded, so a rerun
(every click and every question) queries nothing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analyst.session.display import count_text, escape_markdown, profile_tables
from analyst.session.state import AnalystSession


def render(session: AnalystSession, *, expanded: bool) -> None:
    """Draw the three tabs, inside a panel the caller opens until the first question."""
    dataset = session.dataset
    if dataset is None:
        return
    profile = dataset.profile
    with st.expander("Your data: preview, columns and quality", expanded=expanded):
        preview, columns, quality = st.tabs(["Preview", "Columns", "Data quality"])
        with preview:
            shown, total = len(dataset.preview), profile.row_count
            which = (
                f"The first {shown:,} of {total:,} rows"
                if shown < total
                else f"All {count_text(total, 'row')}"
            )
            st.caption(
                f"{which}, as the app reads them. " + escape_markdown(dataset.read.describe())
            )
            st.dataframe(dataset.preview, hide_index=True)
        with columns:
            _columns(session)
        with quality:
            st.caption("The app reports these notes and never changes your data.")
            if dataset.issues:
                st.markdown("\n".join(f"- {escape_markdown(i.message)}" for i in dataset.issues))
            else:
                st.success("No missing values, duplicate rows or other problems were found.")


def _table(title: str, frame: pd.DataFrame) -> None:
    if not frame.empty:
        st.markdown(f"**{title}**")
        st.dataframe(frame, hide_index=True)


def _columns(session: AnalystSession) -> None:
    dataset = session.dataset
    if dataset is None:
        return
    tables = profile_tables(dataset.profile)
    _table("All columns", tables.overview)
    _table("Numbers", tables.numbers)
    _table("Text and true/false", tables.text)
    _table("Dates", tables.dates)
