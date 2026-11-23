"""The sidebar: choosing a file, how to read it, privacy, and starting over.

The flow for a file (FR-1.1, FR-1.2, FR-1.4, FR-7.4, FR-10.3):

1. The user picks a file. It is *staged*: Streamlit holds its bytes while the choices are made.
2. An Excel file offers its sheets; any file offers the separator and header row, with
   detection as the default. The privacy notice is beside the button.
3. The button loads it. When a conversation exists, replacing the dataset first asks, because
   it clears the conversation.
4. After a successful load the bytes are released from Streamlit's memory and the uploader
   is emptied, so the file lives only as the session's table (privacy.md, section 2).
   A file that fails to load stays staged, so another setting can be tried.
"""

from __future__ import annotations

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.uploaded_file_manager import UploadedFile

from analyst.core.errors import IngestionError
from analyst.core.logging import get_logger
from analyst.ingestion.excel_loader import SheetInfo, list_sheets
from analyst.ingestion.file_loader import SUPPORTED_EXTENSIONS, file_kind
from analyst.ingestion.schema_card import clean_text
from analyst.session.dataset import ReadOptions
from analyst.session.display import count_text, escape_markdown
from analyst.session.state import AnalystSession
from analyst.ui.notices import privacy_notice
from analyst.ui.session_store import empty_uploader, upload_widget_key

log = get_logger(__name__)

PENDING_KEY = "replace_pending"
SHEETS_KEY = "staged_sheets"
SCHEMA_ONLY_KEY = "schema_only"
MAX_HEADER_ROW = 1_000

DELIMITER_CHOICES: dict[str | None, str] = {
    None: "Detect automatically",
    ",": "Comma",
    ";": "Semicolon",
    "\t": "Tab",
    "|": "Pipe",
}
HEADER_CHOICES = {
    "auto": "Detect automatically",
    "row": "Choose the row",
    "none": "The file has no header",
}


def render_data(session: AnalystSession) -> None:
    """Draw the top of the sidebar, the file, and act on what the user did in it.

    This runs before the page is drawn, because loading a file changes the page.
    """
    settings = session.settings
    with st.sidebar:
        st.header("Your data")
        if session.has_dataset and session.dataset is not None:
            profile = session.dataset.profile
            where = f" ({profile.sheet_name})" if profile.sheet_name else ""
            name = escape_markdown(clean_text(profile.source_name))
            st.markdown(f"**{name}**{escape_markdown(where)}")
            st.caption(
                f"{count_text(profile.row_count, 'row')}, "
                f"{count_text(profile.column_count, 'column')}"
            )
        staged = st.file_uploader(
            "Replace the file" if session.has_dataset else "Upload a CSV or Excel file",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            key=upload_widget_key(),
            max_upload_size=settings.max_upload_mb,
            help=f"CSV, TXT, XLSX or XLS, up to {settings.max_upload_mb} MB.",
        )
        if staged is not None:
            _staged_file(session, staged)
        if staged is not None or not session.has_dataset:
            st.caption(privacy_notice(settings.session_idle_minutes))


def schema_only_chosen() -> bool:
    """The state of the schema-only switch, known before the switch is drawn.

    The page needs it before a question runs; the switch itself is drawn after the
    conversation, so that it can say what the conversation has already sent.
    """
    return bool(st.session_state.get(SCHEMA_ONLY_KEY, False))


def render_controls(session: AnalystSession) -> None:
    """Draw the rest of the sidebar, after the conversation, so it shows its latest state."""
    with st.sidebar:
        st.divider()
        _privacy_controls(session)
        _conversation_controls(session)


# -- a staged file --------------------------------------------------------------------------------


def _staged_file(session: AnalystSession, staged: UploadedFile) -> None:
    try:
        kind = file_kind(staged.name)
    except IngestionError as error:
        st.error(error.user_message)
        return
    options = _read_options(session, staged, kind)
    if options is None:
        return
    key = (staged.file_id, options)

    action = "Replace dataset" if session.has_dataset else "Load file"
    clicked = st.button(action, type="primary", key="load_file", width="stretch")
    needs_confirmation = session.has_dataset and bool(session.history)
    if clicked and needs_confirmation:
        st.session_state[PENDING_KEY] = key
    elif clicked:
        _load(session, staged, options, key)

    if st.session_state.get(PENDING_KEY) == key:
        st.warning("Replacing the dataset clears this conversation. Do you want to replace it?")
        yes, no = st.columns(2)
        if yes.button("Yes, replace", key="replace_yes", width="stretch"):
            _load(session, staged, options, key)
        if no.button("Cancel", key="replace_no", width="stretch"):
            st.session_state.pop(PENDING_KEY, None)
            empty_uploader()
            st.rerun()
    failure = session.load_failure
    if failure is not None and failure.key == key:
        st.error(failure.message)


def _load(session: AnalystSession, staged: UploadedFile, options: ReadOptions, key: object) -> None:
    with st.spinner("Reading your file. A large Excel file can take up to 15 seconds."):
        loaded = session.load(staged.getvalue(), staged.name, options, key=key)
    st.session_state.pop(PENDING_KEY, None)
    if loaded:
        _release(staged)
        empty_uploader()
        st.rerun()


def _release(staged: UploadedFile) -> None:
    """Delete the uploaded bytes from Streamlit's memory: the table is all that is kept.

    Streamlit keeps an uploaded file until the user removes it or the session ends, so
    emptying the uploader alone would leave a copy behind for as long as the tab is open.
    If this fails the copy stays until the session ends, and that is logged.
    """
    try:
        ctx = get_script_run_ctx()
        if ctx is not None:
            ctx.uploaded_file_mgr.remove_file(ctx.session_id, staged.file_id)
    except Exception as error:
        log.error("upload_not_released", extra={"error_type": type(error).__name__})


# -- how to read it -------------------------------------------------------------------------------


def _read_options(session: AnalystSession, staged: UploadedFile, kind: str) -> ReadOptions | None:
    """The user's choices for this file, or ``None`` when the file cannot be offered."""
    sheet: str | None = None
    if kind == "excel":
        sheets = _sheets(session, staged)
        if isinstance(sheets, str):
            st.error(sheets)
            return None
        sheet = _choose_sheet(staged, sheets)
    delimiter: str | None = None
    with st.expander("How to read this file"):
        if kind == "csv":
            delimiter = st.selectbox(
                "Separator",
                list(DELIMITER_CHOICES),
                format_func=DELIMITER_CHOICES.__getitem__,
                key=f"delimiter-{staged.file_id}",
            )
        header = st.radio(
            "Header row",
            list(HEADER_CHOICES),
            format_func=HEADER_CHOICES.__getitem__,
            key=f"header-{staged.file_id}",
            help="The row that holds the column names. Rows above it are left out.",
        )
        header_row = 1
        if header == "row":
            header_row = int(
                st.number_input(
                    "Header row number",
                    min_value=1,
                    max_value=MAX_HEADER_ROW,
                    value=1,
                    step=1,
                    key=f"header-row-{staged.file_id}",
                )
            )
    return ReadOptions(sheet=sheet, delimiter=delimiter, header=header, header_row=header_row)


def _sheets(session: AnalystSession, staged: UploadedFile) -> list[SheetInfo] | str:
    """The workbook's sheets, or the message to show when it cannot be opened. Cached."""
    cached = st.session_state.get(SHEETS_KEY)
    if cached is not None and cached[0] == staged.file_id:
        return cached[1]
    result: list[SheetInfo] | str
    try:
        result = list_sheets(staged.getvalue(), settings=session.settings)
        if not result:
            result = IngestionError("empty").user_message
    except IngestionError as error:
        result = error.user_message
    except Exception as error:
        log.error("sheets_failed", extra={"error_type": type(error).__name__})
        result = IngestionError.fallback_message
    st.session_state[SHEETS_KEY] = (staged.file_id, result)
    return result


def _choose_sheet(staged: UploadedFile, sheets: list[SheetInfo]) -> str:
    """Which sheet to read: visible sheets first, and the first one unless the user picks."""
    ordered = [s for s in sheets if s.visible] + [s for s in sheets if not s.visible]
    by_name = {s.name: s for s in ordered}
    if len(ordered) == 1:
        return ordered[0].name

    def label(name: str) -> str:
        info = by_name[name]
        hidden = ", hidden" if not info.visible else ""
        size = f"{count_text(info.rows, 'row')}, {count_text(info.columns, 'column')}"
        return f"{clean_text(name)} ({size}{hidden})"

    return st.selectbox("Sheet", list(by_name), format_func=label, key=f"sheet-{staged.file_id}")


# -- privacy and conversation ---------------------------------------------------------------------


def _privacy_controls(session: AnalystSession) -> None:
    st.toggle(
        "Schema-only mode",
        key=SCHEMA_ONLY_KEY,
        help=(
            "Sends OpenAI no sample rows and no common values from your file. The rows of each "
            "answer are still sent so that they can be explained."
        ),
    )
    if session.schema_only and session.values_may_have_been_sent:
        st.caption(
            "Questions you asked before turning this on may already have included values "
            "from your data. Start a new conversation to be sure none are sent."
        )
    with st.expander("What is sent to OpenAI"):
        st.markdown(privacy_notice(session.settings.session_idle_minutes))


def _conversation_controls(session: AnalystSession) -> None:
    if st.button(
        "New conversation",
        disabled=not session.history,
        width="stretch",
        help="Clears the questions and answers and keeps your data.",
    ):
        session.reset_conversation()
        st.rerun()
