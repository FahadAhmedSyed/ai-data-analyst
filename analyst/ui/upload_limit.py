"""Puts the ceiling on uploads (``analyst.session.upload_guard``) on the running server (T1.21)."""

from __future__ import annotations

from streamlit import runtime

from analyst.session.upload_guard import HELD_FILES, install


def install_on_running_server(max_upload_bytes: int) -> bool:
    """Guard the running Streamlit server's upload store; ``False`` if there is no server.

    Called on every run of the page, so it does its work once. Under ``AppTest`` and in unit
    tests there is no server, and nothing is wrapped.
    """
    if not runtime.exists():
        return False
    return install(runtime.get_instance().uploaded_file_mgr, HELD_FILES * max_upload_bytes)
