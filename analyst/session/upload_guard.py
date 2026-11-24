"""A ceiling on the bytes Streamlit holds for uploads, for every session together (T1.21).

Streamlit's upload route takes a file from any browser that has an open session. It does not
look at the app's password: the password is checked by the page script, and the route never
runs it. A visitor who has not signed in could therefore send file after file, up to the size
limit each, and Streamlit would keep them in memory until the session ends and its time to
live runs out (120 seconds by default). Enough of them would use up the server's memory, and
the restart that follows ends everyone's session.

Streamlit has no setting for this, so the route's file store is wrapped: ``add_file`` refuses a
file that would take the bytes held above ``max_bytes`` and answers 413. A real upload holds a
file only from choosing it until *Load file*, so the ceiling is generous for the testers and
bounds what a visitor can hold. The cost is that a visitor can fill the ceiling and make real
uploads fail until their files expire; the server stays up. See docs/security.md, T13.
"""

from __future__ import annotations

import threading
from typing import Any

from starlette.exceptions import HTTPException

from analyst.core.logging import get_logger

log = get_logger(__name__)

# How many maximum-size files may be held at once, over all sessions (ADR 0014 sizes the
# server for about 8 large sessions).
HELD_FILES = 8
REFUSED_STATUS = 413
REFUSED_MESSAGE = "The server is holding too many uploads right now. Try again in a few minutes."
_MARK = "_analyst_upload_guard"


def held_bytes(manager: Any) -> int:
    """The bytes of file data the manager holds for every session."""
    storage = manager.file_storage  # {session_id: {file_id: UploadedFileRec}}
    return sum(len(file.data) for files in list(storage.values()) for file in list(files.values()))


def install(manager: Any, max_bytes: int) -> bool:
    """Wrap ``manager.add_file`` so the bytes held never go above ``max_bytes``.

    Returns ``True`` if it wrapped the manager, ``False`` if it was already wrapped. The wrap
    is on this one object, so it holds for the routes that were built with it.
    """
    if getattr(manager, _MARK, False):
        return False
    add_file = manager.add_file
    lock = threading.Lock()  # every add goes through here, so check and add are one step

    def guarded_add_file(session_id: str, file: Any) -> None:
        with lock:
            if held_bytes(manager) + len(file.data) > max_bytes:
                log.warning("upload_refused", extra={"bytes": len(file.data), "limit": max_bytes})
                raise HTTPException(status_code=REFUSED_STATUS, detail=REFUSED_MESSAGE)
            add_file(session_id, file)

    manager.add_file = guarded_add_file
    setattr(manager, _MARK, True)
    return True


def install_on_running_server(max_upload_bytes: int) -> bool:
    """Guard the running Streamlit server's upload store; ``False`` if there is no server.

    Called on every run of the page, so it does its work once. Under ``AppTest`` and in unit
    tests there is no server, and nothing is wrapped.
    """
    from streamlit import runtime

    if not runtime.exists():
        return False
    return install(runtime.get_instance().uploaded_file_mgr, HELD_FILES * max_upload_bytes)
