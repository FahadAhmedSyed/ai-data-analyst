"""The ceiling on uploads, put on the running Streamlit server (T1.21, security T13)."""

from __future__ import annotations

import pytest
from starlette.exceptions import HTTPException
from streamlit import runtime
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.uploaded_file_manager import UploadedFileRec

from analyst.session.upload_guard import HELD_FILES
from analyst.ui import upload_limit

MB = 1024 * 1024


def rec(file_id: str, size: int) -> UploadedFileRec:
    return UploadedFileRec(file_id=file_id, name="x.csv", type="text/csv", data=b"a" * size)


def test_with_no_running_server_nothing_is_wrapped() -> None:
    assert not runtime.exists()
    assert upload_limit.install_on_running_server(50 * MB) is False


def test_on_a_running_server_the_stores_ceiling_is_eight_maximum_uploads(monkeypatch) -> None:
    store = MemoryUploadedFileManager(upload_endpoint="/_stcore/upload_file")

    class FakeRuntime:
        uploaded_file_mgr = store

    monkeypatch.setattr(runtime, "exists", lambda: True)
    monkeypatch.setattr(runtime, "get_instance", lambda: FakeRuntime())
    assert upload_limit.install_on_running_server(MB) is True
    for i in range(HELD_FILES):
        store.add_file("s", rec(f"f{i}", MB))
    with pytest.raises(HTTPException):
        store.add_file("s", rec("one-more", 1))
    assert upload_limit.install_on_running_server(MB) is False  # the next page run: no second wrap
