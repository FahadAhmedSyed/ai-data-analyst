"""The ceiling on uploads held in memory (T1.21, security T13), on Streamlit's real store."""

from __future__ import annotations

import pytest
from starlette.exceptions import HTTPException
from streamlit import runtime
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.uploaded_file_manager import UploadedFileRec

from analyst.session import upload_guard
from analyst.session.upload_guard import HELD_FILES, REFUSED_STATUS, held_bytes, install

MB = 1024 * 1024


def rec(file_id: str, size: int) -> UploadedFileRec:
    return UploadedFileRec(file_id=file_id, name="x.csv", type="text/csv", data=b"a" * size)


@pytest.fixture
def manager() -> MemoryUploadedFileManager:
    return MemoryUploadedFileManager(upload_endpoint="/_stcore/upload_file")


def test_files_under_the_ceiling_are_stored_as_before(manager) -> None:
    install(manager, 10 * MB)
    manager.add_file("s1", rec("f1", 4 * MB))
    manager.add_file("s2", rec("f2", 4 * MB))
    assert held_bytes(manager) == 8 * MB
    assert [f.file_id for f in manager.get_files("s1", ["f1"])] == ["f1"]


def test_a_file_that_would_pass_the_ceiling_is_refused_with_413_and_not_stored(manager) -> None:
    install(manager, 10 * MB)
    manager.add_file("s1", rec("f1", 6 * MB))
    with pytest.raises(HTTPException) as refused:
        manager.add_file("s2", rec("f2", 6 * MB))
    assert refused.value.status_code == REFUSED_STATUS == 413
    assert manager.get_files("s2", ["f2"]) == []
    assert held_bytes(manager) == 6 * MB


def test_a_file_that_just_fits_is_stored(manager) -> None:
    install(manager, 10 * MB)
    manager.add_file("s1", rec("f1", 6 * MB))
    manager.add_file("s1", rec("f2", 4 * MB))
    assert held_bytes(manager) == 10 * MB


def test_one_visitor_sending_many_files_is_stopped_at_the_ceiling(manager) -> None:
    install(manager, 10 * MB)
    refused = 0
    for i in range(50):
        try:
            manager.add_file("visitor", rec(f"f{i}", MB))
        except HTTPException:
            refused += 1
    assert held_bytes(manager) == 10 * MB
    assert refused == 40


def test_room_comes_back_when_a_session_ends_or_a_file_is_removed(manager) -> None:
    install(manager, 10 * MB)
    manager.add_file("s1", rec("f1", 6 * MB))
    manager.add_file("s2", rec("f2", 4 * MB))
    with pytest.raises(HTTPException):
        manager.add_file("s3", rec("f3", MB))
    manager.remove_session_files("s1")
    manager.add_file("s3", rec("f3", 5 * MB))
    manager.remove_file("s2", "f2")
    manager.add_file("s4", rec("f4", 4 * MB))
    assert held_bytes(manager) == 9 * MB


def test_the_refusal_is_logged_without_the_file_name_or_content(manager, capsys) -> None:
    from analyst.core.logging import configure_logging

    configure_logging("INFO")
    install(manager, MB)
    with pytest.raises(HTTPException):
        manager.add_file(
            "s1", UploadedFileRec("f1", "secret-name.csv", "text/csv", b"x" * (2 * MB))
        )
    out = capsys.readouterr().out
    assert '"event": "upload_refused"' in out
    assert "secret-name" not in out


def test_installing_twice_wraps_once(manager) -> None:
    assert install(manager, 10 * MB) is True
    wrapped = manager.add_file
    assert install(manager, 1) is False  # a second call must not change the ceiling
    assert manager.add_file is wrapped
    manager.add_file("s1", rec("f1", 5 * MB))


def test_the_ceiling_is_a_fixed_number_of_maximum_size_uploads() -> None:
    assert HELD_FILES == 8


def test_with_no_running_server_nothing_is_wrapped() -> None:
    assert not runtime.exists()
    assert upload_guard.install_on_running_server(50 * MB) is False


def test_on_a_running_server_the_stores_ceiling_is_eight_maximum_uploads(monkeypatch) -> None:
    store = MemoryUploadedFileManager(upload_endpoint="/_stcore/upload_file")

    class FakeRuntime:
        uploaded_file_mgr = store

    monkeypatch.setattr(runtime, "exists", lambda: True)
    monkeypatch.setattr(runtime, "get_instance", lambda: FakeRuntime())
    assert upload_guard.install_on_running_server(MB) is True
    for i in range(HELD_FILES):
        store.add_file("s", rec(f"f{i}", MB))
    with pytest.raises(HTTPException):
        store.add_file("s", rec("one-more", 1))
    assert upload_guard.install_on_running_server(MB) is False  # the next page run: no second wrap
