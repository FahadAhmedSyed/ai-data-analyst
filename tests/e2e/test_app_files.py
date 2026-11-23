"""Choosing a file, how to read it, and replacing it (FR-1.1, FR-1.2, FR-1.4, FR-7.4)."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import openpyxl
import pytest

from analyst.core.errors import IngestionError
from analyst.session.state import AnalystSession
from analyst.ui import sidebar
from analyst.ui.sidebar import PENDING_KEY
from tests.e2e.driver import OTHER, SALES, Page
from tests.e2e.test_app_flows import ASK, HAPPY


def workbook(*sheets: tuple[str, list[list[object]], str]) -> bytes:
    """An .xlsx file; each sheet is ``(name, rows, state)`` with state ``visible`` or ``hidden``."""
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows, state in sheets:
        sheet = book.create_sheet(name)
        sheet.sheet_state = state
        for row in rows:
            sheet.append(row)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def button_keys(page: Page) -> list[str | None]:
    return [b.key for b in page.at.sidebar.button]


SHEETS = workbook(
    ("Lookup", [["Code"], ["x"]], "hidden"),
    ("Sales", [["Region", "Revenue"], ["North", 10], ["South", 20], ["East", 30]], "visible"),
    ("Costs", [["Item", "Cost"], ["Tea", 3]], "visible"),
)


# -- Excel ----------------------------------------------------------------------------------------


def test_a_workbook_with_several_sheets_offers_them_with_the_visible_ones_first() -> None:
    page = Page().upload("book.xlsx", SHEETS)
    sheet = page.at.selectbox(key=f"sheet-{page.at.file_uploader[0].value.file_id}")
    assert sheet.label == "Sheet"
    assert sheet.options == [
        "Sales (4 rows, 2 columns)",
        "Costs (2 rows, 2 columns)",
        "Lookup (2 rows, 1 column, hidden)",
    ]
    assert sheet.value == "Sales"


def test_the_chosen_sheet_is_the_one_loaded_and_is_named_beside_the_file() -> None:
    page = Page().upload("book.xlsx", SHEETS)
    page.at.selectbox[0].select("Costs").run()
    page.load()
    assert page.session.dataset.profile.sheet_name == "Costs"
    assert [c.sql_name for c in page.session.dataset.profile.columns] == ["item", "cost"]
    assert any("book\\.xlsx" in m.value and "Costs" in m.value for m in page.at.sidebar.markdown)
    assert "Read from the sheet Costs" in page.page_text()


def test_a_workbook_with_one_sheet_does_not_ask_which() -> None:
    one = workbook(("Only", [["a", "b"], [1, 2]], "visible"))
    page = Page().upload("one.xlsx", one)
    assert len(page.at.selectbox) == 0
    assert page.load().session.dataset.profile.sheet_name == "Only"


def test_a_workbook_that_cannot_be_opened_says_so_and_offers_no_button() -> None:
    page = Page().upload("broken.xlsx", b"PK\x03\x04" + b"\x00" * 200)
    assert [e.value for e in page.at.sidebar.error]
    assert "load_file" not in button_keys(page)
    assert not page.session.has_dataset


def test_a_workbook_with_no_data_is_offered_and_then_refused_with_the_reason() -> None:
    page = Page().upload("empty.xlsx", workbook(("Empty", [], "visible")))
    assert len(page.at.sidebar.error) == 0
    page.load()
    assert not page.session.has_dataset
    assert len(page.at.sidebar.error) == 1
    assert page.at.file_uploader[0].value is not None


# -- how to read a file ---------------------------------------------------------------------------


def test_the_separator_can_be_chosen_when_detection_is_wrong() -> None:
    data = b"a;b;c\n1;2;3\n4;5;6\n"
    detected = Page().open_file("x.csv", data)
    assert detected.session.dataset.profile.column_count == 3

    chosen = Page().upload("x.csv", data)
    chosen.at.selectbox[0].select(",").run()
    chosen.load()
    assert chosen.session.dataset.profile.column_count == 1
    assert "separated by commas" in chosen.page_text()


def test_the_header_row_can_be_chosen_when_the_file_has_a_title_above_it() -> None:
    data = b"Quarterly report,,\nGenerated today,,\nName,Score,Grade\nAnn,5,A\nBo,7,B\n"
    page = Page().upload("x.csv", data)
    page.at.radio[0].set_value("row").run()
    page.at.number_input[0].set_value(3).run()
    page.load()
    dataset = page.session.dataset
    assert [c.display_name for c in dataset.profile.columns] == ["Name", "Score", "Grade"]
    assert dataset.profile.row_count == 2
    assert "the header is on row 3" in page.page_text()
    assert "2 rows above it were left out" in page.page_text()


def test_the_row_number_is_only_asked_for_when_the_row_is_to_be_chosen() -> None:
    page = Page().upload("x.csv", SALES)
    assert len(page.at.number_input) == 0
    page.at.radio[0].set_value("row").run()
    assert page.at.number_input[0].label == "Header row number"
    assert page.at.number_input[0].min == 1
    assert page.at.number_input[0].max == 1_000
    assert page.at.number_input[0].value == 1
    assert page.at.number_input[0].step == 1


def test_a_file_can_be_read_as_having_no_header() -> None:
    page = Page().upload("x.csv", OTHER)
    page.at.radio[0].set_value("none").run()
    page.load()
    names = [c.display_name for c in page.session.dataset.profile.columns]
    assert names == ["Column 1", "Column 2"]
    assert page.session.dataset.profile.row_count == 3
    assert "no header row was used" in page.page_text()


def test_the_separator_is_not_offered_for_a_workbook() -> None:
    page = Page().upload("book.xlsx", SHEETS)
    assert [s.label for s in page.at.selectbox] == ["Sheet"]


def test_changing_how_to_read_a_file_does_not_load_it() -> None:
    page = Page().upload("x.csv", SALES)
    page.at.radio[0].set_value("none").run()
    assert not page.session.has_dataset


# -- a file that cannot be loaded ------------------------------------------------------------------


def test_a_file_that_fails_to_load_stays_staged_with_the_reason() -> None:
    page = Page().upload("empty.csv", b"").load()
    assert not page.session.has_dataset
    assert len(page.at.sidebar.error) == 1
    assert page.at.file_uploader[0].value is not None
    assert page.at.sidebar.button(key="load_file").label == "Load file"


def test_another_setting_can_be_tried_on_a_staged_file_that_failed() -> None:
    page = Page().upload("x.csv", b"Title\n").load()
    assert not page.session.has_dataset
    page.at.radio[0].set_value("none").run()
    assert len(page.at.sidebar.error) == 0  # the reason belongs to the choice that failed
    page.load()
    assert page.session.has_dataset
    assert page.session.load_failure is None


def test_a_file_that_cannot_be_read_leaves_the_loaded_dataset_alone() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    page.upload("empty.csv", b"")
    page.load()
    page.at.sidebar.button(key="replace_yes").click().run()
    assert len(page.at.sidebar.error) == 1
    assert page.session.has_dataset
    assert page.session.dataset.profile.source_name == "sales.csv"
    assert len(page.session.history) == 1


# -- replacing the dataset -------------------------------------------------------------------------


def test_with_no_questions_asked_a_file_replaces_the_dataset_at_once() -> None:
    page = Page().open_file()
    page.upload("other.csv", OTHER)
    assert page.at.sidebar.button(key="load_file").label == "Replace dataset"
    page.load()
    assert page.session.dataset.profile.source_name == "other.csv"
    assert len(page.at.sidebar.warning) == 0
    assert len(page.clients) == 2  # a new model client for the new dataset


def test_with_questions_asked_replacing_the_dataset_asks_first() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    page.upload("other.csv", OTHER).load()
    (warning,) = page.at.sidebar.warning
    assert "clears this conversation" in warning.value
    assert page.session.dataset.profile.source_name == "sales.csv"
    assert len(page.session.history) == 1
    assert [b.label for b in page.at.sidebar.button if b.label.startswith(("Yes", "Cancel"))] == [
        "Yes, replace",
        "Cancel",
    ]


def test_confirming_replaces_the_dataset_and_clears_the_conversation() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    page.upload("other.csv", OTHER).load()
    page.at.sidebar.button(key="replace_yes").click().run()
    assert page.session.dataset.profile.source_name == "other.csv"
    assert page.session.history == []
    assert len(page.at.chat_message) == 0
    assert len(page.at.sidebar.warning) == 0
    assert page.at.file_uploader[0].value is None
    assert not page.at.chat_input[0].disabled
    assert PENDING_KEY not in page.at.session_state  # the question is not left open


def test_cancelling_keeps_the_dataset_and_the_conversation_and_empties_the_uploader() -> None:
    page = Page(HAPPY).open_file().ask(ASK)
    page.upload("other.csv", OTHER).load()
    page.at.sidebar.button(key="replace_no").click().run()
    assert page.session.dataset.profile.source_name == "sales.csv"
    assert len(page.session.history) == 1
    assert len(page.at.chat_message) == 2
    assert page.at.file_uploader[0].value is None
    assert len(page.at.sidebar.warning) == 0
    assert PENDING_KEY not in page.at.session_state


def test_the_confirmation_is_for_the_file_it_was_asked_about() -> None:
    """Choosing another file or setting while the question is open does not carry the 'yes'."""
    page = Page(HAPPY).open_file().ask(ASK)
    page.upload("other.csv", OTHER).load()
    assert len(page.at.sidebar.warning) == 1
    page.at.radio[0].set_value("none").run()
    assert len(page.at.sidebar.warning) == 0
    assert page.session.dataset.profile.source_name == "sales.csv"


def test_a_replacement_forgets_what_the_conversation_assumed() -> None:
    page = Page(HAPPY, []).open_file().ask(ASK)
    assert page.session.memory.recent()
    page.upload("other.csv", OTHER).load()
    page.at.sidebar.button(key="replace_yes").click().run()
    assert not page.session.memory.recent()
    assert not page.session.memory.carried_assumptions()


# -- the uploaded bytes ----------------------------------------------------------------------------


class _Manager:
    def __init__(self, fail: bool = False) -> None:
        self.removed: list[tuple[str, str]] = []
        self.fail = fail

    def remove_file(self, session_id: str, file_id: str) -> None:
        if self.fail:
            raise OSError("/private/path/sales.csv could not be removed")
        self.removed.append((session_id, file_id))


def _context(manager: _Manager) -> SimpleNamespace:
    return SimpleNamespace(uploaded_file_mgr=manager, session_id="browser-1")


def test_the_uploaded_bytes_are_deleted_from_streamlit_once_the_file_is_loaded(
    monkeypatch,
) -> None:
    manager = _Manager()
    monkeypatch.setattr("analyst.ui.sidebar.get_script_run_ctx", lambda: _context(manager))
    page = Page().upload("sales.csv", SALES)
    assert manager.removed == []
    page.load()
    assert len(manager.removed) == 1
    session_id, file_id = manager.removed[0]
    assert session_id == "browser-1"
    assert file_id


def test_the_uploaded_bytes_are_kept_while_the_file_is_only_staged_or_failed(monkeypatch) -> None:
    manager = _Manager()
    monkeypatch.setattr("analyst.ui.sidebar.get_script_run_ctx", lambda: _context(manager))
    Page().upload("empty.csv", b"").load()
    assert manager.removed == []  # the user may try another setting on the same bytes


def test_a_release_that_fails_is_logged_without_the_error_text_and_does_not_stop_the_load(
    monkeypatch, capsys
) -> None:
    manager = _Manager(fail=True)
    monkeypatch.setattr("analyst.ui.sidebar.get_script_run_ctx", lambda: _context(manager))
    page = Page().open_file()
    assert page.session.has_dataset
    written = capsys.readouterr().out
    (record,) = [
        r
        for r in (json.loads(x) for x in written.splitlines() if x.startswith("{"))
        if r["event"] == "upload_not_released"
    ]
    assert record["error_type"] == "OSError"
    assert "/private/path" not in written


def test_there_is_nothing_to_release_outside_a_streamlit_run(monkeypatch) -> None:
    monkeypatch.setattr("analyst.ui.sidebar.get_script_run_ctx", lambda: None)
    assert Page().open_file().session.has_dataset


@pytest.mark.parametrize("name", ["sales.csv", "sales.txt"])
def test_csv_and_text_files_are_both_read_as_text(name: str) -> None:
    page = Page().open_file(name, SALES)
    assert page.session.dataset.read.kind == "csv"


# -- what the sidebar offers and when --------------------------------------------------------------


def notices(page: Page) -> list[str]:
    return [c.value for c in page.at.sidebar.caption if "sends OpenAI the column names" in c.value]


def test_the_privacy_notice_is_beside_the_button_and_not_in_the_way_once_a_file_is_loaded() -> None:
    page = Page()
    assert len(notices(page)) == 1  # nothing loaded yet
    page.upload("sales.csv", SALES)
    assert len(notices(page)) == 1  # a file is staged: it is told what loading sends
    page.load()
    assert notices(page) == []  # the expander "What is sent to OpenAI" keeps it available
    assert any(e.label == "What is sent to OpenAI" for e in page.at.sidebar.expander)
    page.upload("other.csv", OTHER)
    assert len(notices(page)) == 1  # a replacement is staged


def test_the_switch_starts_off_on_the_very_first_run() -> None:
    assert Page().session.schema_only is False


def test_the_sidebar_is_divided_between_the_file_and_the_settings() -> None:
    assert len(Page().at.sidebar.divider) == 1


def test_the_file_type_is_checked_again_when_a_file_is_staged(monkeypatch) -> None:
    def unsupported(name: str) -> str:
        raise IngestionError("unsupported_type")

    monkeypatch.setattr(sidebar, "file_kind", unsupported)
    page = Page().upload("sales.csv", SALES)
    assert [e.value for e in page.at.sidebar.error] == [
        IngestionError("unsupported_type").user_message
    ]
    assert "load_file" not in button_keys(page)


def test_a_workbooks_sheets_are_listed_once_however_often_the_page_is_redrawn(monkeypatch) -> None:
    listed: list[int] = []
    real = sidebar.list_sheets

    def counting(data, **kwargs):
        listed.append(len(data))
        return real(data, **kwargs)

    monkeypatch.setattr(sidebar, "list_sheets", counting)
    page = Page().upload("book.xlsx", SHEETS)
    page.at.selectbox[0].select("Costs").run()
    page.at.run()
    assert len(listed) == 1
    page.upload("other.xlsx", workbook(("Only", [["a"], [1]], "visible")))
    assert len(listed) == 2  # another file is another workbook


def test_a_workbook_that_fails_in_a_way_nobody_planned_for_is_a_plain_message(
    monkeypatch, capsys
) -> None:
    def broken(data, **kwargs):
        raise RuntimeError("/private/path/book.xlsx SECRET-CELL")

    monkeypatch.setattr(sidebar, "list_sheets", broken)
    page = Page().upload("book.xlsx", SHEETS)
    assert [e.value for e in page.at.sidebar.error] == [IngestionError.fallback_message]
    written = capsys.readouterr().out
    records = [json.loads(x) for x in written.splitlines() if x.startswith("{")]
    (failed,) = [r for r in records if r["event"] == "sheets_failed"]
    assert failed["error_type"] == "RuntimeError"
    assert "SECRET-CELL" not in written and "/private/path" not in written


def test_the_only_sheet_of_a_workbook_is_named_in_the_options_that_are_loaded(monkeypatch) -> None:
    seen: list[object] = []
    real = AnalystSession.load

    def watching(self, data, filename, options=None, *, key=None):
        seen.append(options)
        return real(self, data, filename, options, key=key)

    monkeypatch.setattr(AnalystSession, "load", watching)
    Page().upload("one.xlsx", workbook(("Only", [["a", "b"], [1, 2]], "visible"))).load()
    assert [o.sheet for o in seen] == ["Only"]
