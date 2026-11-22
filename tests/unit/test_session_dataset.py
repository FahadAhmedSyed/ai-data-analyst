"""Loading a file into a session (T1.18): read, type, store, profile.

Expected counts are written by hand from the CSV text in each test.
"""

from __future__ import annotations

import logging

import pytest

from analyst.config.settings import Settings
from analyst.core.errors import IngestionError
from analyst.database.session_db import SessionDatabase
from analyst.session import dataset as dataset_module
from analyst.session.dataset import ReadInfo, ReadOptions, load_dataset
from tests.unit.test_excel_loader import rainfall_book, workbook_bytes

SETTINGS = Settings(_env_file=None)
SALES = b"Region,Revenue\nNorth,10\nSouth,20\nNorth,\n"


@pytest.fixture
def db():
    with SessionDatabase(SETTINGS) as database:
        yield database


def load(db, data=SALES, name="sales.csv", options=None):
    return load_dataset(db, data, name, options, settings=SETTINGS)


# -- what loading gives ---------------------------------------------------------------------------


def test_a_csv_file_is_loaded_typed_and_profiled(db) -> None:
    out = load(db)
    assert db.is_loaded
    assert db.table_name == "sales"
    assert (out.profile.row_count, out.profile.column_count) == (3, 2)
    assert [c.dtype for c in out.profile.columns] == ["text", "integer"]
    assert out.profile.columns[1].null_count == 1
    assert out.profile.source_name == "sales.csv"


def test_the_quality_notes_come_with_the_file(db) -> None:
    out = load(db)
    assert [i.kind for i in out.issues] == ["missing_values"]
    assert "Revenue is missing in 1 rows" in out.issues[0].message


def test_the_preview_is_made_with_the_dataset(db) -> None:
    out = load(db)
    assert out.preview["Region"].tolist() == ["North", "South", "North"]


def test_how_the_file_was_read_is_kept(db) -> None:
    out = load(db, b"A;B\n1;2\n", "semi.csv")
    assert out.read == ReadInfo(
        kind="csv", encoding="utf-8", delimiter=";", header_row=1, skipped_rows=0, sheet=None
    )


def test_an_excel_sheet_is_loaded_by_name(db) -> None:
    book = workbook_bytes(rainfall_book)
    out = load(db, book, "rain.xlsx", ReadOptions(sheet="Sites"))
    assert out.profile.sheet_name == "Sites"
    assert out.profile.row_count == 1
    assert out.read.kind == "excel"
    assert out.read.sheet == "Sites"


def test_an_excel_file_without_a_chosen_sheet_reads_the_first_visible_one(db) -> None:
    out = load(db, workbook_bytes(rainfall_book), "rain.xlsx")
    assert out.profile.sheet_name == "Daily"
    assert out.read.header_row == 3  # the title and a blank row sit above it
    assert out.read.skipped_rows == 2


def test_loading_again_replaces_the_table(db) -> None:
    load(db)
    out = load(db, b"X\n1\n2\n", "other.csv")
    assert db.table_name == "other"
    assert out.profile.row_count == 2


# -- the options ----------------------------------------------------------------------------------


def test_the_separator_can_be_chosen(db) -> None:
    out = load(db, b"A|B\n1|2\n", "p.csv", ReadOptions(delimiter="|"))
    assert out.profile.column_count == 2
    assert out.read.delimiter == "|"


def test_a_header_row_is_counted_from_one(db) -> None:
    csv = b"Report title,,\nnotes,,\nName,Score,Rank\nAnn,5,1\nBo,7,2\n"
    out = load(db, csv, "h.csv", ReadOptions(header="row", header_row=3))
    assert [c.display_name for c in out.profile.columns] == ["Name", "Score", "Rank"]
    assert out.profile.row_count == 2
    assert out.read.header_row == 3
    assert out.read.skipped_rows == 2


def test_a_file_can_be_said_to_have_no_header(db) -> None:
    out = load(db, b"Ann,5\nBo,7\n", "n.csv", ReadOptions(header="none"))
    assert [c.display_name for c in out.profile.columns] == ["Column 1", "Column 2"]
    assert out.profile.row_count == 2
    assert out.read.header_row is None


def test_detected_headers_are_the_default(db) -> None:
    out = load(db, b"Title,,\n,,\nName,Score,\nAnn,5,\n", "d.csv")
    assert [c.display_name for c in out.profile.columns] == ["Name", "Score"]


@pytest.mark.parametrize(
    ("options", "arguments"),
    [
        (ReadOptions(), (None, True)),
        (ReadOptions(header="auto", header_row=9), (None, True)),
        (ReadOptions(header="none", header_row=9), (None, False)),
        (ReadOptions(header="row", header_row=1), (0, True)),
        (ReadOptions(header="row", header_row=5), (4, True)),
        (ReadOptions(header="row", header_row=0), (0, True)),  # never a negative row
        (ReadOptions(header="row"), (0, True)),  # the first row unless the user says another
    ],
)
def test_the_loaders_count_rows_from_zero_and_the_user_from_one(options, arguments) -> None:
    assert options.header_arguments() == arguments


# -- descriptions ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("read", "text"),
    [
        (
            ReadInfo("csv", "utf-8", ",", 1, 0),
            "Read as a text file with columns separated by commas (utf-8); the header is on row 1.",
        ),
        (
            ReadInfo("csv", "windows-1252", ";", 3, 2),
            "Read as a text file with columns separated by semicolons (windows-1252); "
            "the header is on row 3, and 2 rows above it were left out.",
        ),
        (
            ReadInfo("csv", None, "\t", 2, 1),
            "Read as a text file with columns separated by tabs; "
            "the header is on row 2, and 1 row above it was left out.",
        ),
        (
            ReadInfo("csv", None, "|", None, 0),
            "Read as a text file with columns separated by pipes; "
            "no header row was used, so columns are named Column 1, Column 2 and so on.",
        ),
        (
            ReadInfo("csv", None, "#", 1, 0),
            "Read as a text file with columns separated by a custom separator; "
            "the header is on row 1.",
        ),
        (
            ReadInfo("excel", None, None, 1, 0, sheet="Daily"),
            "Read from the sheet Daily; the header is on row 1.",
        ),
        (ReadInfo("excel", None, None, 1, 0), "Read from the sheet; the header is on row 1."),
    ],
)
def test_how_a_file_was_read_is_described_in_a_sentence(read: ReadInfo, text: str) -> None:
    assert read.describe() == text


# -- failure --------------------------------------------------------------------------------------


def test_an_unsupported_file_is_refused_before_anything_is_read(db) -> None:
    with pytest.raises(IngestionError) as caught:
        load(db, b"data", "notes.pdf")
    assert caught.value.category == "unsupported_type"
    assert not db.is_loaded


def test_a_file_that_cannot_be_read_leaves_the_loaded_dataset_alone(db) -> None:
    load(db)
    with pytest.raises(IngestionError):
        load(db, b"", "empty.csv")
    assert db.is_loaded
    assert db.table_name == "sales"


def test_a_header_row_outside_the_file_is_a_message_not_a_crash(db) -> None:
    with pytest.raises(IngestionError) as caught:
        load(db, b"A,B\n1,2\n", "h.csv", ReadOptions(header="row", header_row=50))
    assert "outside the file" in caught.value.user_message


def test_a_failure_after_the_table_was_loaded_leaves_no_table_behind(db, monkeypatch) -> None:
    def broken(_db):
        raise RuntimeError("profiler bug")

    monkeypatch.setattr(dataset_module, "profile_database", broken)
    with pytest.raises(RuntimeError, match="profiler bug"):
        load(db)
    assert not db.is_loaded


def test_an_interruption_also_leaves_no_table_behind(db, monkeypatch) -> None:
    def interrupted(_db):
        raise KeyboardInterrupt

    monkeypatch.setattr(dataset_module, "profile_database", interrupted)
    with pytest.raises(KeyboardInterrupt):
        load(db)
    assert not db.is_loaded


def test_a_file_over_the_row_limit_is_refused_with_the_limit(db) -> None:
    small = Settings(_env_file=None, max_rows=2)
    with pytest.raises(IngestionError) as caught:
        load_dataset(db, SALES, "sales.csv", settings=small)
    assert caught.value.category == "too_many_rows"
    assert "2" in caught.value.user_message


# -- logs -----------------------------------------------------------------------------------------


def test_the_log_line_has_counts_and_no_names_or_values(db, caplog) -> None:
    caplog.set_level(logging.INFO)
    load(db, b"Secret Column\nzebra-value\n", "confidential-payroll.csv")
    (record,) = [r for r in caplog.records if r.getMessage() == "dataset_loaded"]
    assert (record.kind, record.rows, record.columns) == ("csv", 1, 1)
    assert record.duration_ms >= 0
    text = " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)
    for forbidden in ("Secret", "zebra", "confidential", "payroll"):
        assert forbidden not in text
