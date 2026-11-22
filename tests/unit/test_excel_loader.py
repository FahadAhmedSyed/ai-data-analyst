from __future__ import annotations

import io
import time
import zipfile
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import openpyxl
import pytest

from analyst.config.settings import Settings
from analyst.core.errors import IngestionError
from analyst.ingestion.excel_loader import cell_to_text, list_sheets, load_excel
from analyst.ingestion.file_loader import file_kind, load_file

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "excel"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def workbook_bytes(build) -> bytes:
    wb = openpyxl.Workbook()
    build(wb)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def rainfall_book(wb: openpyxl.Workbook) -> None:
    ws = wb.active
    ws.title = "Daily"
    ws.append(["Rainfall report 2024"])
    ws.merge_cells("A1:E1")
    ws.append([])
    ws.append(["Date", "Site", "Rainfall (mm)", "Checked", "Logged at"])
    ws.append([date(2024, 4, 3), "Site A", 1.5, True, datetime(2024, 4, 3, 9, 30)])
    ws.append([date(2024, 4, 4), "Site B", 3, False, datetime(2024, 4, 4, 10, 0)])
    ws.append([date(2024, 4, 5), "Site C", None, None, None])
    sites = wb.create_sheet("Sites")
    sites.append(["Site", "Region"])
    sites.append(["Site A", "Canterbury"])
    hidden = wb.create_sheet("Lookup")
    hidden.sheet_state = "hidden"
    hidden.append(["x"])


# ---------- cell conversion ----------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("Site A", "Site A"),
        (True, "true"),
        (False, "false"),
        (3.0, "3"),
        (-12.0, "-12"),
        (1.5, "1.5"),
        (0.1, "0.1"),
        (1e20, "1e+20"),
        (float("nan"), None),
        (float("inf"), None),
        (42, "42"),
        (date(2024, 4, 3), "2024-04-03"),
        (datetime(2024, 4, 3, 9, 30), "2024-04-03 09:30:00"),
        (dtime(10, 0), "10:00:00"),
        (timedelta(hours=1, minutes=30), "1:30:00"),
    ],
)
def test_cell_to_text(value, expected):
    assert cell_to_text(value) == expected


def test_cell_to_text_other_types():
    class Odd:
        def __str__(self):
            return "odd"

    assert cell_to_text(Odd()) == "odd"


# ---------- sheets ----------


def test_list_sheets_with_visibility_and_size():
    data = workbook_bytes(rainfall_book)
    sheets = list_sheets(data, settings=settings())
    assert [(s.name, s.visible) for s in sheets] == [
        ("Daily", True),
        ("Sites", True),
        ("Lookup", False),
    ]
    assert sheets[0].rows == 6
    assert sheets[0].columns == 5


def test_default_sheet_is_first_visible():
    def build(wb):
        wb.active.title = "Hidden first"
        wb.active.append(["secret"])
        wb.active.sheet_state = "hidden"
        visible = wb.create_sheet("Data")
        visible.append(["A", "B"])
        visible.append(["1", "2"])
        wb.active = 1

    table = load_excel(workbook_bytes(build), "x.xlsx", settings=settings())
    assert table.sheet_name == "Data"


def test_named_sheet():
    table = load_excel(workbook_bytes(rainfall_book), "x.xlsx", sheet="Sites", settings=settings())
    assert table.sheet_name == "Sites"
    assert table.dataframe.to_dict(orient="list") == {"Site": ["Site A"], "Region": ["Canterbury"]}


def test_unknown_sheet():
    with pytest.raises(IngestionError) as info:
        load_excel(workbook_bytes(rainfall_book), "x.xlsx", sheet="Nope", settings=settings())
    assert info.value.category == "corrupt"
    assert "Nope" in info.value.user_message


# ---------- content ----------


def test_title_rows_skipped_and_values_converted():
    table = load_excel(workbook_bytes(rainfall_book), "rain.xlsx", settings=settings())
    df = table.dataframe
    assert table.header_row == 2
    assert table.skipped_rows == 2
    assert table.encoding is None and table.delimiter is None
    assert list(df.columns) == ["Date", "Site", "Rainfall (mm)", "Checked", "Logged at"]
    assert df["Date"].tolist() == ["2024-04-03", "2024-04-04", "2024-04-05"]
    assert df["Rainfall (mm)"].tolist()[:2] == ["1.5", "3"]
    assert df["Rainfall (mm)"].isna().tolist() == [False, False, True]
    assert df["Checked"].tolist()[:2] == ["true", "false"]
    assert df["Logged at"].tolist()[0] == "2024-04-03 09:30:00"
    assert all(str(dt) == "string" for dt in df.dtypes)


def test_header_override_and_no_header():
    data = workbook_bytes(rainfall_book)
    table = load_excel(data, "x.xlsx", header_row=3, settings=settings())
    assert list(table.dataframe.columns)[1] == "Site A"
    table = load_excel(data, "x.xlsx", sheet="Sites", detect_header=False, settings=settings())
    assert list(table.dataframe.columns) == ["Column 1", "Column 2"]
    assert len(table.dataframe) == 2


def test_legacy_xls_fixture():
    data = (FIXTURES / "simple.xls").read_bytes()
    assert [s.name for s in list_sheets(data, settings=settings())] == ["Readings", "Notes"]
    table = load_excel(data, "simple.xls", settings=settings())
    assert table.sheet_name == "Readings"
    assert table.dataframe.to_dict(orient="list") == {
        "Site": ["Site A", "Site B"],
        "Date": ["2024-04-03", "2024-04-04"],
        "Rainfall (mm)": ["1.5", "12"],
    }


# ---------- failures ----------


def test_empty_sheet():
    def build(wb):
        wb.active.title = "Empty"

    with pytest.raises(IngestionError) as info:
        load_excel(workbook_bytes(build), "x.xlsx", settings=settings())
    assert info.value.category == "empty"


def test_header_only_sheet():
    def build(wb):
        wb.active.append(["Site", "Value"])

    with pytest.raises(IngestionError) as info:
        load_excel(workbook_bytes(build), "x.xlsx", settings=settings())
    assert info.value.category == "empty"


def test_zero_bytes():
    with pytest.raises(IngestionError) as info:
        load_excel(b"", "x.xlsx", settings=settings())
    assert info.value.category == "empty"


def test_csv_named_as_excel():
    with pytest.raises(IngestionError) as info:
        load_excel(b"Site,Value\nA,1\n", "x.xlsx", settings=settings())
    assert info.value.category == "corrupt"
    assert "CSV" in info.value.user_message


def test_damaged_zip():
    with pytest.raises(IngestionError) as info:
        load_excel(b"PK\x03\x04" + b"\x00" * 200, "x.xlsx", settings=settings())
    assert info.value.category == "corrupt"


def test_zip_that_is_not_a_workbook():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("hello.txt", "not a workbook")
    with pytest.raises(IngestionError) as info:
        load_excel(buf.getvalue(), "x.xlsx", settings=settings())
    assert info.value.category == "corrupt"


def test_zip_bomb_is_rejected_before_parsing():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"0" * (11 * 1024 * 1024))
    data = buf.getvalue()
    assert len(data) < 100_000
    with pytest.raises(IngestionError) as info:
        load_excel(data, "x.xlsx", settings=settings(max_upload_mb=1))
    assert info.value.category == "too_large"
    assert "expands" in info.value.user_message


def test_too_large_file():
    data = workbook_bytes(rainfall_book)
    with pytest.raises(IngestionError) as info:
        load_excel(data + b"\x00" * (1024 * 1024), "x.xlsx", settings=settings(max_upload_mb=1))
    assert info.value.category == "too_large"


def test_too_many_rows():
    def build(wb):
        wb.active.append(["n"])
        for i in range(12):
            wb.active.append([i])

    with pytest.raises(IngestionError) as info:
        load_excel(workbook_bytes(build), "x.xlsx", settings=settings(max_rows=10))
    assert info.value.category == "too_many_rows"


# ---------- dispatch ----------


@pytest.mark.parametrize(
    ("name", "kind"),
    [("a.csv", "csv"), ("A.TXT", "csv"), ("b.xlsx", "excel"), ("c.XLS", "excel")],
)
def test_file_kind(name, kind):
    assert file_kind(name) == kind


@pytest.mark.parametrize("name", ["a.json", "b.parquet", "noext", "c.xlsm", "d.csv.exe"])
def test_unsupported_extensions(name):
    with pytest.raises(IngestionError) as info:
        file_kind(name)
    assert info.value.category == "unsupported_type"


def test_load_file_dispatches():
    csv_table = load_file(b"a,b\n1,2\n", "x.csv", settings=settings())
    assert csv_table.delimiter == ","
    xl_table = load_file(
        workbook_bytes(rainfall_book), "x.xlsx", sheet="Sites", settings=settings()
    )
    assert xl_table.sheet_name == "Sites"


# ---------- performance ----------


@pytest.mark.slow
def test_large_workbook_reads_quickly():
    def build(wb):
        ws = wb.active
        ws.append(["Date", "Site", "Rain", "Temp", "Flow", "Flag", "Note"])
        for i in range(100_000):
            ws.append([date(2020, 1, 1), "Site A", i * 0.1, 14.2, 2.345, "Good", "ok"])

    data = workbook_bytes(build)
    start = time.perf_counter()
    table = load_excel(data, "big.xlsx", settings=settings())
    elapsed = time.perf_counter() - start
    assert len(table.dataframe) == 100_000
    assert elapsed < 5, f"took {elapsed:.1f}s"
