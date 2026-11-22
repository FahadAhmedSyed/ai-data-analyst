from __future__ import annotations

import codecs
import time

import pytest

from analyst.config.settings import Settings
from analyst.core.errors import IngestionError
from analyst.ingestion.csv_loader import detect_delimiter, detect_encoding, load_csv

ROWS = [
    ["Site", "Date", "Rainfall (mm)", "Notes"],
    ["Site A", "03/04/2024", "1.5", "café closed"],
    ["Site B", "04/04/2024", "", "naïve sensor; reset"],
    ["Site C", "05/04/2024", "12.0", "ok"],
]


def make_csv(rows=ROWS, delimiter=",", encoding="utf-8", bom=b"", quote_all=False) -> bytes:
    lines = []
    for row in rows:
        cells = [f'"{c}"' if quote_all or delimiter in c or '"' in c else c for c in row]
        lines.append(delimiter.join(cells))
    return bom + ("\r\n".join(lines) + "\r\n").encode(encoding)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# ---------- encodings ----------


@pytest.mark.parametrize(
    ("encoding", "bom", "expected"),
    [
        ("utf-8", b"", "utf-8"),
        ("utf-8", codecs.BOM_UTF8, "utf-8-sig"),
        ("utf-16-le", codecs.BOM_UTF16_LE, "utf-16"),
    ],
)
def test_detects_unicode_encodings(encoding, bom, expected):
    data = make_csv(encoding=encoding, bom=bom)
    assert detect_encoding(data) == expected
    table = load_csv(data, "x.csv", settings=settings())
    assert table.encoding == expected
    assert list(table.dataframe.columns) == ROWS[0]
    assert table.dataframe["Notes"].tolist() == ["café closed", "naïve sensor; reset", "ok"]


@pytest.mark.parametrize("encoding", ["cp1252", "latin-1"])
def test_reads_single_byte_western_encodings(encoding):
    rows = [*ROWS, ["Site D", "06/04/2024", "2", "déjà vu, Müller"]]
    data = make_csv(rows=rows, encoding=encoding)
    assert detect_encoding(data) == "cp1252"
    table = load_csv(data, "x.csv", settings=settings())
    notes = table.dataframe["Notes"].tolist()
    assert notes == ["café closed", "naïve sensor; reset", "ok", "déjà vu, Müller"]


def test_falls_back_to_statistical_detection():
    # 0x81 is undefined in Windows-1252, so this must go to charset-normalizer.
    text = "Имя,Город\nАнна,Москва\n" * 20
    data = text.encode("cp1251")
    data = data + "Ѓ".encode("cp1251")  # 0x81 in cp1251
    encoding = detect_encoding(data)
    assert encoding not in ("utf-8", "cp1252")
    assert data.decode(encoding).startswith("Имя")


def test_windows_1252_smart_quotes():
    comment = "\u201cgreat\u201d service \u2013 thanks"
    rows = [["Name", "Comment"]] + [["A", comment]] * 50
    data = make_csv(rows=rows, encoding="cp1252")
    table = load_csv(data, "x.csv", settings=settings())
    assert table.dataframe["Comment"].iloc[0] == comment


# ---------- delimiters ----------


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_detects_delimiters(delimiter):
    data = make_csv(delimiter=delimiter)
    assert detect_delimiter(data.decode()) == delimiter
    table = load_csv(data, "x.csv", settings=settings())
    assert table.delimiter == delimiter
    assert table.dataframe.shape == (3, 4)


def test_delimiter_inside_quotes_is_ignored():
    rows = [["Name", "Address", "Amount"]] + [["A", "1 Main St, Wellington", "5"]] * 10
    data = make_csv(rows=rows, delimiter=";", quote_all=True)
    assert detect_delimiter(data.decode()) == ";"
    table = load_csv(data, "x.csv", settings=settings())
    assert table.dataframe["Address"].iloc[0] == "1 Main St, Wellington"


def test_single_column_file():
    data = b"Site\nA\nB\n"
    assert detect_delimiter(data.decode()) == ","
    table = load_csv(data, "x.csv", settings=settings())
    assert list(table.dataframe.columns) == ["Site"]
    assert table.dataframe["Site"].tolist() == ["A", "B"]


def test_delimiter_override():
    data = b"a;b\n1;2\n"
    table = load_csv(data, "x.csv", delimiter=",", settings=settings())
    assert list(table.dataframe.columns) == ["a;b"]


# ---------- structure ----------


def test_quoted_newlines_and_quotes():
    data = b'Name,Comment\nA,"line one\nline two"\nB,"she said ""hi"""\n'
    table = load_csv(data, "x.csv", settings=settings())
    assert table.dataframe["Comment"].tolist() == ["line one\nline two", 'she said "hi"']


def test_values_are_text_and_blanks_are_missing():
    table = load_csv(make_csv(), "x.csv", settings=settings())
    df = table.dataframe
    assert all(str(dt) == "string" for dt in df.dtypes)
    assert df["Rainfall (mm)"].tolist()[0] == "1.5"
    assert df["Rainfall (mm)"].isna().tolist() == [False, True, False]
    assert df["Date"].tolist()[0] == "03/04/2024"  # untouched: typing happens later


def test_title_rows_above_header_are_skipped():
    data = b"Rainfall report,,\nGenerated 2024-05-01,,\n,,\nSite,Date,Rain\nA,1/1/2024,3\n"
    table = load_csv(data, "x.csv", settings=settings())
    assert list(table.dataframe.columns) == ["Site", "Date", "Rain"]
    assert table.header_row == 3  # the ",," row counts as a row of missing values
    assert table.skipped_rows == 3
    assert len(table.dataframe) == 1


def test_header_row_override():
    data = b"a,b\nc,d\n1,2\n"
    table = load_csv(data, "x.csv", header_row=1, settings=settings())
    assert list(table.dataframe.columns) == ["c", "d"]
    assert table.dataframe.iloc[0].tolist() == ["1", "2"]


def test_headerless_numeric_file_gets_generated_names():
    data = b"1,2,3\n4,5,6\n"
    table = load_csv(data, "x.csv", settings=settings())
    assert table.header_row is None
    assert list(table.dataframe.columns) == ["Column 1", "Column 2", "Column 3"]
    assert len(table.dataframe) == 2


def test_no_header_detection_when_disabled():
    data = b"Site,Value\nA,1\n"
    table = load_csv(data, "x.csv", detect_header=False, settings=settings())
    assert list(table.dataframe.columns) == ["Column 1", "Column 2"]
    assert len(table.dataframe) == 2


def test_duplicate_and_blank_headers():
    data = b"Site,,Site,Value\nA,x,B,1\n"
    table = load_csv(data, "x.csv", settings=settings())
    assert list(table.dataframe.columns) == ["Site", "Column 2", "Site (2)", "Value"]


def test_empty_rows_and_trailing_empty_columns_dropped():
    data = b"Site,Value,,\nA,1,,\n,,,\nB,2,,\n"
    table = load_csv(data, "x.csv", settings=settings())
    assert list(table.dataframe.columns) == ["Site", "Value"]
    assert table.dataframe["Site"].tolist() == ["A", "B"]


# ---------- failures ----------


@pytest.mark.parametrize("data", [b"", b"   \n\n  "])
def test_empty_file(data):
    with pytest.raises(IngestionError) as info:
        load_csv(data, "x.csv", settings=settings())
    assert info.value.category == "empty"


def test_header_only_file():
    with pytest.raises(IngestionError) as info:
        load_csv(b"Site,Value\n", "x.csv", settings=settings())
    assert info.value.category == "empty"


def test_too_large():
    data = b"a\n" + b"x\n" * (1024 * 1024)
    with pytest.raises(IngestionError) as info:
        load_csv(data, "x.csv", settings=settings(max_upload_mb=1))
    assert info.value.category == "too_large"
    assert "1 MB" in info.value.user_message


def test_too_many_rows():
    data = b"a\n" + b"1\n" * 11
    load_csv(b"a\n" + b"1\n" * 10, "x.csv", settings=settings(max_rows=10))
    with pytest.raises(IngestionError) as info:
        load_csv(data, "x.csv", settings=settings(max_rows=10))
    assert info.value.category == "too_many_rows"


def test_too_many_columns():
    data = (",".join(f"c{i}" for i in range(6)) + "\n" + ",".join("1" * 6) + "\n").encode()
    with pytest.raises(IngestionError) as info:
        load_csv(data, "x.csv", settings=settings(max_columns=5))
    assert info.value.category == "too_many_columns"


@pytest.mark.parametrize("magic", [b"PK\x03\x04", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"])
def test_excel_file_with_csv_name(magic):
    with pytest.raises(IngestionError) as info:
        load_csv(magic + b"\x00" * 100, "x.csv", settings=settings())
    assert info.value.category == "corrupt"
    assert "Excel" in info.value.user_message


def test_binary_garbage():
    with pytest.raises(IngestionError) as info:
        load_csv(b"abc\x00def\x00" * 50, "x.csv", settings=settings())
    assert info.value.category == "corrupt"


def test_ragged_rows():
    data = b"a,b\n1,2\n3,4,5\n"
    with pytest.raises(IngestionError) as info:
        load_csv(data, "x.csv", settings=settings())
    assert info.value.category == "corrupt"
    assert "more fields" in info.value.user_message


def test_undecodable_bytes_raise_encoding_error(monkeypatch):
    from analyst.ingestion import csv_loader

    class NoMatch:
        def best(self):
            return None

    monkeypatch.setattr(csv_loader, "from_bytes", lambda _: NoMatch())
    with pytest.raises(IngestionError) as info:
        load_csv(b"\xff\xfe\xfa bad", "x.csv", settings=settings())
    assert info.value.category == "encoding"


def test_wrong_detected_encoding_is_reported(monkeypatch):
    from analyst.ingestion import csv_loader

    monkeypatch.setattr(csv_loader, "detect_encoding", lambda _: "ascii")
    with pytest.raises(IngestionError) as info:
        load_csv("é,b\n1,2\n".encode("latin-1"), "x.csv", settings=settings())
    assert info.value.category == "encoding"


# ---------- performance ----------


@pytest.mark.slow
def test_50_mb_file_loads_within_target():
    row = "03/04/2024 10:15,Site A,1.5,14.2,2.345,Good,maintenance\n"
    header = "Date,Site,Rainfall (mm),Temperature (C),Flow (m3/s),Quality,Notes\n"
    data = (header + row * (50 * 1024 * 1024 // len(row) - 1)).encode()
    start = time.perf_counter()
    table = load_csv(data, "big.csv", settings=settings())
    elapsed = time.perf_counter() - start
    assert table.dataframe.shape[1] == 7
    assert elapsed < 10, f"took {elapsed:.1f}s"
