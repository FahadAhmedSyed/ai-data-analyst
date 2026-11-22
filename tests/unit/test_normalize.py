"""Type inference and identifier tests (T1.05).

Every expected value below was worked out by hand from the input text, never by
running the code under test.
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime

import pandas as pd
import pytest

from analyst.config.settings import Settings
from analyst.core.logging import configure_logging
from analyst.ingestion.csv_loader import load_csv
from analyst.ingestion.normalize import (
    MAX_IDENTIFIER_LENGTH,
    infer_column,
    make_identifiers,
    normalize,
    to_identifier,
)
from analyst.ingestion.tabular import RawTable

NA = pd.NA
T = pd.Timestamp


def text(*values: str | None) -> pd.Series:
    return pd.Series(list(values), dtype="string")


def as_list(series: pd.Series) -> list:
    return [None if pd.isna(v) else v for v in series.tolist()]


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# ---------- identifiers ----------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Rainfall (mm)", "rainfall_mm"),
        ("Max Temp (°C)", "max_temp_c"),
        ("Unit Price ($)", "unit_price"),
        ("Total-Sales%", "total_sales"),
        ("x__y", "x_y"),
        ("Māori name", "maori_name"),
        ("Café", "cafe"),
        ("2024 total", "c_2024_total"),
        ("Order", "order_col"),
        ("select", "select_col"),
        ("GROUP", "group_col"),
        ("Date", "date"),
        ("Month", "month"),
        ("   ", "col"),
        ("温度", "col"),
        ("%", "col"),
        ("a" * 100, "a" * MAX_IDENTIFIER_LENGTH),
        ("a" * 62 + " b", "a" * 62),
    ],
)
def test_to_identifier(name, expected):
    assert to_identifier(name) == expected


# The SQL names the benchmark's gold queries use (eval/README.md, section 1).
BENCHMARK_HEADERS = {
    "Order ID": "order_id",
    "Order Date": "order_date",
    "Region": "region",
    "Product": "product",
    "Category": "category",
    "Quantity": "quantity",
    "Unit Price ($)": "unit_price",
    "Revenue ($)": "revenue",
    "Customer ID": "customer_id",
    "Respondent ID": "respondent_id",
    "Submitted At": "submitted_at",
    "Age group": "age_group",
    "Channel": "channel",
    "Satisfaction": "satisfaction",
    "Likelihood to recommend": "likelihood_to_recommend",
    "Comment": "comment",
    "Date": "date",
    "Site": "site",
    "Rainfall (mm)": "rainfall_mm",
    "Max Temp (°C)": "max_temp_c",
    "Min Temp (°C)": "min_temp_c",
    "Sample ID": "sample_id",
    "Sampled At": "sampled_at",
    "Parameter": "parameter",
    "Result": "result",
    "Unit": "unit",
    "Lab": "lab",
    "Month": "month",
    "Department": "department",
    "Budget": "budget",
    "Actual": "actual",
    "Incident ID": "incident_id",
    "Reported On": "reported_on",
    "Severity": "severity",
    "Resolved On": "resolved_on",
    "Cost": "cost",
}


@pytest.mark.parametrize(("header", "expected"), BENCHMARK_HEADERS.items())
def test_benchmark_headers_map_to_the_documented_names(header, expected):
    assert to_identifier(header) == expected


def test_make_identifiers_dedupes_in_order():
    assert make_identifiers(["Site", "site", "SITE"]) == ["site", "site_2", "site_3"]
    assert make_identifiers(["Rainfall (mm)", "Rainfall mm"]) == ["rainfall_mm", "rainfall_mm_2"]
    assert make_identifiers(["x", "x_2", "x"]) == ["x", "x_2", "x_3"]
    assert make_identifiers(["温度", "湿度"]) == ["col", "col_2"]


def test_make_identifiers_keeps_long_duplicates_within_the_limit():
    names = make_identifiers(["a" * 70, "a" * 70])
    assert names == ["a" * 63, "a" * 61 + "_2"]
    assert all(len(n) <= MAX_IDENTIFIER_LENGTH for n in names)


# ---------- booleans ----------


def test_boolean_words_become_boolean():
    result = infer_column(text("true", "FALSE", " Yes ", "no", None, "N/A"))
    assert result.dtype == "boolean"
    assert as_list(result.series) == [True, False, True, False, None, None]
    assert result.flags == ()


@pytest.mark.parametrize(
    ("values", "dtype"),
    [(("Y", "N", "Y"), "text"), (("1", "0", "1"), "integer")],
)
def test_letters_and_digits_are_not_treated_as_boolean(values, dtype):
    assert infer_column(text(*values)).dtype == dtype


# ---------- numbers ----------


@pytest.mark.parametrize(
    ("values", "dtype", "expected", "flags"),
    [
        (("1", "2", "-3"), "integer", [1, 2, -3], ()),
        (("+5", "-3", "0"), "integer", [5, -3, 0], ()),
        (("1.5", "2", "3.25"), "decimal", [1.5, 2.0, 3.25], ()),
        (("0", "0.5", "10"), "decimal", [0.0, 0.5, 10.0], ()),
        (("1e3", "2"), "decimal", [1000.0, 2.0], ()),
        (("$1,234.50", "$99", "NZ$5"), "decimal", [1234.5, 99.0, 5.0], ("numeric_text",)),
        (("1,234", "12,345,678", "7"), "integer", [1234, 12345678, 7], ("numeric_text",)),
        (("(1,234.50)", "200"), "decimal", [-1234.5, 200.0], ("numeric_text",)),
        (("12%", "7.5%"), "decimal", [12.0, 7.5], ("numeric_text",)),
        (("€3", "£4", "¥5"), "integer", [3, 4, 5], ("numeric_text",)),
        # 2**53 + 1 cannot be held by a float; it must survive exactly.
        (("9007199254740993", "1"), "integer", [9007199254740993, 1], ()),
    ],
)
def test_numbers(values, dtype, expected, flags):
    result = infer_column(text(*values))
    assert result.dtype == dtype
    assert as_list(result.series) == expected
    assert result.flags == flags
    assert result.note.coerced_to_missing == 0


def test_integer_and_decimal_columns_use_nullable_dtypes():
    assert str(infer_column(text("1", None)).series.dtype) == "Int64"
    assert str(infer_column(text("1.5", None)).series.dtype) == "Float64"


def test_percent_note_says_how_values_are_stored():
    assert "12% is stored as 12" in infer_column(text("12%", "5%")).note.detail


def test_missing_markers_in_a_number_column_become_missing_without_a_flag():
    result = infer_column(text("1", "N/A", "-", "", None, "  ", "#DIV/0!", "3"))
    assert result.dtype == "integer"
    assert as_list(result.series) == [1, None, None, None, None, None, None, 3]
    assert result.flags == ()
    assert result.note.coerced_to_missing == 0


def test_codes_with_leading_zeros_stay_text():
    result = infer_column(text("007", "123", "0800"))
    assert result.dtype == "text"
    assert as_list(result.series) == ["007", "123", "0800"]
    assert result.flags == ()
    assert "leading zeros" in result.note.detail


def test_numbers_too_long_for_64_bits_stay_text():
    result = infer_column(text("12345678901234567890", "1"))
    assert result.dtype == "text"
    assert result.flags == ()


def test_decimal_commas_are_not_read_as_thousands():
    result = infer_column(text("1,5", "2,25"))
    assert result.dtype == "text"
    assert as_list(result.series) == ["1,5", "2,25"]


def test_a_few_non_numbers_are_coerced_and_counted():
    values = [str(i) for i in range(1, 20)] + ["unknown"]  # 19 of 20 parse = 95%
    result = infer_column(text(*values))
    assert result.dtype == "integer"
    assert as_list(result.series) == [*range(1, 20), None]
    assert result.flags == ("mixed_types",)
    assert result.note.coerced_to_missing == 1


def test_too_many_non_numbers_keep_the_column_as_text():
    values = [str(i) for i in range(1, 19)] + ["unknown", "pending"]  # 18 of 20 = 90%
    result = infer_column(text(*values))
    assert result.dtype == "text"
    assert as_list(result.series) == values
    assert result.flags == ("mixed_types",)
    assert "90%" in result.note.detail


# ---------- censored values ----------


def test_censored_values_keep_the_column_as_text():
    result = infer_column(text("<0.5", "1.2", "ND", "3.4", ">100"))
    assert result.dtype == "text"
    assert as_list(result.series) == ["<0.5", "1.2", "ND", "3.4", ">100"]
    assert result.flags == ("censored_values",)
    assert "3 censored values" in result.note.detail


def test_one_censored_value_is_never_silently_dropped():
    values = [f"{i}.5" for i in range(25)] + ["<0.5"]  # 25 of 26 parse, above 95%
    result = infer_column(text(*values))
    assert result.dtype == "text"
    assert result.flags == ("censored_values",)
    assert as_list(result.series)[-1] == "<0.5"


# ---------- dates ----------


@pytest.mark.parametrize(
    ("values", "dtype", "expected", "flags"),
    [
        (("2024-01-31", "2024-02-29"), "date", [T(2024, 1, 31), T(2024, 2, 29)], ()),
        (
            ("2024-01-31 10:15:00", "2024-02-29 00:00:00"),
            "timestamp",
            [T(2024, 1, 31, 10, 15), T(2024, 2, 29)],
            (),
        ),
        # Excel date cells arrive as midnight timestamps; they are dates.
        (
            ("2024-01-31 00:00:00", "2024-02-01 00:00:00"),
            "date",
            [T(2024, 1, 31), T(2024, 2, 1)],
            (),
        ),
        # Offsets are converted to UTC: 10:00 at +13:00 is 21:00 the day before.
        (
            ("2024-01-01T10:00:00+13:00", "2024-06-01T10:00:00Z"),
            "timestamp",
            [T(2023, 12, 31, 21), T(2024, 6, 1, 10)],
            (),
        ),
        (("2024-03", "2024-12"), "date", [T(2024, 3, 1), T(2024, 12, 1)], ()),
        (("13/02/2024", "03/04/2024"), "date", [T(2024, 2, 13), T(2024, 4, 3)], ()),
        (("02/13/2024", "03/04/2024"), "date", [T(2024, 2, 13), T(2024, 3, 4)], ()),
        (("13.02.2024", "1.3.2024"), "date", [T(2024, 2, 13), T(2024, 3, 1)], ()),
        (("13-02-2024", "01-03-2024"), "date", [T(2024, 2, 13), T(2024, 3, 1)], ()),
        (("13/02/24", "01/03/24"), "date", [T(2024, 2, 13), T(2024, 3, 1)], ()),
        (
            ("13/02/2024 09:30", "01/03/2024 17:05"),
            "timestamp",
            [T(2024, 2, 13, 9, 30), T(2024, 3, 1, 17, 5)],
            (),
        ),
        (
            ("13/02/2024 9:30 PM", "01/03/2024 10:05 AM"),
            "timestamp",
            [T(2024, 2, 13, 21, 30), T(2024, 3, 1, 10, 5)],
            (),
        ),
        (
            ("5 March 2024", "March 6, 2024", "Sep 2024"),
            "date",
            [T(2024, 3, 5), T(2024, 3, 6), T(2024, 9, 1)],
            (),
        ),
        # A far-future placeholder must not overflow.
        (("2024-01-01", "9999-12-31"), "date", [T(2024, 1, 1), T(9999, 12, 31)], ()),
    ],
)
def test_dates(values, dtype, expected, flags):
    result = infer_column(text(*values))
    assert result.dtype == dtype
    assert as_list(result.series) == expected
    assert result.flags == flags


def test_ambiguous_dates_follow_the_region_default_and_are_flagged():
    day_first = infer_column(text("01/02/2024", "03/04/2024"), day_first_default=True)
    assert as_list(day_first.series) == [T(2024, 2, 1), T(2024, 4, 3)]
    assert day_first.flags == ("ambiguous_date",)
    assert "region default" in day_first.note.detail

    month_first = infer_column(text("01/02/2024", "03/04/2024"), day_first_default=False)
    assert as_list(month_first.series) == [T(2024, 1, 2), T(2024, 3, 4)]
    assert month_first.flags == ("ambiguous_date",)


def test_confirmed_order_is_described():
    assert "confirmed by the values" in infer_column(text("13/02/2024", "01/03/2024")).note.detail


def test_a_value_after_the_sample_can_settle_the_order():
    # 2,016 distinct month-first dates whose parts are all 12 or less, then one
    # that can only be month-first. The detection sample is the first 2,000.
    ambiguous = [
        f"{m:02d}/{d:02d}/{y}"
        for y in range(2000, 2014)
        for m in range(1, 13)
        for d in range(1, 13)
    ]
    values = [*ambiguous, "01/13/2024"]
    result = infer_column(text(*values), day_first_default=True)
    assert result.dtype == "date"
    assert result.flags == ()
    assert as_list(result.series)[0] == T(2000, 1, 1)
    assert as_list(result.series)[1] == T(2000, 1, 2)
    assert as_list(result.series)[-1] == T(2024, 1, 13)


def test_dates_with_invalid_values_are_coerced_and_counted():
    values = [f"{d:02d}/01/2024" for d in range(1, 21)] + ["31/02/2024"]  # 20 of 21 parse
    result = infer_column(text(*values))
    assert result.dtype == "date"
    assert result.flags == ("mixed_types",)
    assert result.note.coerced_to_missing == 1
    assert as_list(result.series)[-1] is None


def test_values_shaped_like_dates_that_mostly_do_not_exist_stay_text():
    # Every value has the dd/mm/yyyy shape, but 30 February and 31 April/June do
    # not exist, so only 1 of 5 parses: far below 95%.
    values = ("30/02/2024", "31/04/2024", "31/06/2024", "30/02/2023", "01/01/2024")
    result = infer_column(text(*values))
    assert result.dtype == "text"
    assert as_list(result.series) == list(values)


@pytest.mark.parametrize(
    "values",
    [
        ("January", "February", "March"),  # no year: would otherwise become year 1
        ("13/02/2024", "02/13/2024"),  # contradictory orders
        ("13/02/2024", "14-02-2024"),  # mixed separators
        ("2023", "2024"),  # years are numbers
    ],
)
def test_values_that_are_not_one_date_layout(values):
    assert infer_column(text(*values)).dtype in {"text", "integer"}


def test_years_alone_are_integers():
    assert infer_column(text("2023", "2024")).dtype == "integer"


def test_date_columns_use_microsecond_datetimes():
    assert str(infer_column(text("2024-01-01", None)).series.dtype) == "datetime64[us]"


# ---------- text, constant and missing ----------


def test_text_is_stripped_and_blank_becomes_missing():
    result = infer_column(text(" hello ", "world", "", None))
    assert result.dtype == "text"
    assert as_list(result.series) == ["hello", "world", None, None]
    assert result.flags == ()


def test_missing_markers_in_text_are_kept_as_written():
    result = infer_column(text("apple", "N/A", "pear"))
    assert as_list(result.series) == ["apple", "N/A", "pear"]


def test_all_missing_column():
    result = infer_column(text(None, "", "  ", "NA", "n/a"))
    assert result.dtype == "text"
    assert result.flags == ("all_missing",)
    assert as_list(result.series) == [None] * 5


@pytest.mark.parametrize(
    ("values", "dtype", "flags"),
    [
        (("a", "a", None), "text", ("constant",)),
        (("5", "5", "5"), "integer", ("constant",)),
        (("5",), "integer", ()),
        (("2024-01-01", "2024-01-01"), "date", ("constant",)),
    ],
)
def test_constant_flag(values, dtype, flags):
    result = infer_column(text(*values))
    assert (result.dtype, result.flags) == (dtype, flags)


def test_output_keeps_the_input_index():
    series = pd.Series(["1", None, "3"], index=[10, 11, 12], dtype="string")
    for values in (series, series.where(series.isna(), "2024-01-0" + series)):
        assert list(infer_column(values).series.index) == [10, 11, 12]


# ---------- whole table ----------


def raw_table(frame: pd.DataFrame, sheet: str | None = None) -> RawTable:
    return RawTable(
        dataframe=frame.astype("string"),
        source_name="orders.csv",
        sheet_name=sheet,
        encoding="utf-8",
        delimiter=",",
        header_row=0,
        skipped_rows=0,
    )


def test_normalize_builds_schemas_and_renames_columns():
    frame = pd.DataFrame(
        {
            "Order ID": ["1", "2", "3"],
            "Order Date": ["13/01/2024", "02/02/2024", "28/02/2024"],
            "Region": ["North", None, "South"],
            "Revenue ($)": ["$1,200.00", "$35.50", "$0.99"],
            "Order": ["a", "b", "c"],
            "order": ["x", "y", "z"],
        }
    )
    table = normalize(raw_table(frame, sheet="Orders"), settings=settings())

    assert [c.sql_name for c in table.columns] == [
        "order_id",
        "order_date",
        "region",
        "revenue",
        "order_col",
        "order_col_2",
    ]
    assert [c.display_name for c in table.columns] == list(frame.columns)
    assert [c.dtype for c in table.columns] == [
        "integer",
        "date",
        "text",
        "decimal",
        "text",
        "text",
    ]
    assert table.columns[3].flags == ("numeric_text",)
    assert list(table.dataframe.columns) == [c.sql_name for c in table.columns]
    assert as_list(table.dataframe["order_date"]) == [T(2024, 1, 13), T(2024, 2, 2), T(2024, 2, 28)]
    assert as_list(table.dataframe["revenue"]) == [1200.0, 35.5, 0.99]
    assert set(table.notes) == {c.sql_name for c in table.columns}
    assert (table.source_name, table.sheet_name) == ("orders.csv", "Orders")


def test_normalize_logs_counts_but_no_names_or_values():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    frame = pd.DataFrame({"Secret Header": ["confidential-value", "other"], "N": ["1", "2"]})
    normalize(raw_table(frame), settings=settings())
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    typed = next(line for line in lines if line["event"] == "table_typed")
    assert typed["columns"] == 2
    assert typed["types"] == {"text": 1, "integer": 1}
    assert "confidential" not in stream.getvalue()
    assert "Secret" not in stream.getvalue()


def test_csv_to_typed_table_end_to_end():
    csv = (
        "﻿Order ID,Order Date,Region,Quantity,Unit Price ($),Customer ID\r\n"
        '1,13/01/2023,North,2,"$1,250.00",C001\r\n'
        "2,05/02/2023,,1,$80.50,C002\r\n"
        "3,28/12/2024,West,7,$9.99,C001\r\n"
    ).encode()
    table = normalize(load_csv(csv, "retail_sales.csv", settings=settings()), settings=settings())
    types = {c.sql_name: c.dtype for c in table.columns}
    assert types == {
        "order_id": "integer",
        "order_date": "date",
        "region": "text",
        "quantity": "integer",
        "unit_price": "decimal",
        "customer_id": "text",
    }
    assert as_list(table.dataframe["unit_price"]) == [1250.0, 80.5, 9.99]
    assert as_list(table.dataframe["order_date"]) == [
        datetime(2023, 1, 13),
        datetime(2023, 2, 5),
        datetime(2024, 12, 28),
    ]
    assert as_list(table.dataframe["region"]) == ["North", None, "West"]


@pytest.mark.slow
def test_50_mb_file_loads_and_types_within_target():
    """NFR-6: a 50 MB file loads in under 20 s (read plus typing)."""
    row = '03/04/2024 10:15,Site A,1.5,14.2,"$2,345.00",Good,maintenance\n'
    header = "Date,Site,Rainfall (mm),Temperature (C),Cost,Quality,Notes\n"
    data = (header + row * (50 * 1024 * 1024 // len(row) - 1)).encode()
    start = time.perf_counter()
    table = normalize(load_csv(data, "big.csv", settings=settings()), settings=settings())
    elapsed = time.perf_counter() - start
    types = [c.dtype for c in table.columns]
    assert types == ["timestamp", "text", "decimal", "decimal", "decimal", "text", "text"]
    assert elapsed < 20, f"took {elapsed:.1f}s"


@pytest.mark.slow
def test_50_mb_file_of_mostly_distinct_values_types_within_target():
    """Worst case for typing: almost every value is different, so nothing is shared."""
    lines = ["ID,Recorded,Reading,Cost,Ref,Notes"]
    size = len(lines[0])
    start = datetime(2020, 1, 1)
    i = 0
    while size < 50 * 1024 * 1024 - 200:
        i += 1
        recorded = (start + pd.Timedelta(minutes=i)).strftime("%d/%m/%Y %H:%M")
        cost = f'"${i * 7 % 999_999:,}.{i % 100:02d}"'
        line = f"{i},{recorded},{i * 0.0137:.4f},{cost},R{i:08d},note {i * 7919}"
        lines.append(line)
        size += len(line) + 1
    data = ("\n".join(lines) + "\n").encode()
    start_time = time.perf_counter()
    table = normalize(load_csv(data, "big.csv", settings=settings()), settings=settings())
    elapsed = time.perf_counter() - start_time
    types = [c.dtype for c in table.columns]
    assert types == ["integer", "timestamp", "decimal", "decimal", "text", "text"]
    assert elapsed < 20, f"took {elapsed:.1f}s"
