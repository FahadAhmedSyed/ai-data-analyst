"""What the interface shows (T1.18): escaping, tables, descriptions and the CSV download.

The profile's expected numbers come from pandas on the same values, never from the code under
test (testing.md, section 4).
"""

from __future__ import annotations

import re
import string
from datetime import date, datetime

import pandas as pd
import pytest

from analyst.config.settings import Settings
from analyst.core.models import FilterCondition, Intent, Measure, TimeRange
from analyst.database.session_db import SessionDatabase
from analyst.session.dataset import load_dataset
from analyst.session.display import (
    FLAG_LABELS,
    MAX_TOP_VALUES,
    PREVIEW_ROWS,
    TYPE_LABELS,
    ZERO_WIDTH_SPACE,
    count_text,
    csv_bytes,
    date_text,
    describe_filter,
    describe_intent,
    describe_time_range,
    display_frame,
    escape_dollars,
    escape_markdown,
    number_text,
    profile_tables,
    share_text,
)

RAINFALL = [1.5, 3, None, 4.5, 10]
CSV = (
    b"Site,Rainfall (mm),Recorded,Active\n"
    b"Site A,1.5,2024-04-03,yes\n"
    b"Site A,3,2024-04-04,no\n"
    b"Site B,,2024-04-05,yes\n"
    b"Site B,4.5,2024-04-06,yes\n"
    b"Site C,10,2024-04-07,no\n"
)


@pytest.fixture
def loaded():
    with SessionDatabase(Settings(_env_file=None)) as db:
        yield load_dataset(db, CSV, "rainfall.csv", settings=Settings(_env_file=None)), db


# -- escaping -------------------------------------------------------------------------------------


@pytest.mark.parametrize("mark", list(string.punctuation))
def test_every_punctuation_mark_is_made_literal(mark: str) -> None:
    assert escape_markdown(f"a{mark}b").replace(ZERO_WIDTH_SPACE, "") == f"a\\{mark}b"


@pytest.mark.parametrize(
    "hostile",
    [
        "![x](http://evil.example/p.png?d=1)",
        "[click](http://evil.example)",
        "<img src=x onerror=alert(1)>",
        "http://evil.example/track",
        "# heading",
        "- item",
        "1. item",
        "> quote",
        "`code`",
        "**bold**",
        ":red[colour] :smile:",
        "$a$ and $$b$$",
        "| a | b |",
    ],
)
def test_text_that_would_become_markup_is_escaped_character_by_character(hostile: str) -> None:
    escaped = escape_markdown(hostile)
    position = 0
    while position < len(escaped):
        char = escaped[position]
        if char == "\\":  # an escape: it must be followed by the mark it makes literal
            assert escaped[position + 1] in string.punctuation
            position += 2
        else:
            assert char not in string.punctuation, f"{char!r} at {position} is not escaped"
            position += 1
    # Unescaped, the text is what was typed (but for the invisible marks that break addresses).
    assert escaped.replace("\\", "").replace(ZERO_WIDTH_SPACE, "") == hostile


def as_a_reader_of_markdown_sees_it(escaped: str) -> str:
    """The text once its backslash escapes are read, which is where a link can still form."""
    return re.sub(r"\\(.)", r"\1", escaped)


@pytest.mark.parametrize(
    "address",
    [
        "http://evil.example/track?d=secret",
        "https://evil.example",
        "HTTPS://EVIL.EXAMPLE",
        "see http://a.b and http://c.d",
        "www.evil.example",
        "WWW.evil.example/x",
        "me@evil.example",
        "mailto:me@evil.example",
        "ftp://evil.example/f",
    ],
)
def test_a_web_or_email_address_cannot_become_a_link_when_the_page_reads_it(address: str) -> None:
    """Streamlit linkifies bare addresses after the escapes are read; the text must not match."""
    seen = as_a_reader_of_markdown_sees_it(escape_markdown(address))
    assert not re.search(r"(?i)\w+://|\bwww\.|@[\w-]", seen), seen
    assert seen.replace(ZERO_WIDTH_SPACE, "") == address  # and nothing visible was lost


def test_the_mark_that_breaks_addresses_is_not_put_into_ordinary_text() -> None:
    for text in ("Revenue: 3 of 4", "a/b", "10:30 am", "e-mail us", "www", "www ", "http", "//x"):
        assert ZERO_WIDTH_SPACE not in escape_markdown(text), text


def test_line_breaks_and_runs_of_spaces_become_one_space() -> None:
    assert escape_markdown("a\n\n  b\tc\r\nd") == "a b c d"


def test_leading_spaces_cannot_make_a_code_block() -> None:
    assert escape_markdown("        code") == "code"


def test_ordinary_words_are_left_alone() -> None:
    assert escape_markdown("Revenue in 2024 by Region") == "Revenue in 2024 by Region"


def test_dollar_signs_are_kept_literal_without_touching_other_markdown() -> None:
    assert escape_dollars("$5 and $10, **bold**") == "\\$5 and \\$10, **bold**"


# -- numbers and dates ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "rows", "text"),
    [
        (0, 100, "0%"),
        (5, 0, "0%"),
        (1, 100, "1%"),
        (1, 3, "33.3%"),
        (1, 4, "25%"),
        (1, 10_000, "under 0.1%"),
        (1, 1_000, "0.1%"),  # exactly a tenth of a percent is shown, not called "under"
        (999, 1_000, "99.9%"),  # exactly 99.9% is shown, not called "over"
        (9_999, 10_000, "over 99.9%"),
        (10_000, 10_000, "100%"),
    ],
)
def test_shares_never_round_a_partial_share_to_nothing_or_everything(
    count: int, rows: int, text: str
) -> None:
    assert share_text(count, rows) == text


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, ""),
        (0, "0"),
        (1234, "1,234"),
        (1234.0, "1,234"),
        (-1234567, "-1,234,567"),
        (4.75, "4.75"),
        (1234.5678, "1,234.57"),
        (-2.5, "-2.50"),
        (0.123456, "0.1235"),
        (0.00001234, "1.234e-05"),
        (True, "True"),
    ],
)
def test_numbers_are_written_for_reading(value, text: str) -> None:
    assert number_text(value) == text


def test_dates_and_times_are_iso_text() -> None:
    assert date_text(date(2024, 4, 3)) == "2024-04-03"
    assert date_text(datetime(2024, 4, 3, 9, 30)) == "2024-04-03 09:30:00"
    assert date_text(None) == ""


# -- the profile ----------------------------------------------------------------------------------


def test_the_overview_has_a_row_per_column_in_order_with_original_names(loaded) -> None:
    dataset, _ = loaded
    overview = profile_tables(dataset.profile).overview
    assert overview["Column"].tolist() == ["Site", "Rainfall (mm)", "Recorded", "Active"]
    assert overview["Type"].tolist() == [
        TYPE_LABELS["text"],
        TYPE_LABELS["decimal"],
        TYPE_LABELS["date"],
        TYPE_LABELS["boolean"],
    ]


def test_missing_values_and_distinct_counts_match_pandas(loaded) -> None:
    dataset, _ = loaded
    overview = profile_tables(dataset.profile).overview.set_index("Column")
    series = pd.Series(RAINFALL, dtype="float")
    assert overview.loc["Rainfall (mm)", "Missing"] == str(int(series.isna().sum()))
    assert overview.loc["Rainfall (mm)", "Missing %"] == "20%"
    assert overview.loc["Rainfall (mm)", "Different values"] == str(series.nunique())
    assert overview.loc["Site", "Missing"] == "0"
    assert overview.loc["Site", "Missing %"] == "0%"
    assert overview.loc["Site", "Different values"] == "3"


def test_the_numbers_table_matches_pandas(loaded) -> None:
    dataset, _ = loaded
    numbers = profile_tables(dataset.profile).numbers
    series = pd.Series(RAINFALL, dtype="float")
    assert numbers["Column"].tolist() == ["Rainfall (mm)"]
    row = numbers.iloc[0]
    assert row["Smallest"] == number_text(series.min())
    assert row["Largest"] == number_text(series.max())
    assert row["Average"] == number_text(series.mean())
    assert row["Std deviation"] == number_text(series.std())
    assert row["25%"] == number_text(series.quantile(0.25))
    assert row["Median"] == number_text(series.median())
    assert row["75%"] == number_text(series.quantile(0.75))
    assert (row["Smallest"], row["Largest"], row["Average"]) == ("1.50", "10", "4.75")


def test_the_text_table_lists_the_most_common_values_with_counts(loaded) -> None:
    dataset, _ = loaded
    text = profile_tables(dataset.profile).text.set_index("Column")
    assert list(text.index) == ["Site", "Active"]
    assert text.loc["Site", "Most common"] == "Site A (2), Site B (2), Site C (1)"
    assert text.loc["Site", "Different values"] == "3"
    assert text.loc["Active", "Most common"] == "true (3), false (2)"


def test_the_dates_table_gives_the_range(loaded) -> None:
    dataset, _ = loaded
    dates = profile_tables(dataset.profile).dates
    assert dates.to_dict("records") == [
        {"Column": "Recorded", "Earliest": "2024-04-03", "Latest": "2024-04-07"}
    ]


def test_every_cell_of_every_profile_table_is_text(loaded) -> None:
    dataset, _ = loaded
    tables = profile_tables(dataset.profile)
    for frame in (tables.overview, tables.numbers, tables.text, tables.dates):
        for column in frame.columns:
            assert all(isinstance(v, str) for v in frame[column])


def test_a_dataset_with_no_such_columns_has_empty_tables() -> None:
    settings = Settings(_env_file=None)
    with SessionDatabase(settings) as db:
        data = load_dataset(db, b"Name,Colour\nAnn,red\nBo,blue\n", "p.csv", settings=settings)
    tables = profile_tables(data.profile)
    assert tables.numbers.empty
    assert tables.dates.empty
    assert len(tables.text) == 2


def test_only_the_five_most_common_values_are_listed() -> None:
    rows = "\n".join(f"v{i}" for i in range(8) for _ in range(8 - i))
    settings = Settings(_env_file=None)
    with SessionDatabase(settings) as db:
        data = load_dataset(db, f"Item\n{rows}\n".encode(), "i.csv", settings=settings)
    common = profile_tables(data.profile).text.iloc[0]["Most common"]
    assert MAX_TOP_VALUES == 5
    assert common == "v0 (8), v1 (7), v2 (6), v3 (5), v4 (4)"


def test_flags_are_described_in_words() -> None:
    settings = Settings(_env_file=None)
    csv = b"Result,Day\n<0.5,01/02/2024\n3,03/02/2024\nND,04/02/2024\n"
    with SessionDatabase(settings) as db:
        data = load_dataset(db, csv, "lab.csv", settings=settings)
    overview = profile_tables(data.profile).overview.set_index("Column")
    assert overview.loc["Result", "Notes"] == FLAG_LABELS["censored_values"]
    assert overview.loc["Day", "Notes"] == FLAG_LABELS["ambiguous_date"]


def test_every_flag_and_type_has_a_label() -> None:
    from typing import get_args

    from analyst.core.models import ColumnFlag, ColumnType

    assert set(TYPE_LABELS) == set(get_args(ColumnType))
    assert set(FLAG_LABELS) == set(get_args(ColumnFlag))


def test_long_or_multiline_names_and_values_are_cut_and_flattened() -> None:
    settings = Settings(_env_file=None)
    name = "N" * 200
    value = "line one\nline two " + "x" * 200
    with SessionDatabase(settings) as db:
        data = load_dataset(db, f'"{name}"\n"{value}"\n'.encode(), "n.csv", settings=settings)
    tables = profile_tables(data.profile)
    assert len(tables.overview.iloc[0]["Column"]) <= 80
    common = tables.text.iloc[0]["Most common"]
    assert "\n" not in common
    assert len(common) <= 80 + len(" (1)")


# -- preview and tables on screen -----------------------------------------------------------------


def test_the_preview_is_the_first_rows_under_the_original_names(loaded) -> None:
    dataset, _ = loaded
    preview = dataset.preview
    assert list(preview.columns) == ["Site", "Rainfall (mm)", "Recorded", "Active"]
    assert preview["Site"].tolist() == ["Site A", "Site A", "Site B", "Site B", "Site C"]
    assert preview["Rainfall (mm)"].dropna().tolist() == [1.5, 3, 4.5, 10]


def test_the_preview_shows_dates_not_midnights(loaded) -> None:
    dataset, _ = loaded
    assert dataset.preview["Recorded"].tolist()[0] == date(2024, 4, 3)


def test_the_preview_stops_at_fifty_rows() -> None:
    settings = Settings(_env_file=None)
    csv = "n\n" + "\n".join(str(i) for i in range(120)) + "\n"
    with SessionDatabase(settings) as db:
        data = load_dataset(db, csv.encode(), "n.csv", settings=settings)
    assert PREVIEW_ROWS == 50
    assert len(data.preview) == 50
    assert data.preview["n"].tolist() == list(range(50))


def test_a_column_with_times_keeps_its_times() -> None:
    frame = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 10:30:00", "2024-01-02 00:00:00"])})
    assert display_frame(frame)["t"].tolist() == frame["t"].tolist()


def test_a_column_of_midnights_with_gaps_becomes_dates() -> None:
    frame = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", None, "2024-03-05"])})
    shown = display_frame(frame)
    assert shown["d"].tolist()[0] == date(2024, 1, 1)
    assert shown["d"].tolist()[2] == date(2024, 3, 5)
    assert pd.isna(shown["d"].tolist()[1])


def test_an_all_missing_date_column_is_left_as_it_is() -> None:
    frame = pd.DataFrame({"d": pd.to_datetime([None, None])})
    assert display_frame(frame)["d"].dtype == frame["d"].dtype


def test_a_timezone_aware_column_is_left_as_it_is() -> None:
    frame = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"]).tz_localize("UTC")})
    assert display_frame(frame)["d"].dtype == frame["d"].dtype


def test_other_columns_are_untouched_and_the_original_frame_is_not_changed() -> None:
    frame = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"]), "n": [1.5], "s": ["x"]})
    before = frame.copy()
    shown = display_frame(frame)
    assert shown["n"].tolist() == [1.5]
    assert shown["s"].tolist() == ["x"]
    pd.testing.assert_frame_equal(frame, before)


# -- describing a question ------------------------------------------------------------------------

NAMES = {"site": "Site", "rainfall": "Rainfall (mm)"}


@pytest.mark.parametrize(
    ("operator", "values", "text"),
    [
        ("=", ["Site A"], "Site is 'Site A'"),
        ("!=", ["Site A"], "Site is not 'Site A'"),
        (">", [5], "Site is more than 5"),
        (">=", [5.5], "Site is at least 5.5"),
        ("<", [5], "Site is less than 5"),
        ("<=", [5], "Site is at most 5"),
        ("in", ["A", "B"], "Site is one of 'A', 'B'"),
        ("not_in", ["A", "B"], "Site is not one of 'A', 'B'"),
        ("between", [1, 10], "Site is between 1 and 10"),
        ("is_null", [], "Site is missing"),
        ("is_not_null", [], "Site is present"),
        ("contains", ["nor"], "Site contains 'nor'"),
        ("month_in", [12, 1, 2], "Site has a month in 12, 1, 2"),
        ("year_in", [2023, 2024], "Site has a year in 2023, 2024"),
    ],
)
def test_each_filter_operator_is_put_into_words(operator, values, text: str) -> None:
    condition = FilterCondition(column="site", operator=operator, values=values)
    assert describe_filter(condition, NAMES) == text


def test_an_unknown_column_in_a_filter_is_shown_as_written() -> None:
    condition = FilterCondition(column="mystery", operator="=", values=[1])
    assert describe_filter(condition, NAMES) == "mystery is 1"


def test_a_value_from_the_data_is_flattened_and_cut() -> None:
    long = "x" * 200
    condition = FilterCondition(column="site", operator="=", values=[f"a\nb {long}"])
    text = describe_filter(condition, NAMES)
    assert "\n" not in text
    assert len(text) < 120


@pytest.mark.parametrize(
    ("start", "end", "text"),
    [
        ("2024-12-01", "2025-02-28", "Summer (2024-12-01 to 2025-02-28)"),
        ("2024-12-01", None, "Summer (from 2024-12-01)"),
        (None, "2025-02-28", "Summer (up to 2025-02-28)"),
        (None, None, "Summer"),
    ],
)
def test_a_period_is_given_with_its_dates(start, end, text: str) -> None:
    assert describe_time_range(TimeRange(start=start, end=end, description="Summer")) == text


def test_a_period_with_dates_and_no_description_is_just_the_dates() -> None:
    assert describe_time_range(TimeRange(start="2024-01-01", end="2024-01-31", description="")) == (
        "2024-01-01 to 2024-01-31"
    )


def intent(**changes) -> Intent:
    fields = {
        "operation": "aggregate",
        "measures": [Measure(column="rainfall", aggregation="avg")],
        "dimensions": ["site"],
        "filters": [],
        "time_range": None,
        "sort": [],
        "limit": None,
        "output": "table",
    }
    return Intent(**{**fields, **changes})


def test_the_filters_come_before_the_period() -> None:
    found = intent(
        filters=[FilterCondition(column="site", operator="=", values=["Site A"])],
        time_range=TimeRange(start="2024-01-01", end="2024-12-31", description="Last year"),
    )
    assert describe_intent(found, NAMES) == (
        "Site is 'Site A'",
        "Last year (2024-01-01 to 2024-12-31)",
    )


def test_a_question_with_neither_has_nothing_to_describe() -> None:
    assert describe_intent(intent(), NAMES) == ()
    assert describe_intent(None, NAMES) == ()


# -- the download ---------------------------------------------------------------------------------


def parse(data: bytes) -> list[list[str]]:
    import csv
    import io

    return list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))


def test_a_download_is_utf8_with_a_byte_order_mark_so_excel_reads_macrons() -> None:
    data = csv_bytes(pd.DataFrame({"Place": ["Whanganui-a-Tara", "Tāmaki Makaurau"]}))
    assert data.startswith(b"\xef\xbb\xbf")
    assert parse(data) == [["Place"], ["Whanganui-a-Tara"], ["Tāmaki Makaurau"]]


def test_numbers_dates_and_gaps_are_written_plainly() -> None:
    frame = pd.DataFrame(
        {
            "n": [1.5, -2.0, None],
            "d": pd.to_datetime(["2024-01-01", "2024-02-03", None]),
            "t": ["a", "b", None],
        }
    )
    assert parse(csv_bytes(frame)) == [
        ["n", "d", "t"],
        ["1.5", "2024-01-01", "a"],
        ["-2.0", "2024-02-03", "b"],
        ["", "", ""],
    ]


@pytest.mark.parametrize("start", ["=", "+", "-", "@", "\t", "\r"])
def test_a_text_cell_that_a_spreadsheet_would_run_as_a_formula_is_defused(start: str) -> None:
    cell = f'{start}HYPERLINK("http://evil.example")'
    rows = parse(csv_bytes(pd.DataFrame({"x": [cell]})))
    assert rows[1][0] == "'" + cell


def test_a_heading_that_would_run_as_a_formula_is_defused_too() -> None:
    assert parse(csv_bytes(pd.DataFrame({"=1+1": ["ok"]})))[0] == ["'=1+1"]


def test_negative_numbers_are_not_defused() -> None:
    rows = parse(csv_bytes(pd.DataFrame({"n": [-5, -0.5], "i": [-1, 2]})))
    assert rows[1:] == [["-5.0", "-1"], ["-0.5", "2"]]


def test_text_that_only_contains_a_formula_character_is_untouched() -> None:
    rows = parse(csv_bytes(pd.DataFrame({"x": ["a=b", "a-b", "x@y"]})))
    assert [r[0] for r in rows[1:]] == ["a=b", "a-b", "x@y"]


def test_two_result_columns_with_one_name_are_both_written_and_both_defused() -> None:
    frame = pd.DataFrame([["=a", "=b"]], columns=["x", "x"])
    assert parse(csv_bytes(frame)) == [["x", "x"], ["'=a", "'=b"]]


def test_a_column_of_plain_objects_is_defused_as_well_as_one_of_the_string_type() -> None:
    """DuckDB hands text over as either; a result must not depend on which."""
    cells = ["=a", "+b", 7, None]
    as_object = pd.DataFrame({"x": pd.Series(cells, dtype=object)})
    as_string = pd.DataFrame({"x": pd.Series(["=a", "+b", "7", None], dtype="string")})
    assert [r[0] for r in parse(csv_bytes(as_object))[1:]] == ["'=a", "'+b", "7", ""]
    assert [r[0] for r in parse(csv_bytes(as_string))[1:]] == ["'=a", "'+b", "7", ""]


def test_the_frame_given_is_not_changed() -> None:
    frame = pd.DataFrame({"x": ["=a"]})
    csv_bytes(frame)
    assert frame["x"].tolist() == ["=a"]


def test_an_empty_result_is_just_its_header() -> None:
    assert parse(csv_bytes(pd.DataFrame({"a": [], "b": []}))) == [["a", "b"]]


@pytest.mark.parametrize(
    ("count", "text"),
    [(0, "0 rows"), (1, "1 row"), (2, "2 rows"), (1_234, "1,234 rows")],
)
def test_a_count_takes_its_noun_in_the_singular_only_for_one(count: int, text: str) -> None:
    assert count_text(count, "row") == text
