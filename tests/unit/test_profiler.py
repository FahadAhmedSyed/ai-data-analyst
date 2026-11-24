"""Profiler and quality summary tests (T1.07).

Statistics are checked two ways: small fixtures with hand-computed answers, and
larger random fixtures whose expected values are computed with pandas inside the
test, never by calling the module under test.
"""

from __future__ import annotations

import io
import json
import math
import time
from datetime import date, datetime

import duckdb
import numpy as np
import pandas as pd
import pytest

from analyst.config.settings import Settings
from analyst.core.logging import configure_logging
from analyst.core.models import ColumnProfile, ColumnSchema, DatasetProfile
from analyst.database.session_db import SessionDatabase
from analyst.ingestion import profiler
from analyst.ingestion.csv_loader import load_csv
from analyst.ingestion.normalize import ColumnNote, TypedTable, normalize
from analyst.ingestion.profiler import (
    CHUNK_EXPRESSIONS,
    profile_database,
    quality_summary,
)

T = pd.Timestamp


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def schema(name: str, dtype: str, *flags: str, display: str | None = None) -> ColumnSchema:
    return ColumnSchema(
        sql_name=name, display_name=display or name, dtype=dtype, flags=tuple(flags)
    )


def profile_of(frame: pd.DataFrame, columns: list[ColumnSchema], source="data.csv", sheet=None):
    table = TypedTable(frame, tuple(columns), source, sheet)
    with SessionDatabase(settings()) as db:
        db.load(table, "data")
        return profile_database(db)


def column(profile: DatasetProfile, name: str) -> ColumnProfile:
    found = profile.column(name)
    assert found is not None
    return found


# ---------- hand-computed fixture ----------


@pytest.fixture(scope="module")
def small() -> DatasetProfile:
    frame = pd.DataFrame(
        {
            "n": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "x": pd.array([1.0, 2.0, 4.0, 8.0, None], dtype="Float64"),
            "city": pd.array(
                ["Wellington", "Auckland", "Wellington", None, "Auckland"], dtype="string"
            ),
            "ok": pd.array([True, True, False, None, True], dtype="boolean"),
            "day": pd.Series(
                [T("2024-03-01"), T("2024-03-15"), pd.NaT, T("2024-01-31"), T("2024-03-15")],
                dtype="datetime64[us]",
            ),
        }
    )
    columns = [
        schema("n", "integer"),
        schema("x", "decimal"),
        schema("city", "text"),
        schema("ok", "boolean"),
        schema("day", "date"),
    ]
    return profile_of(frame, columns, source="Rainfall.csv", sheet="Daily")


def test_table_level_facts(small):
    assert small.table_name == "data"
    assert small.source_name == "Rainfall.csv"
    assert small.sheet_name == "Daily"
    assert small.row_count == 5
    assert small.column_count == 5
    assert [c.sql_name for c in small.columns] == ["n", "x", "city", "ok", "day"]
    assert small.duplicate_rows == 0


def test_integer_statistics_by_hand(small):
    n = column(small, "n")
    # 1..5: mean 3, sample variance 10/4 = 2.5, quartiles 2, 3, 4 (linear interpolation).
    assert (n.null_count, n.distinct_count) == (0, 5)
    assert (n.min_value, n.max_value) == (1, 5)
    assert isinstance(n.min_value, int)
    assert n.mean == 3.0
    assert n.std == pytest.approx(math.sqrt(2.5))
    assert (n.q1, n.median, n.q3) == (2.0, 3.0, 4.0)
    assert n.top_values == ()


def test_decimal_statistics_by_hand_ignore_missing(small):
    x = column(small, "x")
    # 1, 2, 4, 8: mean 3.75; deviations -2.75, -1.75, 0.25, 4.25 give squares summing to
    # 7.5625 + 3.0625 + 0.0625 + 18.0625 = 28.75, variance 28.75 / 3. Quartiles: positions
    # 0.75, 1.5, 2.25 give 1.75, 3.0, 5.0.
    assert (x.null_count, x.distinct_count) == (1, 4)
    assert (x.min_value, x.max_value) == (1.0, 8.0)
    assert x.mean == 3.75
    assert x.std == pytest.approx(math.sqrt(28.75 / 3))
    assert (x.q1, x.median, x.q3) == (1.75, 3.0, 5.0)


def test_text_column_lists_frequent_values_and_has_no_numbers(small):
    city = column(small, "city")
    assert (city.null_count, city.distinct_count) == (1, 2)
    assert [(v.value, v.count) for v in city.top_values] == [("Auckland", 2), ("Wellington", 2)]
    assert (city.min_value, city.max_value, city.mean, city.std, city.q1) == (None,) * 5


def test_boolean_column_counts_true_and_false(small):
    ok = column(small, "ok")
    assert (ok.null_count, ok.distinct_count) == (1, 2)
    assert [(v.value, v.count) for v in ok.top_values] == [("true", 3), ("false", 1)]


def test_date_column_range_keeps_python_dates(small):
    day = column(small, "day")
    assert (day.null_count, day.distinct_count) == (1, 3)
    assert day.date_min == date(2024, 1, 31)
    assert day.date_max == date(2024, 3, 15)
    assert not isinstance(day.date_max, datetime)
    assert (day.min_value, day.max_value) == (None, None)


def test_primary_date_column_and_latest_date(small):
    assert small.primary_date_column == "day"
    assert small.latest_date == date(2024, 3, 15)


# ---------- agreement with pandas on a larger fixture ----------


@pytest.fixture(scope="module")
def large() -> tuple[pd.DataFrame, DatasetProfile]:
    rng = np.random.default_rng(42)
    rows = 5_000
    frame = pd.DataFrame(
        {
            "count_col": pd.array(rng.integers(-50, 500, rows), dtype="Int64"),
            "amount": pd.array(np.round(rng.normal(100, 30, rows), 3), dtype="Float64"),
            "site": pd.array(
                rng.choice(["Site A", "Site B", "Site C", "Site D"], rows), dtype="string"
            ),
            "code": pd.array([f"c{v}" for v in rng.integers(0, 400, rows)], dtype="string"),
            "when": pd.Series(
                pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, rows), unit="D"),
                dtype="datetime64[us]",
            ),
        }
    )
    frame.loc[rng.choice(rows, 250, replace=False), "amount"] = pd.NA
    frame.loc[rng.choice(rows, 120, replace=False), "site"] = pd.NA
    frame.loc[rng.choice(rows, 60, replace=False), "when"] = pd.NaT
    frame = pd.concat([frame, frame.iloc[:30]], ignore_index=True)  # 30 exact duplicates
    columns = [
        schema("count_col", "integer"),
        schema("amount", "decimal"),
        schema("site", "text"),
        schema("code", "text"),
        schema("when", "date"),
    ]
    return frame, profile_of(frame, columns)


@pytest.mark.parametrize("name", ["count_col", "amount", "site", "code", "when"])
def test_counts_match_pandas(large, name):
    frame, profile = large
    col = column(profile, name)
    assert col.null_count == int(frame[name].isna().sum())
    assert col.distinct_count == int(frame[name].nunique(dropna=True))
    assert profile.row_count == len(frame)


@pytest.mark.parametrize("name", ["count_col", "amount"])
def test_numeric_statistics_match_pandas(large, name):
    frame, profile = large
    col = column(profile, name)
    series = frame[name].dropna().astype("float64")
    assert col.min_value == series.min()
    assert col.max_value == series.max()
    assert col.mean == pytest.approx(series.mean(), rel=1e-12)
    assert col.std == pytest.approx(series.std(ddof=1), rel=1e-12)
    assert col.q1 == pytest.approx(series.quantile(0.25), rel=1e-12)
    assert col.median == pytest.approx(series.quantile(0.5), rel=1e-12)
    assert col.q3 == pytest.approx(series.quantile(0.75), rel=1e-12)


def test_date_range_matches_pandas(large):
    frame, profile = large
    when = column(profile, "when")
    assert when.date_min == frame["when"].min().date()
    assert when.date_max == frame["when"].max().date()
    assert profile.latest_date == frame["when"].max().date()


def test_frequent_values_match_pandas(large):
    frame, profile = large
    counts = frame["site"].dropna().value_counts()
    expected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    assert [(v.value, v.count) for v in column(profile, "site").top_values] == expected


def test_high_cardinality_text_with_repeats_still_lists_its_top_20(large):
    frame, profile = large
    code = column(profile, "code")
    assert code.distinct_count > 50  # above the schema-card limit, but values repeat
    counts = frame["code"].dropna().value_counts()
    expected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    assert len(code.top_values) == 20
    assert [(v.value, v.count) for v in code.top_values] == expected


def test_duplicate_rows_match_pandas(large):
    frame, profile = large
    assert profile.duplicate_rows == int(frame.duplicated().sum())
    assert profile.duplicate_rows >= 30


# ---------- edge cases ----------


def test_columns_with_one_row_or_no_values():
    frame = pd.DataFrame(
        {
            "one": pd.array([7], dtype="Int64"),
            "none": pd.array([None], dtype="string"),
            "d": pd.Series([pd.NaT], dtype="datetime64[us]"),
        }
    )
    profile = profile_of(
        frame, [schema("one", "integer"), schema("none", "text"), schema("d", "date")]
    )
    one = column(profile, "one")
    assert (one.min_value, one.max_value, one.mean) == (7, 7, 7.0)
    assert one.std is None  # a single value has no sample deviation
    assert (one.q1, one.median, one.q3) == (7.0, 7.0, 7.0)
    none = column(profile, "none")
    assert (none.null_count, none.distinct_count, none.top_values) == (1, 0, ())
    d = column(profile, "d")
    assert (d.null_count, d.date_min, d.date_max) == (1, None, None)
    assert profile.primary_date_column is None
    assert profile.latest_date is None


def test_all_missing_numeric_column_has_no_statistics():
    frame = pd.DataFrame({"x": pd.array([None, None], dtype="Float64")})
    x = column(profile_of(frame, [schema("x", "decimal")]), "x")
    assert x.null_count == 2
    assert (x.min_value, x.max_value, x.mean, x.std, x.q1, x.median, x.q3) == (None,) * 7


def test_an_empty_table_profiles_without_dividing_by_zero():
    frame = pd.DataFrame({"x": pd.array([], dtype="Int64"), "t": pd.array([], dtype="string")})
    profile = profile_of(frame, [schema("x", "integer"), schema("t", "text")])
    assert (profile.row_count, profile.duplicate_rows) == (0, 0)
    assert column(profile, "x").null_count == 0
    assert column(profile, "t").top_values == ()


def test_text_where_every_value_is_different_lists_no_values_beyond_fifty():
    values = [f"comment number {i}" for i in range(60)]
    frame = pd.DataFrame({"comment": pd.array(values, dtype="string")})
    comment = column(profile_of(frame, [schema("comment", "text")]), "comment")
    assert comment.distinct_count == 60
    assert comment.top_values == ()


def test_short_text_where_every_value_is_different_still_lists_values():
    # Fifty or fewer distinct values are listed even when none repeats: the language
    # model needs the exact spelling to write a filter.
    frame = pd.DataFrame({"name": pd.array(["Ana", "Ben", "Cy"], dtype="string")})
    name = column(profile_of(frame, [schema("name", "text")]), "name")
    assert [(v.value, v.count) for v in name.top_values] == [("Ana", 1), ("Ben", 1), ("Cy", 1)]


def test_ties_are_broken_by_value_so_results_are_stable():
    frame = pd.DataFrame({"t": pd.array(["b", "a", "c", "a", "b", "c", "d"], dtype="string")})
    top = column(profile_of(frame, [schema("t", "text")]), "t").top_values
    assert [(v.value, v.count) for v in top] == [("a", 2), ("b", 2), ("c", 2), ("d", 1)]


def test_missing_values_count_as_equal_when_finding_duplicate_rows():
    frame = pd.DataFrame(
        {
            "a": pd.array([1, 1, None, None, 2], dtype="Int64"),
            "b": pd.array(["x", "x", None, None, "y"], dtype="string"),
        }
    )
    profile = profile_of(frame, [schema("a", "integer"), schema("b", "text")])
    assert profile.duplicate_rows == 2  # rows 1 and 3 repeat rows 0 and 2


def test_primary_date_column_is_the_fullest_and_leftmost_on_ties():
    frame = pd.DataFrame(
        {
            "resolved": pd.Series([T("2024-05-01"), pd.NaT, pd.NaT], dtype="datetime64[us]"),
            "reported": pd.Series(
                [T("2024-01-01"), T("2024-02-01"), T("2024-03-01")], dtype="datetime64[us]"
            ),
            "closed": pd.Series(
                [T("2025-01-01"), T("2025-02-01"), T("2025-03-01")], dtype="datetime64[us]"
            ),
        }
    )
    columns = [schema("resolved", "date"), schema("reported", "date"), schema("closed", "date")]
    profile = profile_of(frame, columns)
    assert profile.primary_date_column == "reported"  # no missing values, left of "closed"
    assert profile.latest_date == date(2024, 3, 1)


def test_timestamp_columns_keep_their_time_in_the_latest_date():
    frame = pd.DataFrame(
        {"at": pd.Series([T("2024-03-01 08:30"), T("2024-03-02 17:45")], dtype="datetime64[us]")}
    )
    profile = profile_of(frame, [schema("at", "timestamp")])
    assert profile.latest_date == datetime(2024, 3, 2, 17, 45)
    assert column(profile, "at").date_min == datetime(2024, 3, 1, 8, 30)


def test_flags_and_display_names_are_carried_through():
    frame = pd.DataFrame({"r": pd.array(["<0.5", "1.2"], dtype="string")})
    profile = profile_of(frame, [schema("r", "text", "censored_values", display="Result")])
    r = column(profile, "r")
    assert r.display_name == "Result"
    assert r.flags == ("censored_values",)


# ---------- chunking and fallbacks ----------


def test_chunks_cover_every_column_once_in_order_and_stay_within_the_limit():
    columns = [
        schema(f"c{i}", dtype)
        for i, dtype in enumerate(["integer", "text", "date", "decimal", "boolean"] * 30)
    ]
    groups = profiler._chunks(columns)
    assert [c.sql_name for g in groups for c in g] == [c.sql_name for c in columns]
    for group in groups:
        assert sum(profiler._width(c) for c in group) <= CHUNK_EXPRESSIONS
    assert len(groups) > 1


def test_a_wide_table_matches_pandas_column_by_column():
    rng = np.random.default_rng(7)
    rows, width = 400, 120
    data, columns = {}, []
    for i in range(width):
        kind = ["integer", "decimal", "text"][i % 3]
        name = f"c{i}"
        if kind == "integer":
            data[name] = pd.array(rng.integers(0, 1000, rows), dtype="Int64")
        elif kind == "decimal":
            data[name] = pd.array(rng.normal(0, 5, rows), dtype="Float64")
        else:
            data[name] = pd.array(rng.choice(list("abcdefgh"), rows), dtype="string")
        columns.append(schema(name, kind))
    frame = pd.DataFrame(data)
    profile = profile_of(frame, columns)
    assert profile.column_count == width
    for i, col in enumerate(profile.columns):
        series = frame[f"c{i}"]
        assert col.distinct_count == int(series.nunique())
        if columns[i].dtype != "text":
            assert col.mean == pytest.approx(series.astype("float64").mean(), rel=1e-12)


def _fail_when(monkeypatch, fragment: str) -> None:
    original = SessionDatabase.query_rows

    def failing(self, sql, params=None):
        if fragment in sql:
            raise duckdb.OutOfMemoryException("Out of Memory Error: test")
        return original(self, sql, params)

    monkeypatch.setattr(SessionDatabase, "query_rows", failing)


def test_distinct_counts_fall_back_to_approximate_when_memory_runs_out(monkeypatch):
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    frame = pd.DataFrame({"t": pd.array([f"v{i % 40}" for i in range(400)], dtype="string")})
    _fail_when(monkeypatch, "count(DISTINCT")
    profile = profile_of(frame, [schema("t", "text")])
    assert column(profile, "t").distinct_count == pytest.approx(40, rel=0.05)
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert any(e["event"] == "profile_distinct_approximate" for e in events)
    assert next(e for e in events if e["event"] == "table_profiled")["exact_distinct"] is False


def test_duplicate_count_falls_back_to_a_row_hash_when_memory_runs_out(monkeypatch):
    frame = pd.DataFrame({"a": pd.array([1, 1, 2, 3, 3, 3], dtype="Int64")})
    _fail_when(monkeypatch, "SELECT DISTINCT *")
    assert profile_of(frame, [schema("a", "integer")]).duplicate_rows == 3


def test_other_memory_failures_are_not_swallowed(monkeypatch):
    frame = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    _fail_when(monkeypatch, "count(DISTINCT")
    _fail_when(monkeypatch, "approx_count_distinct")
    with pytest.raises(duckdb.OutOfMemoryException):
        profile_of(frame, [schema("a", "integer")])


def test_profiling_needs_a_loaded_session():
    with pytest.raises(RuntimeError, match="No dataset"):
        profile_database(SessionDatabase(settings()))


def test_profiling_logs_counts_but_no_names_or_values():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    frame = pd.DataFrame(
        {"secret_column": pd.array(["confidential-value", "other"], dtype="string")}
    )
    profile_of(frame, [schema("secret_column", "text")], source="payroll.csv")
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    done = next(e for e in events if e["event"] == "table_profiled")
    assert (done["rows"], done["columns"], done["duplicate_rows"], done["exact_distinct"]) == (
        2,
        1,
        0,
        True,
    )
    for private in ("confidential", "secret_column", "payroll"):
        assert private not in stream.getvalue()


# ---------- quality summary ----------


def kinds(issues):
    return [i.kind for i in issues]


def make_profile(rows=100, duplicates=0, **columns: ColumnProfile) -> DatasetProfile:
    return DatasetProfile(
        table_name="data",
        source_name="x.csv",
        row_count=rows,
        column_count=len(columns),
        columns=tuple(columns.values()),
        duplicate_rows=duplicates,
    )


def col(name, dtype="text", display=None, nulls=0, distinct=5, flags=()) -> ColumnProfile:
    return ColumnProfile(
        sql_name=name,
        display_name=display or name,
        dtype=dtype,
        flags=flags,
        null_count=nulls,
        distinct_count=distinct,
    )


def test_a_clean_dataset_has_no_issues():
    assert quality_summary(make_profile(a=col("a"), b=col("b", "integer"))) == ()


def test_duplicate_rows_are_reported_first_with_a_share():
    profile = make_profile(rows=200, duplicates=12, a=col("a", nulls=10, display="Rainfall (mm)"))
    issues = quality_summary(profile)
    assert kinds(issues) == ["duplicate_rows", "missing_values"]
    assert issues[0].message == "12 rows (6%) are exact copies of an earlier row."
    assert (issues[0].count, issues[0].column) == (12, None)
    assert issues[1].message == "Rainfall (mm) is missing in 10 rows (5%)."
    assert (issues[1].count, issues[1].column) == (10, "a")


def test_one_missing_value_is_one_row_not_one_rows():
    profile = make_profile(rows=200, duplicates=0, a=col("a", nulls=1, display="Rainfall (mm)"))
    (issue,) = quality_summary(profile)
    assert issue.message == "Rainfall (mm) is missing in 1 row (0.5%)."


@pytest.mark.parametrize(
    ("count", "rows", "expected"),
    [
        (0, 100, "0%"),
        (1, 1000, "0.1%"),
        (1, 5000, "under 0.1%"),
        (412, 10_962, "3.8%"),
        (20, 100, "20%"),
        (1, 3, "33.3%"),
        (9999, 10_000, "over 99.9%"),
        (100, 100, "100%"),
        (0, 0, "0%"),
    ],
)
def test_percent_wording(count, rows, expected):
    assert profiler._percent(count, rows) == expected


def test_all_missing_is_reported_once_not_also_as_missing_values():
    profile = make_profile(rows=50, a=col("a", nulls=50, distinct=0, flags=("all_missing",)))
    issues = quality_summary(profile)
    assert kinds(issues) == ["all_missing"]
    assert issues[0].message == "a has no values."
    assert issues[0].count == 50


def test_numeric_columns_with_unreadable_values_report_the_count_from_the_notes():
    profile = make_profile(
        qty=col("qty", "integer", display="Quantity", flags=("mixed_types",), nulls=3)
    )
    notes = {"qty": ColumnNote("whole numbers", 3)}
    issues = quality_summary(profile, notes)
    mixed = next(i for i in issues if i.kind == "mixed_types")
    assert (
        mixed.message
        == "3 values in Quantity could not be read as whole numbers and were treated as missing."
    )
    assert mixed.count == 3
    no_notes = next(i for i in quality_summary(profile) if i.kind == "mixed_types")
    assert (
        no_notes.message
        == "Some values in Quantity could not be read as whole numbers and were treated as missing."
    )


def test_dates_with_unreadable_values_use_date_wording():
    profile = make_profile(d=col("d", "date", display="Sampled", flags=("mixed_types",)))
    message = next(i for i in quality_summary(profile) if i.kind == "mixed_types").message
    assert (
        message == "Some values in Sampled could not be read as dates and were treated as missing."
    )


def test_text_that_mixes_numbers_and_text_is_explained():
    profile = make_profile(t=col("t", "text", display="Reading", flags=("mixed_types",)))
    (issue,) = quality_summary(profile)
    assert issue.message == "Reading mixes numbers and text, so it was kept as text."


def test_censored_and_ambiguous_columns_are_explained():
    profile = make_profile(
        r=col("r", display="Result", flags=("censored_values",)),
        d=col("d", "date", display="Order Date", flags=("ambiguous_date",)),
    )
    notes = {"d": ColumnNote("dd/mm/yyyy, order assumed from the region default")}
    issues = quality_summary(profile, notes)
    assert kinds(issues) == ["censored_values", "ambiguous_date"]
    assert issues[0].message == (
        "Result has censored values such as '<0.5' or 'ND'. "
        "They are kept as text and not converted to numbers."
    )
    assert issues[1].message == (
        "The day and month order in Order Date could not be settled from the values "
        "(dd/mm/yyyy, order assumed from the region default)."
    )


def test_issues_are_grouped_by_kind_then_column_order():
    profile = make_profile(
        rows=10,
        duplicates=1,
        a=col("a", nulls=1),
        b=col("b", flags=("constant",)),
        c=col("c", "date", flags=("ambiguous_date",)),
        d=col("d", nulls=2, flags=("censored_values",)),
    )
    issues = quality_summary(profile)
    assert [(i.kind, i.column) for i in issues] == [
        ("duplicate_rows", None),
        ("censored_values", "d"),
        ("ambiguous_date", "c"),
        ("missing_values", "a"),
        ("missing_values", "d"),
        ("constant", "b"),
    ]
    assert issues[-1].message == "b has the same value in every row."


def test_messages_never_contain_cell_values():
    csv = (
        b"Site,Result,Reported\r\n"
        b"TOPSECRET-SITE,<0.5,03/04/2024\r\n"
        b"TOPSECRET-SITE,1.2,04/05/2024\r\n"
        b"TOPSECRET-SITE,ND,\r\n"
        b"TOPSECRET-SITE,1.2,04/05/2024\r\n"
    )
    typed = normalize(load_csv(csv, "x.csv", settings=settings()), settings=settings())
    with SessionDatabase(settings()) as db:
        db.load(typed)
        issues = quality_summary(profile_database(db), typed.notes)
    assert {
        "censored_values",
        "ambiguous_date",
        "missing_values",
        "constant",
        "duplicate_rows",
    } <= set(kinds(issues))
    assert all("TOPSECRET" not in i.message and "1.2" not in i.message for i in issues)


# ---------- end to end ----------


def test_csv_to_profile_end_to_end():
    csv = (
        b"Order ID,Order Date,Region,Revenue ($)\r\n"
        b'1,13/01/2024,North,"$1,200.50"\r\n'
        b"2,02/02/2024,,$3.00\r\n"
        b"3,28/02/2024,South,$4.25\r\n"
        b"3,28/02/2024,South,$4.25\r\n"
    )
    typed = normalize(load_csv(csv, "Retail Sales.csv", settings=settings()), settings=settings())
    with SessionDatabase(settings()) as db:
        db.load(typed)
        profile = profile_database(db)
    assert (profile.table_name, profile.row_count, profile.duplicate_rows) == ("retail_sales", 4, 1)
    assert profile.primary_date_column == "order_date"
    assert profile.latest_date == date(2024, 2, 28)
    revenue = column(profile, "revenue")
    # 1200.5, 3, 4.25, 4.25: mean 303.0, max 1200.5.
    assert (revenue.mean, revenue.max_value) == (303.0, 1200.5)
    assert column(profile, "region").null_count == 1


@pytest.mark.slow
def test_a_million_row_profile_takes_under_five_seconds():
    """NFR-6: a 1 million-row profile in under 5 s."""
    rows = 1_000_000
    base = pd.Series(range(rows))
    frame = pd.DataFrame(
        {
            "id": base.astype("Int64"),
            "value": ((base * 7) % 10_007 / 3.0).astype("Float64"),
            "qty": ((base * 13) % 101).astype("Int64"),
            "site": ("Site " + (base % 6).astype(str)).astype("string"),
            "product": ("Product " + (base % 200).astype(str)).astype("string"),
            "customer": ("C" + (base % 5_000).astype(str)).astype("string"),
            "comment": ("note " + base.astype(str)).astype("string"),
            "day": (pd.Timestamp("2020-01-01") + pd.to_timedelta(base % 1_500, unit="D")).astype(
                "datetime64[us]"
            ),
        }
    )
    columns = [
        schema("id", "integer"),
        schema("value", "decimal"),
        schema("qty", "integer"),
        schema("site", "text"),
        schema("product", "text"),
        schema("customer", "text"),
        schema("comment", "text"),
        schema("day", "date"),
    ]
    table = TypedTable(frame, tuple(columns), "big.csv", None)
    with SessionDatabase(settings()) as db:
        db.load(table, "data")
        started = time.perf_counter()
        profile = profile_database(db)
        elapsed = time.perf_counter() - started
    assert profile.row_count == rows
    assert column(profile, "comment").top_values == ()
    assert column(profile, "site").distinct_count == 6
    assert elapsed < 5, f"profile took {elapsed:.1f}s"
