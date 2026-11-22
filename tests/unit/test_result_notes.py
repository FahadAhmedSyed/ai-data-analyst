"""Statements about a result that code writes (T1.14, FR-6.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst.core.models import FilterCondition, Intent, QueryResult, TimeRange
from analyst.pipeline.result_notes import (
    MAX_MISSING_NOTES,
    describe_filter,
    empty_result_answer,
    format_number,
    result_caveats,
    template_answer,
)


def result_of(frame: pd.DataFrame, *, truncated: bool = False) -> QueryResult:
    return QueryResult(dataframe=frame, row_count=len(frame), truncated=truncated)


def intent(filters=(), time_range=None) -> Intent:
    return Intent(
        operation="aggregate",
        measures=[],
        dimensions=[],
        filters=list(filters),
        time_range=time_range,
        sort=[],
        limit=None,
        output="table",
    )


def cond(column, operator, *values) -> FilterCondition:
    return FilterCondition(column=column, operator=operator, values=list(values))


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0, "0"),
        (7, "7"),
        (1234, "1,234"),
        (1234567, "1,234,567"),
        (-1234, "-1,234"),
        (12.0, "12"),
        (1234.5, "1,234.50"),
        (1234.5678, "1,234.57"),
        (0.5, "0.50"),
        (0.0042, "0.0042"),
        (0.00001234, "1.234e-05"),
        (-0.5, "-0.50"),
        (np.int64(1234), "1,234"),
        (np.float64(2.5), "2.50"),
        (float("inf"), "inf"),
        (float("nan"), "nan"),
        (True, "True"),
    ],
)
def test_numbers_are_formatted_as_people_write_them(value, text):
    assert format_number(value) == text


# -- caveats ----------------------------------------------------------------------------------


def test_a_complete_result_needs_no_caveat():
    assert result_caveats(result_of(pd.DataFrame({"a": [1, 2]}))) == []


def test_a_truncated_result_says_so():
    frame = pd.DataFrame({"a": range(10_000)})
    assert result_caveats(result_of(frame, truncated=True)) == [
        "The result was cut at 10,000 rows, so there are more rows than are shown."
    ]


def test_missing_values_are_counted_per_column():
    frame = pd.DataFrame({"region": ["N", None, "S", None], "revenue": [1.0, 2.0, 3.0, 4.0]})
    assert result_caveats(result_of(frame)) == ["region is missing in 2 of 4 rows."]


def test_a_column_with_no_values_says_so():
    frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
    assert result_caveats(result_of(frame)) == ["a has no value in any row."]


def test_truncation_comes_before_missing_values_and_both_are_given():
    frame = pd.DataFrame({"a": [1.0, None]})
    notes = result_caveats(result_of(frame, truncated=True))
    assert notes[0].startswith("The result was cut at 2 rows")
    assert notes[1] == "a is missing in 1 of 2 rows."


def test_only_the_first_few_missing_columns_are_named():
    frame = pd.DataFrame({f"c{i}": [1.0, None] for i in range(MAX_MISSING_NOTES + 3)})
    notes = result_caveats(result_of(frame))
    assert len(notes) == MAX_MISSING_NOTES + 1
    assert notes[-1] == "3 more columns also have missing values."
    assert notes[0] == "c0 is missing in 1 of 2 rows."


def test_column_names_are_cleaned_before_they_are_shown():
    frame = pd.DataFrame({"bad\nname " + "x" * 200: [1.0, None]})
    (note,) = result_caveats(result_of(frame))
    assert "\n" not in note
    assert len(note) < 150


# -- empty results ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "text"),
    [
        (cond("region", "=", "North"), "region is 'North'"),
        (cond("region", "!=", "North"), "region is not 'North'"),
        (cond("revenue", ">", 100), "revenue is above 100"),
        (cond("revenue", ">=", 100.5), "revenue is at least 100.50"),
        (cond("revenue", "<", 5), "revenue is below 5"),
        (cond("revenue", "<=", 5), "revenue is at most 5"),
        (cond("region", "in", "North", "South"), "region is one of 'North', 'South'"),
        (cond("region", "not_in", "East"), "region is not one of 'East'"),
        (
            cond("order_date", "between", "2024-01-01", "2024-03-31"),
            "order_date is between '2024-01-01' and '2024-03-31'",
        ),
        (cond("revenue", "between", 10, 20.5), "revenue is between 10 and 20.50"),
        (cond("region", "is_null"), "region is missing"),
        (cond("region", "is_not_null"), "region is present"),
        (cond("product", "contains", "chair"), "product contains 'chair'"),
        (cond("order_date", "month_in", 1, 2), "order_date has month in 1, 2"),
        (cond("order_date", "year_in", 2024), "order_date has year in 2024"),
    ],
)
def test_each_filter_is_described_in_words(condition, text):
    assert describe_filter(condition) == text


def test_filter_text_is_cleaned_and_cut():
    text = describe_filter(cond("note", "=", "line one\nline two " + "z" * 200))
    assert "\n" not in text
    assert text.startswith("note is 'line one line two zzz")
    assert text.endswith("…'")


def test_an_empty_answer_lists_the_filters_and_the_date_range():
    answer = empty_result_answer(
        intent(
            [cond("region", "=", "Atlantis"), cond("revenue", ">", 1000)],
            TimeRange(start="2024-01-01", end=None, description="this year"),
        )
    )
    assert answer == (
        "The query ran but returned no rows. It filtered on: region is 'Atlantis'; "
        "revenue is above 1,000; date range 2024-01-01 to open."
    )


def test_an_empty_answer_without_filters_says_the_table_has_nothing():
    expected = (
        "The query ran but returned no rows. "
        "It had no filters, so the table has nothing that fits the question."
    )
    assert empty_result_answer(intent()) == expected
    assert empty_result_answer(None) == expected


# -- the template answer --------------------------------------------------------------------


def test_the_template_for_an_empty_result_names_the_filters():
    empty = result_of(pd.DataFrame({"a": pd.Series([], dtype=float)}))
    assert template_answer(empty, intent([cond("region", "=", "X")])).endswith("region is 'X'.")


def test_the_template_for_one_number_states_it():
    answer = template_answer(result_of(pd.DataFrame({"total_revenue": [1234.5678]})))
    assert answer == "Here is the result of the query: total_revenue is 1,234.57."


def test_the_template_for_one_integer_states_it():
    answer = template_answer(result_of(pd.DataFrame({"orders": np.array([42], dtype="int64")})))
    assert answer == "Here is the result of the query: orders is 42."


@pytest.mark.parametrize("value", ["text", True, None, np.nan])
def test_the_template_for_one_value_that_is_not_a_number_only_counts(value):
    answer = template_answer(result_of(pd.DataFrame({"x": [value]})))
    assert answer == "Here is the result of the query: 1 row and 1 column."


def test_the_template_for_a_table_counts_rows_and_columns():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert template_answer(result_of(frame)) == (
        "Here is the result of the query: 3 rows and 2 columns."
    )


def test_the_template_for_one_row_of_several_columns_is_singular():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    assert template_answer(result_of(frame)) == (
        "Here is the result of the query: 1 row and 2 columns."
    )
