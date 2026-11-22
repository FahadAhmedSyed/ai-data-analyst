"""The grounding check (T1.14, ADR 0017). Expected values are written by hand."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from analyst.core.models import KeyNumber, ModelExplanation, QueryResult
from analyst.pipeline.grounding import (
    AllowedNumbers,
    GroundingIssue,
    GroundingReport,
    NumberToken,
    check_grounding,
    extract_numbers,
)


def result_of(frame: pd.DataFrame, *, truncated: bool = False) -> QueryResult:
    return QueryResult(dataframe=frame, row_count=len(frame), truncated=truncated)


def explanation(text="", key_numbers=(), caveats=()) -> ModelExplanation:
    return ModelExplanation(
        answer_markdown=text, key_numbers=list(key_numbers), caveats=list(caveats)
    )


def key(value, column, label="a number") -> KeyNumber:
    return KeyNumber(value=value, column=column, label=label)


# -- reading numbers out of text -------------------------------------------------------------


def targets(text: str) -> list[list[float]]:
    return [[t for t, _ in token.candidates] for token in extract_numbers(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234", [[1234.0]]),
        ("1,234", [[1234.0]]),
        ("1,234.5", [[1234.5]]),
        ("1,234,567.89", [[1234567.89]]),
        ("$1,234", [[1234.0]]),
        ("NZ$99.95", [[99.95]]),
        ("-3.2", [[3.2]]),
        ("−3.2", [[3.2]]),
        ("+7", [[7.0]]),
        (".5", [[0.5]]),
        ("0.5", [[0.5]]),
        ("1e3", [[1000.0]]),
        ("2.5E-3", [[0.0025]]),
        ("12%", [[12.0, 0.12]]),
        ("12 %", [[12.0, 0.12]]),
        ("12 percent", [[12.0, 0.12]]),
        ("12 per cent", [[12.0, 0.12]]),
        ("0.5%", [[0.5, 0.005]]),
        ("1.2 million", [[1_200_000.0]]),
        ("3 thousand", [[3000.0]]),
        ("2 Billion", [[2e9]]),
        ("3k", [[3.0, 3000.0]]),
        ("5m", [[5.0, 5_000_000.0]]),
        ("2bn", [[2.0, 2e9]]),
        ("Total is 5.", [[5.0]]),
        ("5 regions and 12 products", [[5.0], [12.0]]),
        ("from 2024-12-31", [[2024.0], [12.0], [31.0]]),
        ("ages 18-24", [[18.0], [24.0]]),
        ("between 1.5 and 2", [[1.5], [2.0]]),
        ("(1,234)", [[1234.0]]),
    ],
)
def test_numbers_are_read_in_every_common_format(text, expected):
    assert targets(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Q3 revenue",
        "Q10",
        "p95 latency",
        "H2",
        "v2 and v2.1",
        "the 1st, 2nd, 3rd and 4th place",
        "no digits at all",
        "three regions and twelve products",
        "item_7",
        "",
    ],
)
def test_digits_that_are_not_quantities_are_not_numbers(text):
    assert extract_numbers(text) == []


def test_list_numbering_and_heading_numbers_are_not_counted():
    text = "## 1. Summary\n1. First point\n2) Second point\n- 5 regions\n* 12 products\n3. Third"
    assert targets(text) == [[5.0], [12.0]]


def test_a_number_after_a_list_marker_is_still_read():
    assert targets("1. 1,500 sales") == [[1500.0]]


def test_the_written_text_of_a_token_includes_its_suffix():
    shown = [t.text for t in extract_numbers("12%, 1.2 million, 3k and 45")]
    assert shown == ["12%", "1.2 million", "3k", "45"]


@pytest.mark.parametrize(
    ("text", "half"),
    [("12", 0.5), ("12.3", 0.05), ("12.34", 0.005), ("1,234.5", 0.05), ("1e3", 0.0)],
)
def test_the_rounding_allowance_is_half_a_unit_of_the_last_digit(text, half):
    ((_, tolerance),) = extract_numbers(text)[0].candidates
    assert tolerance == pytest.approx(half)


def test_scaled_numbers_scale_their_rounding_allowance():
    ((target, tolerance),) = extract_numbers("1.2 million")[0].candidates
    assert (target, tolerance) == (pytest.approx(1.2e6), pytest.approx(50_000))


# -- matching ---------------------------------------------------------------------------------


def allowed(*values: float) -> AllowedNumbers:
    numbers = AllowedNumbers()
    numbers.add_values(values)
    return numbers


def accepts(numbers: AllowedNumbers, text: str) -> bool:
    return all(numbers.matches(token) for token in extract_numbers(text))


@pytest.mark.parametrize(
    ("value", "written", "ok"),
    [
        (1234.5678, "1234.5678", True),
        (1234.5678, "1,234.57", True),
        (1234.5678, "1,234.6", True),
        (1234.5678, "1,235", True),
        (1234.5678, "1,234", False),  # more than half a unit away; 1,235 is the rounding
        (1234.5678, "1,236", False),
        (1234.5678, "1,200", False),  # trailing zeros are not treated as rounding
        (1234.5678, "1,234.58", False),
        (1234.5678, "1.2 million", False),
        (1_234_567.0, "1.2 million", True),
        (1_234_567.0, "1.23 million", True),
        (1_234_567.0, "1.3 million", False),
        (12.4, "12", True),
        (11.6, "12", True),
        (12.6, "12", False),
        (0.55, "0.5", True),  # the boundary counts
        (0.55, "0.6", True),
        (3.2, "-3.2", True),
        (-3.2, "3.2", True),
        (-3.2, "-3.2", True),
        (0.12, "12%", True),
        (12.0, "12%", True),
        (0.1234, "12%", True),
        (0.1234, "12.3%", True),
        (0.1234, "13%", False),
        (1000.0, "1e3", True),
        (1000.0, "1e4", False),
        (5.0, "5m", True),  # metres, or
        (5_000_000.0, "5m", True),  # millions
        (3000.0, "3k", True),
        (0.0, "0", True),
        (7.0, "8", False),
    ],
)
def test_matching_after_rounding(value, written, ok):
    assert accepts(allowed(value), written) is ok


def test_nothing_matches_an_empty_list():
    assert not accepts(AllowedNumbers(), "5")


def test_any_listed_value_will_do():
    numbers = allowed(1.0, 50.0, 1000.0)
    assert accepts(numbers, "50")
    assert not accepts(numbers, "49")


def test_non_finite_values_are_never_allowed():
    numbers = allowed(float("nan"), float("inf"), float("-inf"))
    assert not accepts(numbers, "0")
    assert numbers.matches(NumberToken("x", ((float("inf"), 0.0),))) is False


def test_numbers_can_be_allowed_by_text():
    numbers = AllowedNumbers()
    numbers.add_text("top 5 regions in the last 12 months, 2024-12-31, 10%")
    for written in ("5", "12", "2024", "31", "10%", "10", "0.1"):
        assert accepts(numbers, written), written
    assert not accepts(numbers, "6")


def test_a_number_added_by_text_is_not_widened_by_the_texts_own_rounding():
    numbers = AllowedNumbers()
    numbers.add_text("about 12")
    assert not accepts(numbers, "12.4")


# -- checking an explanation against a result ------------------------------------------------

SALES = pd.DataFrame(
    {
        "region": ["North", "South", "East", "West"],
        "total_revenue": [1500.25, 1200.0, 980.5, 410.75],
        "orders": [30, 24, 20, 9],
        "share": [0.3612, 0.2889, 0.2360, 0.0989],
    }
)


def check(text="", frame=SALES, **kwargs):
    kwargs.setdefault("question", "Total revenue by region?")
    return check_grounding(
        explanation(text, kwargs.pop("key_numbers", ()), kwargs.pop("caveats", ())),
        result_of(frame),
        **kwargs,
    )


def test_numbers_copied_from_the_result_are_grounded():
    report = check("North made 1,500.25 from 30 orders, and South 1,200 from 24.")
    assert report.ok
    assert report.issues == ()


def test_rounded_numbers_are_grounded():
    assert check("North made about 1,500 and West 410.8.").ok


def test_a_percentage_of_a_fraction_column_is_grounded():
    assert check("North has 36% of the revenue and West 9.9%.").ok


def test_the_row_and_column_counts_are_grounded():
    assert check("There are 4 regions in 4 columns.").ok
    assert not check("There are 5 regions.").ok


def test_an_invented_number_is_caught():
    report = check("Revenue grew by 15% compared with last year.")
    assert not report.ok
    assert [(i.kind, i.text) for i in report.issues] == [("number", "15%")]


def test_a_derived_number_is_caught():
    """The gap between North and South, and their ratio, were not produced by a query."""
    assert not check("North made 300.25 more than South.").ok
    assert not check("North made 1.25 times as much as South.").ok
    assert not check("The four regions made 4,091.5 altogether.").ok


def test_several_ungrounded_numbers_are_each_reported_once():
    report = check("Up 77 and 88, then 77 again.")
    assert [i.text for i in report.issues] == ["77", "88"]


def test_numbers_in_the_question_are_grounded():
    assert not check("Showing the top 3 regions.").ok
    assert check("Showing the top 3 regions.", question="Top 3 regions by revenue").ok


def test_numbers_in_the_assumptions_are_grounded():
    text = "Using the 12 months to 31 December 2024."
    assert not check(text).ok
    assert check(text, assumptions=["Last 12 months means 2024-01-01 to 2024-12-31."]).ok


def test_extra_numbers_the_pipeline_supplied_are_grounded():
    text = "Only the first 10,000 rows are counted."
    assert not check(text).ok
    assert check(text, extra_numbers=[10_000]).ok


def test_numbers_in_caveats_and_key_number_labels_are_checked_too():
    assert not check("Fine.", caveats=["Based on 99 orders."]).ok
    assert check("Fine.", caveats=["Based on 30 orders."]).ok
    assert not check("Fine.", key_numbers=[key(30, "orders", "Orders in 2019")]).ok


def test_digits_in_text_values_and_column_names_are_grounded():
    frame = pd.DataFrame(
        {"age_group": ["18-24", "25-34", "Tier 2"], "revenue_2024": [10.0, 20.0, 30.0]}
    )
    assert check("The 18-24 group and Tier 2 customers, in 2024.", frame=frame).ok
    assert not check("The 35-44 group.", frame=frame).ok


def test_the_year_month_and_day_of_result_dates_are_grounded():
    frame = pd.DataFrame(
        {"month": pd.to_datetime(["2024-03-01", "2024-04-01"]), "revenue": [5.5, 6.5]}
    )
    assert check("From 1 March 2024 to 1 April 2024, 5.5 then 6.5.", frame=frame).ok
    assert not check("In 2023.", frame=frame).ok
    assert not check("On the 15th, 15 orders.", frame=frame).ok


def test_object_columns_with_dates_decimals_and_text_are_understood():
    frame = pd.DataFrame(
        {
            "when": pd.Series([date(2024, 5, 6), datetime(2025, 7, 8, 9, 10)], dtype=object),
            "price": pd.Series([Decimal("12.50"), Decimal("3.25")], dtype=object),
            "label": pd.Series(["Batch 42", None], dtype=object),
            "other": pd.Series([[1, 2], {"a": 3}], dtype=object),
        }
    )
    assert check("2024, 5, 6, 2025, 7, 8, 12.5, 3.25, 42.", frame=frame).ok
    assert not check("43", frame=frame).ok


def test_booleans_inside_an_object_column_are_not_numbers():
    frame = pd.DataFrame(
        {"mixed": pd.Series([True, "x", 40.0], dtype=object), "other": ["a", "b", "c"]}
    )
    assert check("40", frame=frame).ok
    assert not check("1", frame=frame).ok  # True is not the number 1


def test_missing_infinite_and_boolean_values_are_ignored():
    frame = pd.DataFrame(
        {
            "x": [1.5, np.nan, np.inf],
            "flag": [True, False, True],
            "n": pd.array([7, None, 9], dtype="Int64"),
        }
    )
    assert check("1.5, 7 and 9", frame=frame).ok
    assert not check("4", frame=frame).ok
    assert check("2", frame=frame).ok  # within rounding of 1.5; a boolean adds nothing


def test_a_large_result_is_matched_quickly_and_exactly():
    frame = pd.DataFrame({"v": np.arange(100_000, dtype=float) * 3})
    assert check("299,997 is there.", frame=frame).ok
    assert not check("299,998 is not.", frame=frame).ok


def test_an_empty_result_allows_only_zero_counts():
    frame = pd.DataFrame({"region": pd.Series([], dtype=object)})
    assert check("The query returned 0 rows in 1 column.", frame=frame).ok
    assert not check("It returned 5 rows.", frame=frame).ok


def test_text_with_no_numbers_is_always_grounded():
    assert check("North leads, followed by South.").ok
    assert check("").ok


# -- key numbers ------------------------------------------------------------------------------


def test_key_numbers_must_be_values_of_their_column():
    report = check(
        "ok",
        key_numbers=[
            key(1500.25, "total_revenue"),
            key(1500.3, "total_revenue"),  # rounded to its own precision
            key(24, "orders"),
            key(0.2889, "share"),
            key(28.89, "share"),  # shown as a percentage
        ],
    )
    assert report.ok


def test_key_numbers_with_the_wrong_value_or_column_are_caught():
    report = check(
        "ok",
        key_numbers=[
            key(1500.25, "orders", "Wrong column"),
            key(77, "orders", "Not a value"),
            key(30, "no_such_column", "No column"),
            key(1, "region", "Text column"),
        ],
    )
    assert [i.text for i in report.issues] == [
        "Wrong column",
        "Not a value",
        "No column",
        "Text column",
    ]
    assert all(i.kind == "key_number" for i in report.issues)
    assert "orders" in report.issues[0].reason
    assert "no numeric column 'no_such_column'" in report.issues[2].reason


def test_the_column_name_may_differ_in_case():
    assert check("ok", key_numbers=[key(30, "ORDERS")]).ok


def test_a_key_number_for_a_boolean_or_empty_column_is_caught():
    frame = pd.DataFrame({"flag": [True, False], "gone": [np.nan, np.nan]})
    assert not check("ok", frame=frame, key_numbers=[key(1, "flag")]).ok
    assert not check("ok", frame=frame, key_numbers=[key(1, "gone")]).ok


def test_a_key_number_that_is_not_in_the_text_is_still_checked_against_the_result():
    assert not check("Nothing numeric here.", key_numbers=[key(31, "orders")]).ok


def test_key_numbers_do_not_make_other_numbers_acceptable():
    """The text is checked against the result, not against the model's own key numbers."""
    assert not check("It was 31.", key_numbers=[key(31, "orders")]).ok


# -- the report ----------------------------------------------------------------------------------


def test_the_report_gives_feedback_without_quoting_data():
    report = check("Up 77 and 88.", key_numbers=[key(5, "orders", "Bad key")])
    feedback = report.feedback()
    assert "77, 88" in feedback
    assert "Bad key" in feedback
    assert "North" not in feedback


def test_feedback_for_each_kind_alone():
    assert GroundingReport(()).feedback() == ""
    only_numbers = GroundingReport((GroundingIssue("number", "5", "x"),))
    assert only_numbers.feedback() == "These numbers are not in the result: 5."
    only_keys = GroundingReport((GroundingIssue("key_number", "L", "x"),))
    assert only_keys.feedback() == "These key numbers do not match their column: L."


def test_ok_reflects_the_issues():
    assert GroundingReport(()).ok
    assert not GroundingReport((GroundingIssue("number", "5", "x"),)).ok
