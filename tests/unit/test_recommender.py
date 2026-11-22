"""Tests for the chart recommender (T1.15): which chart a result gets, and which it refuses."""

from __future__ import annotations

import pandas as pd
import pytest

from analyst.core.models import ChartSpec, ChartType, QueryResult
from analyst.visualization import fit_problem, recommend_chart
from analyst.visualization.recommender import (
    MAX_BAR_CATEGORIES,
    column_kind,
    column_kinds,
)

MONTHS = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])


def result(**columns: object) -> QueryResult:
    frame = pd.DataFrame(columns)
    return QueryResult(dataframe=frame, row_count=len(frame))


def spec(
    kind: ChartType, x: str | None = None, y: list[str] | None = None, color: str | None = None
) -> ChartSpec:
    return ChartSpec(type=kind, x=x, y=y or [], color=color)


NONE = spec("none")


# --- column kinds ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (pd.Series([1, 2, 3]), "numeric"),
        (pd.Series([1.5, None, 3.0]), "numeric"),
        (pd.Series(["a", "b"]), "text"),
        (pd.Series(["a", None]), "text"),
        (pd.Series([True, False]), "text"),
        (pd.Series(["a", "b"], dtype="category"), "text"),
        (pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02"])), "date"),
        (pd.Series([None, None], dtype="float64"), "other"),
        (pd.Series([], dtype="float64"), "other"),
        (pd.Series([[1], [2]]), "other"),
    ],
)
def test_column_kind(values: pd.Series, expected: str) -> None:
    assert column_kind(values) == expected


def test_column_kinds_is_keyed_by_column_name() -> None:
    frame = pd.DataFrame({"when": MONTHS, "region": list("abcd"), "n": [1, 2, 3, 4]})
    assert column_kinds(frame) == {"when": "date", "region": "text", "n": "numeric"}


# --- rules: what a result of each shape gets ------------------------------------------------


def test_one_date_and_one_number_is_a_line() -> None:
    chart = recommend_chart(result(month=MONTHS, revenue=[1, 2, 3, 4]))
    assert chart == spec("line", "month", ["revenue"])


def test_one_date_and_several_numbers_is_a_line_with_several_value_columns() -> None:
    chart = recommend_chart(result(month=MONTHS, revenue=[1, 2, 3, 4], orders=[4, 3, 2, 1]))
    assert chart == spec("line", "month", ["revenue", "orders"])


def test_date_text_number_is_one_line_per_text_value() -> None:
    frame = result(
        month=list(MONTHS[:2]) * 2,
        region=["north", "north", "south", "south"],
        revenue=[1, 2, 3, 4],
    )
    assert recommend_chart(frame) == spec("line", "month", ["revenue"], "region")


def test_date_text_and_two_numbers_is_not_charted() -> None:
    frame = result(
        month=list(MONTHS[:2]) * 2,
        region=["a", "a", "b", "b"],
        revenue=[1, 2, 3, 4],
        orders=[1, 2, 3, 4],
    )
    assert recommend_chart(frame) == NONE


def test_text_and_a_number_is_a_bar_chart() -> None:
    chart = recommend_chart(result(region=["n", "s", "e"], revenue=[3, 1, 2]))
    assert chart == spec("bar", "region", ["revenue"])


def test_text_and_several_numbers_is_a_bar_chart_with_several_value_columns() -> None:
    chart = recommend_chart(result(region=["n", "s", "e"], a=[3, 1, 2], b=[1, 2, 3]))
    assert chart == spec("bar", "region", ["a", "b"])


def test_two_numbers_is_a_scatter_plot() -> None:
    chart = recommend_chart(result(price=[1.0, 2.0, 3.0, 4.0], units=[9, 7, 4, 1]))
    assert chart == spec("scatter", "price", ["units"])


def test_two_numbers_with_two_rows_is_not_a_scatter_plot() -> None:
    assert recommend_chart(result(a=[1, 2], b=[3, 4])) == NONE


def test_one_number_column_of_ten_rows_is_a_histogram() -> None:
    chart = recommend_chart(result(age=list(range(10))))
    assert chart == spec("histogram", "age", [])


def test_one_number_column_of_nine_rows_is_not_charted() -> None:
    assert recommend_chart(result(age=list(range(9)))) == NONE


@pytest.mark.parametrize(
    "frame",
    [
        result(total=[42]),
        result(region=["n"], revenue=[1]),
        result(region=["n", "s"]),
        result(a=[1, 2, 3], b=[1, 2, 3], c=[1, 2, 3]),
        result(a=["x", "y"], b=["u", "v"], n=[1, 2]),
        result(month=MONTHS),
        result(blank=[None, None, None], n=[1, 2, 3]),
        result(items=[[1], [2], [3]], n=[1, 2, 3]),
        pd.DataFrame().pipe(lambda f: QueryResult(dataframe=f, row_count=0)),
    ],
    ids=[
        "single value",
        "single row",
        "only text",
        "three numbers",
        "two text columns",
        "only a date",
        "an all-empty column",
        "a column of lists",
        "no rows",
    ],
)
def test_shapes_without_a_good_chart_get_none(frame: QueryResult) -> None:
    assert recommend_chart(frame) == NONE


def test_thirty_bars_are_drawn_and_thirty_one_are_not() -> None:
    assert MAX_BAR_CATEGORIES == 30
    ok = result(k=[f"c{i}" for i in range(30)], v=range(30))
    too_many = result(k=[f"c{i}" for i in range(31)], v=range(31))
    assert recommend_chart(ok).type == "bar"
    assert recommend_chart(too_many) == NONE


def test_a_date_with_two_text_columns_is_not_charted() -> None:
    frame = result(
        month=list(MONTHS[:2]) * 2,
        region=["a", "a", "b", "b"],
        channel=["web", "web", "shop", "shop"],
        revenue=[1, 2, 3, 4],
    )
    assert recommend_chart(frame) == NONE


def test_eight_lines_are_drawn_and_nine_are_not() -> None:
    def grouped(groups: int) -> QueryResult:
        names = [f"g{g}" for g in range(groups) for _ in range(2)]
        return result(month=list(MONTHS[:2]) * groups, g=names, v=range(2 * groups))

    assert recommend_chart(grouped(8)).color == "g"
    assert recommend_chart(grouped(9)) == NONE


def test_a_line_per_group_needs_a_group_with_more_than_one_point() -> None:
    frame = result(month=MONTHS, region=list("abcd"), revenue=[1, 2, 3, 4])
    assert fit_problem(frame, spec("line", "month", ["revenue"], "region")) is not None
    assert recommend_chart(frame) == NONE
    mixed = result(month=[*MONTHS[:2], MONTHS[0]], region=["a", "a", "b"], revenue=[1, 2, 3])
    assert fit_problem(mixed, spec("line", "month", ["revenue"], "region")) is None


def test_rows_that_share_an_x_position_are_not_drawn_as_a_line() -> None:
    frame = result(month=[MONTHS[0], MONTHS[0], MONTHS[1]], revenue=[1, 2, 3])
    assert recommend_chart(frame) == NONE


def test_rows_that_share_a_category_are_not_drawn_as_bars() -> None:
    assert recommend_chart(result(region=["n", "n", "s"], revenue=[1, 2, 3])) == NONE


# --- the model's suggestion ----------------------------------------------------------------


def test_a_fitting_suggestion_is_used_in_place_of_the_rules() -> None:
    frame = result(region=["n", "s", "e"], a=[3, 1, 2], b=[1, 2, 3])
    suggestion = spec("bar", "region", ["b"])
    assert recommend_chart(frame, suggestion) == suggestion


def test_a_suggestion_naming_a_missing_column_falls_back_to_the_rules() -> None:
    frame = result(region=["n", "s", "e"], revenue=[3, 1, 2])
    assert recommend_chart(frame, spec("bar", "country", ["revenue"])) == spec(
        "bar", "region", ["revenue"]
    )


def test_a_suggestion_of_the_wrong_type_for_the_columns_falls_back_to_the_rules() -> None:
    frame = result(region=["n", "s", "e"], revenue=[3, 1, 2])
    assert recommend_chart(frame, spec("line", "region", ["revenue"])).type == "bar"


def test_a_suggestion_of_none_is_respected() -> None:
    frame = result(region=["n", "s", "e"], revenue=[3, 1, 2])
    assert recommend_chart(frame, NONE) == NONE


def test_no_suggestion_means_the_rules_decide() -> None:
    frame = result(region=["n", "s", "e"], revenue=[3, 1, 2])
    assert recommend_chart(frame, None).type == "bar"


def test_a_suggestion_may_choose_a_chart_the_rules_would_not() -> None:
    frame = result(month=MONTHS, region=list("abcd"), revenue=[1, 2, 3, 4])
    suggestion = spec("bar", "region", ["revenue"])
    assert recommend_chart(frame, suggestion) == suggestion
    assert recommend_chart(frame) == NONE


# --- why a chart is refused ----------------------------------------------------------------

BARS = result(region=["n", "s", "e"], revenue=[3, 1, 2], note=["a", "b", "c"])


@pytest.mark.parametrize(
    ("chart", "frame", "reason"),
    [
        (spec("bar", "region", ["revenue"]), result(region=["n"], revenue=[1]), "too few rows"),
        (
            spec("scatter", "a", ["b"]),
            result(a=[1, 2], b=[1, 2]),
            "too few rows",
        ),
        (spec("bar", None, ["revenue"]), BARS, "the x column is missing"),
        (spec("bar", "country", ["revenue"]), BARS, "the x column is missing"),
        (spec("bar", "region", ["nope", "gone"]), BARS, "no such column: nope, gone"),
        (spec("bar", "region", ["revenue"], "nope"), BARS, "no such column: nope"),
        (spec("bar", "region", ["region"]), BARS, "a column is used twice"),
        (spec("bar", "region", []), BARS, "needs between 1 and 5 value columns"),
        (
            spec("bar", "region", ["revenue"] * 1 + ["a", "b", "c", "d", "e"]),
            result(
                region=["n", "s"], revenue=[1, 2], a=[1, 2], b=[1, 2], c=[1, 2], d=[1, 2], e=[1, 2]
            ),
            "needs between 1 and 5 value columns",
        ),
        (spec("bar", "region", ["note"]), BARS, "the value columns must be numeric"),
        (spec("line", "region", ["revenue"]), BARS, "a line chart needs a date or number"),
        (spec("bar", "revenue", ["revenue"]), BARS, "a column is used twice"),
        (
            spec("bar", "revenue", ["note"]),
            result(revenue=[1, 2, 3], note=[1, 2, 3]),
            "a bar chart needs categories",
        ),
        (
            spec("scatter", "region", ["revenue"]),
            BARS,
            "a scatter plot needs a number on the x axis",
        ),
        (
            spec("scatter", "a", ["b", "c"]),
            result(a=[1, 2, 3], b=[1, 2, 3], c=[1, 2, 3]),
            "a scatter plot draws one value column",
        ),
        (
            spec("bar", "region", ["revenue"], "revenue_2"),
            result(region=["n", "s"], revenue=[1, 2], revenue_2=[3, 4]),
            "the colour column must hold categories",
        ),
        (
            spec("bar", "region", ["a", "b"], "g"),
            result(region=["n", "s"], a=[1, 2], b=[1, 2], g=["x", "y"]),
            "exactly one value column",
        ),
        (spec("histogram", "region", []), result(region=list("abcdefghij")), "needs a numeric"),
        (spec("histogram", "n", ["m"]), result(n=range(10), m=range(10)), "draws one column"),
        (spec("histogram", "n", []), result(n=range(9)), "too few rows for a histogram"),
    ],
)
def test_fit_problem_names_the_reason(chart: ChartSpec, frame: QueryResult, reason: str) -> None:
    problem = fit_problem(frame, chart)
    assert problem is not None
    assert reason in problem


def test_fit_problem_is_none_for_a_chart_that_fits_and_for_none() -> None:
    assert fit_problem(BARS, spec("bar", "region", ["revenue"])) is None
    assert fit_problem(BARS, NONE) is None
    assert fit_problem(result(n=[1]), NONE) is None


def test_line_charts_accept_a_numeric_x_axis() -> None:
    frame = result(day=[1, 2, 3], visits=[5, 6, 7])
    assert fit_problem(frame, spec("line", "day", ["visits"])) is None


def test_bar_charts_accept_a_date_x_axis() -> None:
    frame = result(month=MONTHS, visits=[5, 6, 7, 8])
    assert fit_problem(frame, spec("bar", "month", ["visits"])) is None


def test_scatter_plots_are_limited_to_three_groups() -> None:
    def grouped(groups: int) -> QueryResult:
        n = groups * 2
        return result(a=range(n), b=range(n), g=[f"g{i}" for i in range(groups) for _ in range(2)])

    chart = spec("scatter", "a", ["b"], "g")
    assert fit_problem(grouped(3), chart) is None
    assert "more than 3 groups" in (fit_problem(grouped(4), chart) or "")


def test_fit_problem_never_repeats_values_from_the_data() -> None:
    private = "SSN-123-45-6789"
    frame = result(region=[private, "b", "c"], revenue=[1, 2, 3])
    for chart in (
        spec("line", "region", ["revenue"]),
        spec("scatter", "region", ["revenue"]),
        spec("bar", "region", ["region"]),
        spec("histogram", "region", []),
    ):
        assert private not in (fit_problem(frame, chart) or "")


def test_recommend_chart_always_returns_a_spec_that_fits() -> None:
    frames = [
        result(month=MONTHS, revenue=[1, 2, 3, 4]),
        result(region=["n", "s"], revenue=[1, 2]),
        result(a=[1, 2, 3], b=[3, 2, 1]),
        result(a=range(12)),
        result(a=["x", "y"]),
        result(a=[1]),
    ]
    for frame in frames:
        assert fit_problem(frame, recommend_chart(frame)) is None
