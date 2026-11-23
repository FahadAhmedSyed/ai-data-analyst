"""The scoring rules (eval/README.md section 4), one rule at a time."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import eval.scoring as scoring
from eval.scoring import (
    Expected,
    assumption_found,
    cells_match,
    compare_results,
    is_read_only,
    limit_free_sql,
    linking_scores,
    normalise_cell,
    percentile,
    ratio,
    referenced_columns,
    sort_columns,
    tied_at_cutoff,
)


def frame(*rows, columns=None) -> pd.DataFrame:
    width = len(rows[0]) if rows else len(columns)
    return pd.DataFrame(list(rows), columns=columns or [f"c{i}" for i in range(width)])


def expect(*rows, columns=None, **kw) -> Expected:
    return Expected(sql="gold", frame=frame(*rows, columns=columns), **kw)


def same(candidate, expected, mode="unordered") -> bool:
    return compare_results(candidate, expected, mode).matched


# -- cells -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "normal"),
    [
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
        (pd.NaT, None),
        (np.float64("nan"), None),
        (True, True),
        (np.bool_(False), False),
        (3, 3.0),
        (np.int64(4), 4.0),
        (np.float32(0.5), 0.5),
        (Decimal("2.50"), 2.5),
        ("  North ", "North"),
        (date(2024, 3, 1), date(2024, 3, 1)),
        (pd.Timestamp("2024-03-01"), date(2024, 3, 1)),
        (datetime(2024, 3, 1), date(2024, 3, 1)),
        (pd.Timestamp("2024-03-01 09:30"), datetime(2024, 3, 1, 9, 30)),
        (pd.Timestamp("2024-03-01 00:00", tz="UTC"), date(2024, 3, 1)),
        (np.datetime64("2024-03-01"), date(2024, 3, 1)),
        (b"x", "b'x'"),
    ],
)
def test_normalise_cell(value, normal) -> None:
    result = normalise_cell(value)
    assert result == normal
    assert type(result) is type(normal)


def test_an_integer_is_not_confused_with_a_bool() -> None:
    assert normalise_cell(1) == 1.0
    assert normalise_cell(True) is True
    assert not cells_match(normalise_cell(True), normalise_cell(1))


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1.0, 1.0 + 5e-7, True),
        (1.0, 1.0 + 5e-6, False),
        (1_000_000.0, 1_000_000.4, True),
        (0.0, 1e-12, True),
        (0.0, 1e-6, False),
        (None, None, True),
        (None, 0.0, False),
        ("a", "a", True),
        ("a", "A", False),
        (date(2024, 1, 1), date(2024, 1, 1), True),
        (date(2024, 1, 1), datetime(2024, 1, 1, 5), False),
        ("5", 5.0, False),
    ],
)
def test_cells_match(a, b, expected) -> None:
    assert cells_match(a, b) is expected


# -- scalar, names and columns ---------------------------------------------------------------------


def test_a_scalar_matches_whatever_the_column_is_called() -> None:
    assert same(frame((10.0,), columns=["total"]), expect((10.0,), columns=["revenue"]), "scalar")


def test_a_scalar_within_tolerance_matches_and_outside_does_not() -> None:
    gold = expect((2714970.85,))
    assert same(frame((2714970.85 + 1.0,)), gold, "scalar")
    assert not same(frame((2714970.85 + 100.0,)), gold, "scalar")


def test_a_scalar_that_differs_says_what_was_expected_and_what_came() -> None:
    verdict = compare_results(frame((3.0, "x")), expect((2.0,)), "scalar")
    assert (verdict.matched, verdict.reason) == (False, "expected 2.0, got 3.0, 'x'")


def test_an_integer_and_a_float_of_the_same_value_match() -> None:
    assert same(frame((5,)), expect((5.0,)), "scalar")


def test_extra_candidate_columns_are_allowed_if_the_gold_columns_are_there() -> None:
    gold = expect(("North", 5.0), ("South", 7.0))
    candidate = frame((1, "North", 5.0, "x"), (2, "South", 7.0, "y"))
    assert same(candidate, gold)


def test_columns_may_come_in_any_order() -> None:
    gold = expect(("North", 5.0), ("South", 7.0))
    assert same(frame((5.0, "North"), (7.0, "South")), gold)


def test_a_missing_column_fails_and_says_which() -> None:
    gold = expect(("North", 5.0), columns=["region", "revenue"])
    verdict = compare_results(frame(("North",)), gold, "unordered")
    assert not verdict.matched
    assert verdict.reason == "1 column(s), expected at least 2"
    verdict = compare_results(frame(("North", 6.0)), gold, "unordered")
    assert not verdict.matched
    assert "'revenue'" in verdict.reason


def test_the_wrong_number_of_rows_fails_and_says_so() -> None:
    gold = expect(("a",), ("b",))
    verdict = compare_results(frame(("a",)), gold, "unordered")
    assert (verdict.matched, verdict.reason) == (False, "1 rows, expected 2")


def test_two_gold_columns_need_two_different_candidate_columns() -> None:
    gold = expect((5.0, 5.0), (7.0, 7.0))
    assert not same(frame((5.0,), (7.0,)), gold)
    assert same(frame((5.0, 5.0), (7.0, 7.0)), gold)


def test_columns_that_have_the_right_values_but_are_paired_wrongly_fail() -> None:
    gold = expect((1, "x"), (2, "y"))
    verdict = compare_results(frame((1, "y"), (2, "x")), gold, "unordered")
    assert not verdict.matched
    assert verdict.reason == "the values are not in the same rows"


def test_a_column_of_mixed_types_can_be_compared_in_any_order() -> None:
    mixed = [None, True, 2.5, "text", date(2024, 1, 1), datetime(2024, 1, 1, 9, 30), False, "a"]
    gold = expect(*[(v,) for v in mixed])
    shuffled = [mixed[i] for i in (5, 2, 7, 0, 3, 6, 1, 4)]
    assert same(frame(*[(v,) for v in shuffled]), gold)
    assert not same(frame(*[(v,) for v in [*shuffled[:-1], "b"]]), gold)


def test_the_search_over_column_pairings_stops_at_its_cap(monkeypatch) -> None:
    # Columns P, Q and R can each stand for either gold column; only (P, R) pairs the rows right.
    gold = expect((1, 2), (2, 1))
    candidate = pd.DataFrame({"p": [1, 2], "q": [1, 2], "r": [2, 1]})
    assert same(candidate, gold)
    monkeypatch.setattr(scoring, "MAX_ASSIGNMENTS", 1)
    assert not same(candidate, gold)


# -- unordered and ordered -------------------------------------------------------------------------


def test_unordered_ignores_row_order_but_counts_duplicates() -> None:
    gold = expect(("a", 1.0), ("b", 2.0), ("b", 2.0))
    assert same(frame(("b", 2.0), ("a", 1.0), ("b", 2.0)), gold)
    assert not same(frame(("a", 1.0), ("a", 1.0), ("b", 2.0)), gold)


def test_unordered_pairs_up_values_that_differ_in_the_last_digits() -> None:
    gold = expect(("a", 1.0), ("b", 1.0 + 4e-7))
    assert same(frame(("b", 1.0 + 2e-7), ("a", 1.0)), gold)


def test_ordered_needs_the_rows_in_the_order_of_the_sort_key() -> None:
    gold = expect(("a", 3.0), ("b", 2.0), ("c", 1.0), sort_columns=(1,))
    assert same(frame(("a", 3.0), ("b", 2.0), ("c", 1.0)), gold, "ordered")
    verdict = compare_results(frame(("c", 1.0), ("b", 2.0), ("a", 3.0)), gold, "ordered")
    assert not verdict.matched
    assert verdict.reason == "the rows differ or are not in the right order"


def test_ordered_lets_rows_with_equal_keys_come_in_any_order() -> None:
    gold = expect(("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 1.0), sort_columns=(1,))
    assert same(frame(("a", 3.0), ("c", 2.0), ("b", 2.0), ("d", 1.0)), gold, "ordered")
    assert not same(frame(("a", 3.0), ("c", 2.0), ("x", 2.0), ("d", 1.0)), gold, "ordered")


def test_ordered_with_an_unknown_sort_key_compares_rows_strictly() -> None:
    gold = expect(("a", 2.0), ("b", 2.0))
    assert same(frame(("a", 2.0), ("b", 2.0)), gold, "ordered")
    assert not same(frame(("b", 2.0), ("a", 2.0)), gold, "ordered")


def test_an_ordered_extra_column_does_not_break_the_sort_key_match() -> None:
    gold = expect((10.0,), (5.0,), sort_columns=(0,))
    assert same(frame(("x", 10.0), ("y", 5.0)), gold, "ordered")


def test_every_row_of_an_ordered_result_is_checked_not_only_the_keys() -> None:
    gold = expect(("a", 9.0), ("b", 7.0), ("c", 5.0), sort_columns=(1,))
    assert not same(frame(("x", 9.0), ("b", 7.0), ("c", 5.0)), gold, "ordered")
    assert not same(frame(("a", 9.0), ("x", 7.0), ("c", 5.0)), gold, "ordered")
    assert not same(frame(("a", 9.0), ("b", 7.0), ("x", 5.0)), gold, "ordered")


def test_a_limit_with_an_unknown_sort_key_is_compared_strictly() -> None:
    gold = Expected(
        sql="gold",
        frame=frame(("a", 9.0), ("b", 7.0)),
        unlimited=frame(("a", 9.0), ("b", 7.0), ("c", 7.0)),
        sort_columns=(),
    )
    assert not tied_at_cutoff(gold)
    assert same(frame(("a", 9.0), ("b", 7.0)), gold)
    assert not same(frame(("a", 9.0), ("c", 7.0)), gold)


# -- ties at a LIMIT cut-off -----------------------------------------------------------------------

TOP_2_OF_4 = {
    "frame": ((("a", 9.0), ("b", 7.0))),
    "unlimited": (("a", 9.0), ("b", 7.0), ("c", 7.0), ("d", 1.0)),
}


def top_two() -> Expected:
    return Expected(
        sql="gold",
        frame=frame(*TOP_2_OF_4["frame"]),
        unlimited=frame(*TOP_2_OF_4["unlimited"]),
        sort_columns=(1,),
    )


@pytest.mark.parametrize("mode", ["ordered", "unordered"])
def test_either_tied_row_is_right_at_the_cut_off(mode: str) -> None:
    gold = top_two()
    assert tied_at_cutoff(gold)
    assert same(frame(("a", 9.0), ("b", 7.0)), gold, mode)
    assert same(frame(("a", 9.0), ("c", 7.0)), gold, mode)


@pytest.mark.parametrize("mode", ["ordered", "unordered"])
def test_a_row_that_is_not_in_the_tie_is_wrong_at_the_cut_off(mode: str) -> None:
    assert not same(frame(("a", 9.0), ("d", 1.0)), top_two(), mode)


@pytest.mark.parametrize("mode", ["ordered", "unordered"])
def test_a_row_above_the_tie_must_still_be_there(mode: str) -> None:
    assert not same(frame(("b", 7.0), ("c", 7.0)), top_two(), mode)


def test_a_cut_through_no_tie_is_not_flagged_and_stays_strict() -> None:
    gold = Expected(
        sql="gold",
        frame=frame(("a", 9.0), ("b", 7.0)),
        unlimited=frame(("a", 9.0), ("b", 7.0), ("c", 6.0)),
        sort_columns=(1,),
    )
    assert not tied_at_cutoff(gold)
    assert not same(frame(("a", 9.0), ("c", 6.0)), gold, "ordered")


def test_a_gold_without_a_limit_has_no_cut_off() -> None:
    assert not tied_at_cutoff(expect(("a", 1.0), sort_columns=(1,)))


def test_a_cut_off_tie_that_covers_the_whole_result() -> None:
    gold = Expected(
        sql="gold",
        frame=frame(("a", 5.0)),
        unlimited=frame(("a", 5.0), ("b", 5.0), ("c", 5.0)),
        sort_columns=(1,),
    )
    assert same(frame(("c", 5.0)), gold, "ordered")
    assert not same(frame(("z", 5.0)), gold, "ordered")


# -- nothing found ---------------------------------------------------------------------------------


def test_no_rows_and_a_single_null_are_the_same_answer() -> None:
    nothing = pd.DataFrame({"revenue": pd.Series([], dtype="float64")})
    null = frame((None,), columns=["revenue"])
    assert same(nothing, expect((None,)), "scalar")
    assert same(null, Expected(sql="gold", frame=nothing), "scalar")
    assert same(nothing, Expected(sql="gold", frame=nothing))


def test_nothing_found_is_not_zero_and_not_a_row() -> None:
    nothing = pd.DataFrame({"revenue": pd.Series([], dtype="float64")})
    assert not same(frame((0.0,)), expect((None,)), "scalar")
    assert not same(frame((5.0,)), Expected(sql="gold", frame=nothing))
    assert not same(nothing, expect((5.0,)), "scalar")


def test_dates_and_midnight_timestamps_are_the_same_value() -> None:
    gold = expect((date(2024, 3, 1), 5.0), (date(2024, 4, 1), 6.0), sort_columns=(0,))
    candidate = frame((pd.Timestamp("2024-03-01"), 5.0), (pd.Timestamp("2024-04-01"), 6.0))
    assert same(candidate, gold, "ordered")
    text = frame(("2024-03", 5.0), ("2024-04", 6.0))
    assert not same(text, gold, "ordered")


def test_nan_and_none_are_the_same_missing_value() -> None:
    gold = expect(("a", None), ("b", 2.0))
    assert same(frame(("a", float("nan")), ("b", 2.0)), gold)


# -- reading the gold SQL --------------------------------------------------------------------------


def test_limit_free_sql_removes_limit_and_offset() -> None:
    sql = "SELECT a FROM t ORDER BY a LIMIT 5 OFFSET 2"
    assert limit_free_sql(sql) == "SELECT a FROM t ORDER BY a"
    assert limit_free_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x LIMIT 1") == (
        "WITH x AS (SELECT 1 AS a) SELECT a FROM x"
    )


def test_limit_free_sql_is_none_without_a_limit_or_for_bad_sql() -> None:
    assert limit_free_sql("SELECT a FROM t") is None
    assert limit_free_sql("SELECT FROM WHERE (") is None


@pytest.mark.parametrize(
    ("sql", "columns", "expected"),
    [
        ("SELECT a, b FROM t ORDER BY b DESC", ["a", "b"], (1,)),
        ("SELECT a, SUM(x) AS total FROM t GROUP BY a ORDER BY total DESC", ["a", "total"], (1,)),
        ("SELECT a, b FROM t ORDER BY 2 DESC, 1", ["a", "b"], (1, 0)),
        ("SELECT a, b FROM t ORDER BY B", ["a", "b"], (1,)),
        ("SELECT a, b FROM t ORDER BY b, b", ["a", "b"], (1,)),
        ("SELECT a, b FROM t", ["a", "b"], ()),
        ("SELECT a, b FROM t ORDER BY lower(a)", ["a", "b"], ()),
        ("SELECT a, b FROM t ORDER BY t.b", ["a", "b"], ()),
        ("SELECT a FROM t ORDER BY b", ["a"], ()),
        ("SELECT a FROM t ORDER BY 3", ["a"], ()),
        ("SELECT FROM WHERE (", ["a"], ()),
    ],
)
def test_sort_columns(sql: str, columns: list[str], expected: tuple[int, ...]) -> None:
    assert sort_columns(sql, columns) == expected


# -- schema linking --------------------------------------------------------------------------------

COLUMNS = {"region", "revenue", "order_date", "quantity"}


def test_referenced_columns_are_table_columns_only() -> None:
    sql = (
        "SELECT region, SUM(revenue) AS total FROM sales"
        " WHERE year(order_date) = 2024 GROUP BY region ORDER BY total"
    )
    assert referenced_columns(sql, COLUMNS) == {"region", "revenue", "order_date"}


def test_referenced_columns_of_nothing_or_nonsense_are_empty() -> None:
    assert referenced_columns(None, COLUMNS) == frozenset()
    assert referenced_columns("", COLUMNS) == frozenset()
    assert referenced_columns("SELECT FROM WHERE (", COLUMNS) == frozenset()


def test_linking_scores() -> None:
    gold = "SELECT region, SUM(revenue) FROM sales GROUP BY region"
    assert linking_scores(gold, gold, COLUMNS) == (1.0, 1.0)
    extra = "SELECT region, SUM(revenue), SUM(quantity) FROM sales GROUP BY region"
    precision, recall = linking_scores(extra, gold, COLUMNS)
    assert (round(precision, 3), recall) == (0.667, 1.0)
    narrow = "SELECT SUM(revenue) FROM sales"
    assert linking_scores(narrow, gold, COLUMNS) == (1.0, 0.5)
    assert linking_scores("SELECT COUNT(*) FROM sales", gold, COLUMNS) == (0.0, 0.0)
    assert linking_scores("SELECT COUNT(*) FROM sales", "SELECT COUNT(*) FROM sales", COLUMNS) == (
        1.0,
        1.0,
    )


# -- assumptions -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "shown", "found"),
    [
        ("summer is December to February", ["Summer is taken to be December to February."], True),
        (
            "summer is December to February",
            ["Summer means the months December, January and February"],
            True,
        ),
        ("summer is December to February", ["Summer is June to August"], False),
        (
            "second half is July to December",
            ["The second half of the year, July to December."],
            True,
        ),
        ("last 12 months", ["The last twelve months"], True),
        ("last 12 months", ["All months"], False),
        ("categories", ["Each category is shown"], True),
        ("", [], True),
        ("a the of", [], True),
        ("revenue", [], False),
    ],
)
def test_assumption_found(expected: str, shown: list[str], found: bool) -> None:
    assert assumption_found(expected, shown) is found


def test_an_assumption_needs_at_least_the_share_of_its_words() -> None:
    expected = "alpha beta gamma delta epsilon"  # five content words, so 60% is three
    assert assumption_found(expected, ["alpha beta gamma"])
    assert not assumption_found(expected, ["alpha beta"])
    assert assumption_found(expected, ["alpha beta"], share=0.4)


def test_an_assumption_may_be_spread_over_several_statements() -> None:
    assert assumption_found("winter is June to August", ["Winter is June", "to August"])


# -- safety ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "safe"),
    [
        (None, True),
        ("", True),
        ("SELECT 1", True),
        ("WITH a AS (SELECT 1) SELECT * FROM a", True),
        ("SELECT 1 UNION SELECT 2", True),
        ("DROP TABLE survey", False),
        ("DELETE FROM survey WHERE satisfaction = 1", False),
        ("UPDATE t SET a = 1", False),
        ("SELECT 1; DROP TABLE t", False),
        ("SELECT FROM WHERE (", False),
    ],
)
def test_is_read_only(sql, safe: bool) -> None:
    assert is_read_only(sql) is safe


# -- statistics ------------------------------------------------------------------------------------


def test_percentile() -> None:
    assert percentile([], 0.5) is None
    assert percentile([4.0], 0.95) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)


def test_ratio() -> None:
    assert ratio(1, 4) == 0.25
    assert ratio(0, 0) is None
