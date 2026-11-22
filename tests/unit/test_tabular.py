from __future__ import annotations

import pandas as pd
import pytest

from analyst.core.errors import IngestionError
from analyst.ingestion.tabular import (
    apply_header,
    check_limits,
    detect_header_row,
    drop_empty,
    make_unique_names,
)


def grid(rows: list[list[object]]) -> pd.DataFrame:
    width = max(len(r) for r in rows)
    padded = [r + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded, dtype="string")


def test_header_on_first_row():
    g = grid([["Site", "Date", "Rainfall (mm)"], ["A", "01/02/2024", "3.2"]])
    assert detect_header_row(g) == 0


def test_header_below_title_and_blank_rows():
    g = grid(
        [
            ["Rainfall report 2020-2024", None, None],
            [None, None, None],
            ["Source: council", None, None],
            ["Site", "Date", "Rainfall (mm)"],
            ["A", "01/02/2024", "3.2"],
        ]
    )
    assert detect_header_row(g) == 3


def test_header_may_have_some_blank_cells():
    g = grid([["Site", None, "Rainfall"], ["A", "x", "3.2"]])
    assert detect_header_row(g) == 0


def test_numeric_first_row_means_no_header():
    g = grid([["1", "2.5", "3"], ["4", "5", "6"]])
    assert detect_header_row(g) is None


def test_number_like_values():
    g = grid([["$1,200.50", "(3.4)", "12%"], ["1", "2", "3"]])
    assert detect_header_row(g) is None


def test_empty_grid_has_no_header():
    assert detect_header_row(pd.DataFrame()) is None


def test_unique_names():
    names = make_unique_names(["Site", None, "  Rainfall \n (mm) ", "site", "Site (2)", ""])
    assert names == ["Site", "Column 2", "Rainfall (mm)", "site (2)", "Site (2) (2)", "Column 6"]
    assert len({n.casefold() for n in names}) == len(names)


def test_unique_names_triple_duplicate():
    assert make_unique_names(["A", "A", "A"]) == ["A", "A (2)", "A (3)"]


def test_apply_header_drops_preamble():
    g = grid([["Title"], ["Site", "Value"], ["A", "1"], ["B", "2"]])
    table, skipped = apply_header(g, 1)
    assert list(table.columns) == ["Site", "Value"]
    assert skipped == 1
    assert table["Site"].tolist() == ["A", "B"]


def test_apply_header_none_generates_names():
    g = grid([["1", "2"], ["3", "4"]])
    table, skipped = apply_header(g, None)
    assert list(table.columns) == ["Column 1", "Column 2"]
    assert skipped == 0
    assert len(table) == 2


@pytest.mark.parametrize("row", [-1, 5])
def test_apply_header_out_of_range(row):
    with pytest.raises(IngestionError) as info:
        apply_header(grid([["a"], ["b"]]), row)
    assert info.value.category == "no_header"


def test_drop_empty_rows_and_unnamed_empty_columns():
    table = pd.DataFrame(
        {
            "Site": ["A", None, "B", "  "],
            "Column 2": [None, None, None, None],
            "Notes": [None, None, None, None],
        },
        dtype="string",
    )
    out = drop_empty(table)
    assert list(out.columns) == ["Site", "Notes"]  # named empty columns are kept
    assert out["Site"].tolist() == ["A", "B"]


def test_check_limits():
    t = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]}, dtype="string")
    check_limits(t, max_rows=3, max_columns=2)
    with pytest.raises(IngestionError) as info:
        check_limits(t, max_rows=2, max_columns=2)
    assert info.value.category == "too_many_rows"
    with pytest.raises(IngestionError) as info:
        check_limits(t, max_rows=3, max_columns=1)
    assert info.value.category == "too_many_columns"
    with pytest.raises(IngestionError) as info:
        check_limits(t.iloc[:0], max_rows=3, max_columns=2)
    assert info.value.category == "empty"
