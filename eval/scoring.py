"""Scoring rules (eval/README.md section 4) as pure functions: no model, no database.

The central one is ``compare_results``: does the candidate's result say the same as the gold
result? It follows the README rule, with the choices made explicit here:

* **Values, not names.** Column names are ignored. Each gold column is matched to a
  different candidate column that holds the same values, so the candidate may name, order
  and add columns as it likes (extra columns are allowed, a missing one is not).
* **Numbers** match within a relative 1e-6 (and 1e-9 absolute, for values near zero).
  Dates and timestamps at midnight are the same thing.
* **Modes.** ``scalar``: one row, one value. ``unordered``: the rows as a multiset.
  ``ordered``: the rows in the order of the gold's ``ORDER BY`` keys; rows with equal keys
  may come in any order among themselves.
* **Ties at a LIMIT cut-off.** If the gold takes the top 5 and the 5th and 6th are equal,
  either is right. The tied rows come from the gold query run without its LIMIT.
* **Nothing found is nothing found.** No rows, and one row of NULL (a SUM over no rows),
  are the same answer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection, Hashable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

REL_TOL = 1e-6
ABS_TOL = 1e-9
# Beyond this many rows, rows that are not equal to nine digits are not paired up one by one.
PAIRING_LIMIT = 2_000
MAX_ASSIGNMENTS = 500

Cell = None | bool | float | str | date | datetime


# -- cells and rows --------------------------------------------------------------------------------


def normalise_cell(value: Any) -> Cell:
    """A cell as the comparison sees it: ``None``, bool, float, str, date or datetime."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | Decimal | np.integer | np.floating):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime(warn=False)
    if isinstance(value, datetime):
        value = value.replace(tzinfo=None)
        return value.date() if value.time() == datetime.min.time() else value
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return value.strip()
    return str(value)


def cells_match(a: Cell, b: Cell) -> bool:
    """Whether two normalised cells say the same thing."""
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return type(a) is type(b) and a == b


def _rows(frame: pd.DataFrame) -> list[tuple[Cell, ...]]:
    return [
        tuple(normalise_cell(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]


def _rows_match(a: Sequence[Cell], b: Sequence[Cell]) -> bool:
    return len(a) == len(b) and all(cells_match(x, y) for x, y in zip(a, b, strict=True))


def _hash_key(row: Sequence[Cell]) -> tuple[Hashable, ...]:
    """A row as a hashable key in which floats equal to nine digits are the same."""
    return tuple(float(f"{c:.9g}") if isinstance(c, float) else c for c in row)


def _sort_key(cell: Cell) -> tuple[int, Any]:
    """An order over mixed cells, so two columns with the same values sort the same way."""
    if cell is None:
        return (5, 0)
    if isinstance(cell, bool):
        return (0, cell)
    if isinstance(cell, float):
        return (1, cell)
    if isinstance(cell, str):
        return (2, cell)
    if isinstance(cell, datetime):
        return (4, cell)
    return (3, cell)


def _take(pool: Sequence[Sequence[Cell]], wanted: Sequence[Sequence[Cell]]) -> list | None:
    """``pool`` without one matching row for each of ``wanted``; ``None`` if one is missing."""
    left = list(pool)
    for row in wanted:
        for position, candidate in enumerate(left):
            if _rows_match(candidate, row):
                del left[position]
                break
        else:
            return None
    return left


def _multiset_equal(a: Sequence[Sequence[Cell]], b: Sequence[Sequence[Cell]]) -> bool:
    if len(a) != len(b):
        return False
    if Counter(map(_hash_key, a)) == Counter(map(_hash_key, b)):
        return True
    return len(a) <= PAIRING_LIMIT and _take(a, b) == []


def _column_equal(a: Sequence[Cell], b: Sequence[Cell]) -> bool:
    """Two columns hold the same values, whatever their order."""
    if len(a) != len(b):
        return False
    first, second = sorted(a, key=_sort_key), sorted(b, key=_sort_key)
    return all(cells_match(x, y) for x, y in zip(first, second, strict=True))


# -- reading the gold SQL --------------------------------------------------------------------------


def _parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, dialect="duckdb")
    except SqlglotError:
        return None


def limit_free_sql(sql: str) -> str | None:
    """``sql`` without its top-level LIMIT and OFFSET, or ``None`` if it has no LIMIT."""
    tree = _parse(sql)
    if tree is None or tree.args.get("limit") is None:
        return None
    tree.set("limit", None)
    tree.set("offset", None)
    return tree.sql(dialect="duckdb")


def sort_columns(sql: str, columns: Sequence[str]) -> tuple[int, ...]:
    """The output columns (by position) that ``sql``'s ORDER BY sorts on.

    Returns ``()`` when there is no ORDER BY, or when a key is not simply an output column
    named by alias or by its number (then the order of rows is compared strictly).
    """
    tree = _parse(sql)
    order = tree.args.get("order") if tree is not None else None
    if order is None:
        return ()
    names = [str(c).lower() for c in columns]
    found: list[int] = []
    for item in order.expressions:
        node = item.this
        if isinstance(node, exp.Literal) and node.is_int and 1 <= int(node.name) <= len(names):
            found.append(int(node.name) - 1)
        elif isinstance(node, exp.Column) and not node.table and node.name.lower() in names:
            found.append(names.index(node.name.lower()))
        else:
            return ()
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class Expected:
    """A gold query's result, and what is needed to judge ties.

    Attributes:
        sql: The gold query.
        frame: Its result.
        unlimited: The result of the same query without its LIMIT; ``None`` if it has none.
        sort_columns: Positions of the columns its ORDER BY sorts on; ``()`` if unknown.
    """

    sql: str
    frame: pd.DataFrame
    unlimited: pd.DataFrame | None = None
    sort_columns: tuple[int, ...] = ()


@dataclass(frozen=True)
class Comparison:
    """The verdict on one candidate result and why."""

    matched: bool
    reason: str = ""


# -- comparing results -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ties:
    """The rows tied at a LIMIT cut-off: ``slots`` of them are in the gold result."""

    group: list[tuple[Cell, ...]]
    slots: int
    key: tuple[Cell, ...]


def _key(row: Sequence[Cell], columns: Sequence[int]) -> tuple[Cell, ...]:
    return tuple(row[i] for i in columns)


def _ties(gold: list[tuple[Cell, ...]], expected: Expected) -> _Ties | None:
    if expected.unlimited is None or not expected.sort_columns or not gold:
        return None
    columns = expected.sort_columns
    cutoff = _key(gold[-1], columns)
    group = [r for r in _rows(expected.unlimited) if _rows_match(_key(r, columns), cutoff)]
    slots = sum(1 for r in gold if _rows_match(_key(r, columns), cutoff))
    return _Ties(group, slots, cutoff) if len(group) > slots else None


def tied_at_cutoff(expected: Expected) -> bool:
    """Whether the gold's LIMIT cuts through rows that tie on the sort key, so that more
    than one set of rows is a right answer. The shipped cases avoid this by their data."""
    return _ties(_rows(expected.frame), expected) is not None


def _unordered_equal(
    rows: list[tuple[Cell, ...]],
    gold: list[tuple[Cell, ...]],
    expected: Expected,
    ties: _Ties | None,
) -> bool:
    if ties is None:
        return _multiset_equal(rows, gold)
    columns = expected.sort_columns
    fixed = [r for r in gold if not _rows_match(_key(r, columns), ties.key)]
    rest = _take(rows, fixed)
    return rest is not None and len(rest) == ties.slots and _take(ties.group, rest) is not None


def _ordered_equal(
    rows: list[tuple[Cell, ...]],
    gold: list[tuple[Cell, ...]],
    expected: Expected,
    ties: _Ties | None,
) -> bool:
    columns = expected.sort_columns
    if not columns:
        return all(_rows_match(r, g) for r, g in zip(rows, gold, strict=True))
    if not all(
        _rows_match(_key(r, columns), _key(g, columns)) for r, g in zip(rows, gold, strict=True)
    ):
        return False
    start = 0
    while start < len(gold):
        end = start + 1
        while end < len(gold) and _rows_match(_key(gold[end], columns), _key(gold[start], columns)):
            end += 1
        run = rows[start:end]
        if end == len(gold) and ties is not None:
            if _take(ties.group, run) is None:
                return False
        elif not _multiset_equal(run, gold[start:end]):
            return False
        start = end
    return True


def _assignments(options: list[list[int]]) -> Iterator[tuple[int, ...]]:
    """Each gold column paired with a different candidate column, in turn."""

    def extend(position: int, used: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if position == len(options):
            yield used
            return
        for choice in options[position]:
            if choice not in used:
                yield from extend(position + 1, (*used, choice))

    return extend(0, ())


def _nothing(rows: list[tuple[Cell, ...]]) -> bool:
    """No rows, or one row with nothing in it."""
    return not rows or (len(rows) == 1 and all(c is None for c in rows[0]))


def compare_results(candidate: pd.DataFrame, expected: Expected, mode: str) -> Comparison:
    """Whether ``candidate`` is the same answer as ``expected`` (see the module docstring)."""
    gold = _rows(expected.frame)
    rows = _rows(candidate)
    if _nothing(gold) and _nothing(rows):
        return Comparison(True, "nothing found, as expected")
    wide = expected.frame.shape[1]
    if candidate.shape[1] < wide:
        return Comparison(False, f"{candidate.shape[1]} column(s), expected at least {wide}")
    if len(rows) != len(gold):
        return Comparison(False, f"{len(rows)} rows, expected {len(gold)}")
    ties = _ties(gold, expected)
    # At a tied cut-off a column may hold other tied rows' values, so it only has to be
    # drawn from the full result; the rows are judged together afterwards.
    pool = _rows(expected.unlimited) if ties is not None and expected.unlimited is not None else []
    gold_columns = [[row[i] for row in gold] for i in range(wide)]
    candidate_columns = [[row[i] for row in rows] for i in range(candidate.shape[1])]
    options = [
        [
            c
            for c, values in enumerate(candidate_columns)
            if _column_equal(wanted, values)
            or (
                pool
                and _take([(v,) for v in (r[i] for r in pool)], [(v,) for v in values]) is not None
            )
        ]
        for i, wanted in enumerate(gold_columns)
    ]
    for position, choices in enumerate(options):
        if not choices:
            if mode == "scalar" and wide == 1 and len(rows) == 1:
                got = ", ".join(repr(value) for value in rows[0])
                return Comparison(False, f"expected {gold[0][0]!r}, got {got}")
            name = str(expected.frame.columns[position])
            return Comparison(False, f"no column holds the values of {name!r}")
    for count, chosen in enumerate(_assignments(options)):
        if count >= MAX_ASSIGNMENTS:
            break
        projected = [tuple(row[c] for c in chosen) for row in rows]
        if mode == "scalar":
            same = _rows_match(projected[0], gold[0])
        elif mode == "ordered":
            same = _ordered_equal(projected, gold, expected, ties)
        else:
            same = _unordered_equal(projected, gold, expected, ties)
        if same:
            return Comparison(True, "match")
    if mode == "ordered":
        return Comparison(False, "the rows differ or are not in the right order")
    return Comparison(False, "the values are not in the same rows")


# -- schema linking --------------------------------------------------------------------------------


def referenced_columns(sql: str | None, columns: Collection[str]) -> frozenset[str]:
    """The table columns a query refers to (names that are aliases only are not counted)."""
    tree = _parse(sql) if sql else None
    if tree is None:
        return frozenset()
    known = {c.lower() for c in columns}
    return frozenset(
        node.name.lower() for node in tree.find_all(exp.Column) if node.name.lower() in known
    )


def linking_scores(
    candidate_sql: str | None, gold_sql: str, columns: Collection[str]
) -> tuple[float, float]:
    """Precision and recall of the columns the candidate used against the gold's."""
    used = referenced_columns(candidate_sql, columns)
    wanted = referenced_columns(gold_sql, columns)
    hit = len(used & wanted)
    precision = hit / len(used) if used else (1.0 if not wanted else 0.0)
    recall = hit / len(wanted) if wanted else 1.0
    return precision, recall


# -- assumptions -----------------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
        "on", "or", "that", "the", "this", "to", "was", "were", "with",
    }
)  # fmt: skip
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}  # fmt: skip


def _tokens(text: str) -> set[str]:
    found = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _STOP_WORDS:
            continue
        word = _NUMBER_WORDS.get(word, word)
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        found.add(word)
    return found


def assumption_found(expected: str, shown: Sequence[str], *, share: float = 0.6) -> bool:
    """Whether the response states what ``expected`` says: at least ``share`` of the
    expected phrase's content words appear in the assumptions together. A heuristic, which is
    why the report also lists each expected assumption beside what was shown."""
    wanted = _tokens(expected)
    if not wanted:
        return True
    have: set[str] = set()
    for text in shown:
        have |= _tokens(text)
    return len(wanted & have) / len(wanted) >= share


# -- safety ----------------------------------------------------------------------------------------


def is_read_only(sql: str | None) -> bool:
    """True when there is no SQL, or it is one query (a SELECT, possibly with a WITH)."""
    if not sql:
        return True
    try:
        trees = sqlglot.parse(sql, dialect="duckdb")
    except SqlglotError:
        return False
    return len(trees) == 1 and isinstance(trees[0], exp.Query)


# -- statistics ------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile (0 to 1) of ``values`` by linear interpolation; ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def ratio(hits: int, total: int) -> float | None:
    """``hits / total``, or ``None`` when there was nothing to measure."""
    return hits / total if total else None


__all__ = [
    "Cell",
    "Comparison",
    "Expected",
    "assumption_found",
    "cells_match",
    "compare_results",
    "is_read_only",
    "limit_free_sql",
    "linking_scores",
    "normalise_cell",
    "percentile",
    "ratio",
    "referenced_columns",
    "sort_columns",
    "tied_at_cutoff",
]
