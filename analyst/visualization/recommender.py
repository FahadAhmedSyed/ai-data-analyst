"""Choose a chart for a query result from its shape (T1.15, architecture.md section 9).

The rules, applied to the result's column types:

=========================================================  =========
Result shape                                               Chart
=========================================================  =========
one date column and numeric columns                        line
one date, one text column (few values), one numeric        line, one line per text value
one text column (30 or fewer values) and numeric columns   bar
two numeric columns and nothing else                       scatter
one numeric column of raw values (ten rows or more)        histogram
anything else, or a single value                           none (table only)
=========================================================  =========

The model also suggests a chart (``GenerationResult.chart``). Its suggestion is used only
if every column it names exists and suits the chart type; otherwise the rules above decide.
A suggestion of ``none`` is respected: the model knows what was asked, for example a plain
listing, and a chart of it would add nothing.

Why charts are refused rather than bent to fit: a chart that draws several rows at one x
position, a bar for each of 400 categories, or a line for every one of 30 groups reads as
noise, and the table already shows the result. So a result that does not fit gets no chart.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from analyst.core.models import ChartSpec, QueryResult

Kind = Literal["date", "text", "numeric", "other"]

MAX_BAR_CATEGORIES = 30
MAX_SERIES = 8  # distinct colours; also the number of fixed categorical slots
MAX_SCATTER_SERIES = 3  # all-pairs forms keep to the first three slots (dataviz palette)
MAX_Y_COLUMNS = 5
MIN_ROWS = 2
MIN_SCATTER_ROWS = 3
MIN_HISTOGRAM_ROWS = 10


def column_kind(column: pd.Series) -> Kind:
    """How a result column can be drawn: as a date, a category, a number, or not at all."""
    if column.dropna().empty:
        return "other"
    if pd.api.types.is_bool_dtype(column):
        return "text"
    if pd.api.types.is_numeric_dtype(column):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(column):
        return "date"
    if pd.api.types.is_string_dtype(column) or isinstance(column.dtype, pd.CategoricalDtype):
        return "text"
    return "other"


def column_kinds(frame: pd.DataFrame) -> dict[str, Kind]:
    """The kind of each result column, keyed by column name."""
    return {str(name): column_kind(frame[name]) for name in frame.columns}


def _distinct(frame: pd.DataFrame, name: str) -> int:
    return int(frame[name].nunique(dropna=True))


def _repeats(frame: pd.DataFrame, keys: list[str]) -> bool:
    return bool(frame.duplicated(subset=keys).any())


def fit_problem(result: QueryResult, spec: ChartSpec) -> str | None:
    """Why ``spec`` cannot be drawn from ``result``, or ``None`` if it can.

    The reason names columns and counts only, never values from the data.
    """
    if spec.type == "none":
        return None
    frame = result.dataframe
    kinds = column_kinds(frame)
    if len(frame) < (MIN_SCATTER_ROWS if spec.type == "scatter" else MIN_ROWS):
        return "too few rows"
    if spec.x is None or spec.x not in kinds:
        return "the x column is missing from the result"
    used = [spec.x, *spec.y, *([spec.color] if spec.color else [])]
    missing = [c for c in used if c not in kinds]
    if missing:
        return f"no such column: {', '.join(missing)}"
    if len(set(used)) != len(used):
        return "a column is used twice"
    if spec.type == "histogram":
        return _histogram_problem(frame, spec, kinds)
    if not 1 <= len(spec.y) <= MAX_Y_COLUMNS:
        return f"needs between 1 and {MAX_Y_COLUMNS} value columns"
    if any(kinds[y] != "numeric" for y in spec.y):
        return "the value columns must be numeric"
    if spec.color is not None:
        problem = _color_problem(frame, spec, kinds)
        if problem:
            return problem
    return _axis_problem(frame, spec, kinds)


def _histogram_problem(frame: pd.DataFrame, spec: ChartSpec, kinds: dict[str, Kind]) -> str | None:
    if len(frame) < MIN_HISTOGRAM_ROWS:
        return "too few rows for a histogram"
    if spec.y or spec.color:
        return "a histogram draws one column"
    assert spec.x is not None  # noqa: S101 - checked by the caller
    return None if kinds[spec.x] == "numeric" else "a histogram needs a numeric column"


def _color_problem(frame: pd.DataFrame, spec: ChartSpec, kinds: dict[str, Kind]) -> str | None:
    assert spec.color is not None  # noqa: S101
    if kinds[spec.color] != "text":
        return "the colour column must hold categories"
    if len(spec.y) != 1:
        return "colouring by a column needs exactly one value column"
    limit = MAX_SCATTER_SERIES if spec.type == "scatter" else MAX_SERIES
    if _distinct(frame, spec.color) > limit:
        return f"more than {limit} groups"
    return None


def _axis_problem(frame: pd.DataFrame, spec: ChartSpec, kinds: dict[str, Kind]) -> str | None:
    assert spec.x is not None  # noqa: S101
    x_kind = kinds[spec.x]
    keys = [spec.x, *([spec.color] if spec.color else [])]
    if spec.type == "line":
        if x_kind not in {"date", "numeric"}:
            return "a line chart needs a date or number on the x axis"
        if _repeats(frame, keys):
            return "several rows share one x position"
        if spec.color is not None and frame.groupby(spec.color).size().max() < MIN_ROWS:
            return "every group has a single point, so there is no line to draw"
    elif spec.type == "bar":
        if x_kind not in {"text", "date"}:
            return "a bar chart needs categories on the x axis"
        if _distinct(frame, spec.x) > MAX_BAR_CATEGORIES:
            return f"more than {MAX_BAR_CATEGORIES} bars"
        if _repeats(frame, keys):
            return "several rows share one category"
    elif spec.type == "scatter":
        if x_kind != "numeric":
            return "a scatter plot needs a number on the x axis"
        if len(spec.y) != 1:
            return "a scatter plot draws one value column"
    return None


def _rule_based(result: QueryResult) -> ChartSpec:
    frame = result.dataframe
    kinds = column_kinds(frame)
    dates = [c for c, k in kinds.items() if k == "date"]
    texts = [c for c, k in kinds.items() if k == "text"]
    numbers = [c for c, k in kinds.items() if k == "numeric"]
    others = [c for c, k in kinds.items() if k == "other"]
    candidate = ChartSpec(type="none", x=None, y=[], color=None)
    if others:
        return candidate
    if len(dates) == 1 and numbers and len(texts) <= 1:
        if not texts:
            candidate = ChartSpec(type="line", x=dates[0], y=numbers, color=None)
        elif len(numbers) == 1:
            candidate = ChartSpec(type="line", x=dates[0], y=numbers, color=texts[0])
    elif not dates and len(texts) == 1 and numbers:
        candidate = ChartSpec(type="bar", x=texts[0], y=numbers, color=None)
    elif not dates and not texts and len(numbers) == 2:
        candidate = ChartSpec(type="scatter", x=numbers[0], y=[numbers[1]], color=None)
    elif not dates and not texts and len(numbers) == 1:
        candidate = ChartSpec(type="histogram", x=numbers[0], y=[], color=None)
    return (
        candidate
        if fit_problem(result, candidate) is None
        else ChartSpec(type="none", x=None, y=[], color=None)
    )


def recommend_chart(result: QueryResult, suggestion: ChartSpec | None = None) -> ChartSpec:
    """The chart to draw for ``result``: the model's suggestion if it fits, else by rule.

    Args:
        result: The executed result.
        suggestion: The model's chart suggestion, if it gave one.

    Returns:
        A spec that ``fit_problem`` accepts. ``type`` is ``"none"`` when the result is best
        shown as a table alone.
    """
    if suggestion is not None and fit_problem(result, suggestion) is None:
        return suggestion
    return _rule_based(result)
