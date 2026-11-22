"""Pick what of a result the explainer is shown, and summarise the rest (T1.14, ADR 0010).

The explainer sees every row of a short result (50 rows or fewer). Of a longer one it sees
the first 20 rows plus statistics computed here, in code, over all returned rows, so it can
describe the whole result without being sent all of it. Wide results are cut to their first
30 columns, which keeps the prompt small whatever the table looks like.

The statistics are numbers the model may write in its answer. They are not values of any
result column, so ``ResultDigest.numbers`` hands them to the grounding check as extra
allowed numbers; without that, quoting a correct maximum would count as an invented number.

Nothing here calls a model, and every number is computed by pandas from the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from analyst.core.models import QueryResult
from analyst.ingestion.schema_card import clean_text
from analyst.pipeline.result_notes import format_number

FULL_ROWS = 50
SAMPLE_ROWS = 20
MAX_COLUMNS = 30
TOP_VALUES = 3

Kind = Literal["number", "date", "text", "other"]


@dataclass(frozen=True)
class ColumnStats:
    """One column's summary: a line of text for the prompt and the numbers it contains."""

    name: str
    kind: Kind
    line: str
    numbers: tuple[float, ...]


@dataclass(frozen=True)
class ResultDigest:
    """What the explainer is shown of one result."""

    rows: pd.DataFrame  # the rows shown, over the columns shown
    columns_shown: tuple[str, ...]
    columns_hidden: int
    total_rows: int
    complete: bool  # every returned row is in ``rows``
    stats: tuple[ColumnStats, ...]  # empty when ``complete``

    @property
    def numbers(self) -> tuple[float, ...]:
        """Numbers the explainer may quote that are not values of the result."""
        values = [float(len(self.rows))]
        if self.columns_hidden:
            values.append(float(self.columns_hidden))
        for stats in self.stats:
            values.extend(stats.numbers)
        return tuple(values)


def _kind(column: pd.Series) -> Kind:
    if column.dropna().empty:
        return "other"
    if pd.api.types.is_bool_dtype(column):
        return "text"
    if pd.api.types.is_numeric_dtype(column):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(column):
        return "date"
    if pd.api.types.is_string_dtype(column) or isinstance(column.dtype, pd.CategoricalDtype):
        return "text"
    return "other"


def _count_text(present: int, missing: int) -> str:
    return f"{format_number(present)} values, {format_number(missing)} missing"


def _number_stats(name: str, column: pd.Series) -> ColumnStats:
    """Statistics over the finite values; infinite and missing ones are counted as missing."""
    values = pd.to_numeric(column, errors="coerce").dropna()
    values = values[[math.isfinite(float(v)) for v in values]]
    present = len(values)
    missing = len(column) - present
    if values.empty:
        return ColumnStats(name, "number", f"{name} (number): no finite values", ())
    figures = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "sum": float(values.sum()),
    }
    text = "; ".join(f"{label} {format_number(v)}" for label, v in figures.items())
    line = f"{name} (number): {_count_text(present, missing)}; {text}"
    return ColumnStats(name, "number", line, (float(present), float(missing), *figures.values()))


def _date_stats(name: str, column: pd.Series, present: int, missing: int) -> ColumnStats:
    stamps = pd.to_datetime(column).dropna()
    midnight = bool((stamps == stamps.dt.normalize()).all())
    fmt = "%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S"
    line = (
        f"{name} (date): {_count_text(present, missing)}; "
        f"earliest {stamps.min().strftime(fmt)}; latest {stamps.max().strftime(fmt)}"
    )
    return ColumnStats(name, "date", line, (float(present), float(missing)))


def _text_stats(name: str, column: pd.Series, present: int, missing: int) -> ColumnStats:
    counts = column.dropna().astype(str).value_counts()
    top = [(clean_text(str(value)), int(count)) for value, count in counts.head(TOP_VALUES).items()]
    common = ", ".join(f"'{value}' {format_number(count)}" for value, count in top)
    line = (
        f"{name} (text): {_count_text(present, missing)}; "
        f"{format_number(len(counts))} distinct; most common: {common}"
    )
    numbers = (float(present), float(missing), float(len(counts)), *(float(c) for _, c in top))
    return ColumnStats(name, "text", line, numbers)


def column_stats(name: str, column: pd.Series) -> ColumnStats:
    """Summarise one result column over all of its rows."""
    label = clean_text(name)
    missing = int(column.isna().sum())
    present = len(column) - missing
    kind = _kind(column)
    if kind == "number":
        return _number_stats(label, column)
    if kind == "date":
        return _date_stats(label, column, present, missing)
    if kind == "text":
        return _text_stats(label, column, present, missing)
    return ColumnStats(label, "other", f"{label} (other): not summarised", ())


def digest_result(result: QueryResult) -> ResultDigest:
    """Choose the rows and statistics the explainer is shown for ``result``."""
    frame = result.dataframe
    shown_columns = [str(c) for c in frame.columns[:MAX_COLUMNS]]
    hidden = max(len(frame.columns) - MAX_COLUMNS, 0)
    complete = len(frame) <= FULL_ROWS
    rows = frame.iloc[: len(frame) if complete else SAMPLE_ROWS, :MAX_COLUMNS]
    stats: tuple[ColumnStats, ...] = ()
    if not complete:
        stats = tuple(column_stats(str(c), frame[c]) for c in frame.columns[:MAX_COLUMNS])
    return ResultDigest(
        rows=rows,
        columns_shown=tuple(shown_columns),
        columns_hidden=hidden,
        total_rows=len(frame),
        complete=complete,
        stats=stats,
    )
