"""Statements about a result that code writes, so they cannot be wrong (T1.14, FR-6.3).

Some things users must be told do not need a language model, and are better without one:

* a result was cut at the row cap;
* a column has missing values;
* the query ran but matched nothing, and which filters it applied;
* the fallback answer used when an explanation fails the grounding check (ADR 0017).

Every number written here is read straight from the result or the intent, so none of it
passes through ``check_grounding``.
"""

from __future__ import annotations

import math
import numbers

import pandas as pd

from analyst.core.models import FilterCondition, Intent, QueryResult
from analyst.ingestion.schema_card import clean_text

MAX_MISSING_NOTES = 5


def format_number(value: float | int) -> str:
    """A number as people write it: ``1,234``, ``1,234.57``, ``0.0042``."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (math.isfinite(value) and float(value).is_integer()):
        return f"{int(value):,}"
    if not math.isfinite(value):
        return str(value)
    return f"{value:,.2f}" if abs(value) >= 0.01 else f"{value:.4g}"


def result_caveats(result: QueryResult) -> list[str]:
    """Notes the user must see whatever the explanation says: truncation and missing values."""
    notes: list[str] = []
    if result.truncated:
        notes.append(
            f"The result was cut at {format_number(result.row_count)} rows, "
            "so there are more rows than are shown."
        )
    frame = result.dataframe
    missing = [(str(name), int(frame[name].isna().sum())) for name in frame.columns]
    missing = [(name, count) for name, count in missing if count]
    for name, count in missing[:MAX_MISSING_NOTES]:
        if count == result.row_count:
            notes.append(f"{clean_text(name)} has no value in any row.")
        else:
            notes.append(
                f"{clean_text(name)} is missing in {format_number(count)} of "
                f"{format_number(result.row_count)} rows."
            )
    if len(missing) > MAX_MISSING_NOTES:
        notes.append(
            f"{format_number(len(missing) - MAX_MISSING_NOTES)} more columns also have "
            "missing values."
        )
    return notes


_OPERATOR_TEXT = {
    "=": "is",
    "!=": "is not",
    ">": "is above",
    ">=": "is at least",
    "<": "is below",
    "<=": "is at most",
    "in": "is one of",
    "not_in": "is not one of",
    "between": "is between",
    "is_null": "is missing",
    "is_not_null": "is present",
    "contains": "contains",
    "month_in": "has month in",
    "year_in": "has year in",
}


def _value(value: str | float, *, plain: bool = False) -> str:
    """A filter value: text quoted, numbers with separators, calendar numbers as they are."""
    if isinstance(value, numbers.Real):
        return str(int(value)) if plain and float(value).is_integer() else format_number(value)
    return f"'{clean_text(str(value))}'"


def describe_filter(condition: FilterCondition) -> str:
    """One filter in words, for example ``region is one of 'North', 'South'``."""
    column = clean_text(condition.column)
    verb = _OPERATOR_TEXT.get(condition.operator, condition.operator)
    values = list(condition.values)
    if condition.operator in {"is_null", "is_not_null"} or not values:
        return f"{column} {verb}"
    if condition.operator == "between" and len(values) == 2:
        return f"{column} {verb} {_value(values[0])} and {_value(values[1])}"
    plain = condition.operator in {"month_in", "year_in"}
    return f"{column} {verb} " + ", ".join(_value(v, plain=plain) for v in values)


def empty_result_answer(intent: Intent | None) -> str:
    """The answer for a query that ran and returned no rows, naming the filters it used."""
    text = "The query ran but returned no rows."
    parts: list[str] = []
    if intent is not None:
        parts += [describe_filter(f) for f in intent.filters]
        if intent.time_range is not None:
            low, high = intent.time_range.start, intent.time_range.end
            parts.append(f"date range {low or 'open'} to {high or 'open'}")
    if parts:
        return f"{text} It filtered on: " + "; ".join(parts) + "."
    return f"{text} It had no filters, so the table has nothing that fits the question."


def template_answer(result: QueryResult, intent: Intent | None = None) -> str:
    """The plain answer used when no explanation passes the grounding check (ADR 0017)."""
    if result.is_empty:
        return empty_result_answer(intent)
    frame = result.dataframe
    if result.row_count == 1 and len(frame.columns) == 1:
        value = frame.iloc[0, 0]
        name = clean_text(str(frame.columns[0]))
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and not pd.isna(value):
            return f"Here is the result of the query: {name} is {format_number(value)}."
    columns = len(frame.columns)
    rows = format_number(result.row_count)
    return (
        f"Here is the result of the query: {rows} {_plural(result.row_count, 'row')} "
        f"and {columns} {_plural(columns, 'column')}."
    )


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else noun + "s"
