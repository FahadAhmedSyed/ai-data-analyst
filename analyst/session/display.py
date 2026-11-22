"""Turning profiles, results and intents into things the interface can show (T1.18).

Everything here is a pure function with no Streamlit in it, so what a user is shown is
tested without a browser. Three rules run through it:

* **Text from the file or the model never becomes markup** (security.md, T12). A column
  name, a file name or a filter value can hold ``![x](http://...)`` or a link; shown through
  markdown it would load an address chosen by the file's author. ``escape_markdown`` makes
  such text literal, and the interface uses it for everything that did not come from our own
  code or from the cleaned explanation.
* **Tables are made of text.** Every cell of the profile tables is already formatted, so a
  column never mixes numbers and text (which Arrow, the format Streamlit sends tables in,
  refuses) and nothing is rounded in two places.
* **A download is safe to open in a spreadsheet** (``csv_bytes``).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from analyst.core.models import (
    ColumnProfile,
    ColumnType,
    DatasetProfile,
    FilterCondition,
    Intent,
    TimeRange,
)
from analyst.database.session_db import SessionDatabase, quote_identifier
from analyst.ingestion.schema_card import clean_text

PREVIEW_ROWS = 50
MAX_TOP_VALUES = 5

TYPE_LABELS: dict[ColumnType, str] = {
    "integer": "Whole number",
    "decimal": "Decimal number",
    "text": "Text",
    "boolean": "True/false",
    "date": "Date",
    "timestamp": "Date and time",
}
FLAG_LABELS = {
    "ambiguous_date": "day/month order assumed",
    "censored_values": "has censored values",
    "mixed_types": "mixed types",
    "numeric_text": "numbers read from text",
    "all_missing": "no values",
    "constant": "one value only",
}
_NUMERIC: frozenset[ColumnType] = frozenset({"integer", "decimal"})
_TEMPORAL: frozenset[ColumnType] = frozenset({"date", "timestamp"})
_OPERATOR_TEXT = {
    "=": "is",
    "!=": "is not",
    ">": "is more than",
    ">=": "is at least",
    "<": "is less than",
    "<=": "is at most",
    "in": "is one of",
    "not_in": "is not one of",
    "between": "is between",
    "is_null": "is missing",
    "is_not_null": "is present",
    "contains": "contains",
    "month_in": "has a month in",
    "year_in": "has a year in",
}
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


# -- text -----------------------------------------------------------------------------------------


_AUTOLINK_STARTS = re.compile(r"(?<=:)(?=//)|(?<=www)(?=\.)|(?<=@)", re.IGNORECASE)
ZERO_WIDTH_SPACE = "\u200b"


def escape_markdown(text: str) -> str:
    """Make ``text`` appear literally when shown through markdown.

    Every ASCII punctuation mark is backslash-escaped, which CommonMark defines as the way to
    write it literally, so no link, image, heading, list, emphasis, code span, emoji
    shortcode or colour directive can form. Line breaks and runs of spaces become one space,
    so text cannot begin a block of its own.

    That is not enough for web and email addresses. Streamlit's markdown turns a bare
    ``http://...``, ``www....`` or ``name@host`` into a link *after* the escapes are read,
    so a column name or a typed question could carry a link to an address of the file's
    author. A zero-width space is put inside each such start (after ``:`` before ``//``, after
    ``www``, after ``@``), which stops the match and is invisible on the page.
    """
    flat = " ".join(text.split())
    flat = _AUTOLINK_STARTS.sub(ZERO_WIDTH_SPACE, flat)
    return "".join(f"\\{ch}" if ch in string.punctuation else ch for ch in flat)


def escape_dollars(text: str) -> str:
    """Keep ``$`` literal in markdown, which would otherwise read ``$5 and $10`` as a formula."""
    return text.replace("$", "\\$")


def share_text(count: int, rows: int) -> str:
    """A share such as ``3.8%`` or ``20%``, never rounding a partial share to 0% or 100%."""
    if rows == 0 or count == 0:
        return "0%"
    share = 100 * count / rows
    if share < 0.1:
        return "under 0.1%"
    if count < rows and share > 99.9:
        return "over 99.9%"
    return f"{share:.1f}".removesuffix(".0") + "%"


def count_text(count: int, noun: str) -> str:
    """A count with its noun, ``1 row`` or ``2,000 rows``. The noun takes a plain ``s``."""
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def number_text(value: float | int | None) -> str:
    """A number for reading: thousands separators, two decimals above 1, else 4 significant."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"


def date_text(value: date | datetime | None) -> str:
    """A date as ISO text, with the time for a timestamp."""
    if value is None:
        return ""
    return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()


# -- the dataset ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileTables:
    """The profile as four tables. Each is empty when the dataset has no such column."""

    overview: pd.DataFrame
    numbers: pd.DataFrame
    text: pd.DataFrame
    dates: pd.DataFrame


def _overview_row(column: ColumnProfile, rows: int) -> dict[str, str]:
    notes = "; ".join(FLAG_LABELS[f] for f in column.flags)
    return {
        "Column": clean_text(column.display_name),
        "Type": TYPE_LABELS[column.dtype],
        "Missing": f"{column.null_count:,}",
        "Missing %": share_text(column.null_count, rows),
        "Different values": f"{column.distinct_count:,}",
        "Notes": notes,
    }


def _number_row(column: ColumnProfile) -> dict[str, str]:
    return {
        "Column": clean_text(column.display_name),
        "Smallest": number_text(column.min_value) if not isinstance(column.min_value, str) else "",
        "Largest": number_text(column.max_value) if not isinstance(column.max_value, str) else "",
        "Average": number_text(column.mean),
        "Std deviation": number_text(column.std),
        "25%": number_text(column.q1),
        "Median": number_text(column.median),
        "75%": number_text(column.q3),
    }


def _text_row(column: ColumnProfile) -> dict[str, str]:
    common = ", ".join(
        f"{clean_text(v.value)} ({v.count:,})" for v in column.top_values[:MAX_TOP_VALUES]
    )
    return {
        "Column": clean_text(column.display_name),
        "Different values": f"{column.distinct_count:,}",
        "Most common": common,
    }


def _date_row(column: ColumnProfile) -> dict[str, str]:
    return {
        "Column": clean_text(column.display_name),
        "Earliest": date_text(column.date_min),
        "Latest": date_text(column.date_max),
    }


def profile_tables(profile: DatasetProfile) -> ProfileTables:
    """The profile for display (FR-3.1 to FR-3.5), in the columns' original names and order."""
    columns = profile.columns
    rows = profile.row_count
    return ProfileTables(
        overview=pd.DataFrame([_overview_row(c, rows) for c in columns]),
        numbers=pd.DataFrame([_number_row(c) for c in columns if c.dtype in _NUMERIC]),
        text=pd.DataFrame([_text_row(c) for c in columns if c.dtype in {"text", "boolean"}]),
        dates=pd.DataFrame([_date_row(c) for c in columns if c.dtype in _TEMPORAL]),
    )


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A table for the screen. A date-and-time column whose values are all midnight is a column
    of dates (DuckDB hands dates over as timestamps), so it is shown as dates, not as
    ``2024-01-01 00:00:00``."""
    shown = frame.copy()
    for position in range(shown.shape[1]):
        column = shown.iloc[:, position]
        if not pd.api.types.is_datetime64_any_dtype(column) or column.dt.tz is not None:
            continue
        present = column.dropna()
        if len(present) and (present == present.dt.normalize()).all():
            shown.isetitem(position, column.dt.date)
    return shown


def preview_frame(db: SessionDatabase, profile: DatasetProfile) -> pd.DataFrame:
    """The first ``PREVIEW_ROWS`` rows as loaded and typed, under the user's own column names."""
    sql = f"SELECT * FROM {quote_identifier(db.table_name)} LIMIT {PREVIEW_ROWS}"  # noqa: S608
    return display_frame(db.query_df(sql).rename(columns=profile.display_names))


# -- an answer ------------------------------------------------------------------------------------


def _value_text(value: str | float) -> str:
    if isinstance(value, str):
        return f"'{clean_text(value)}'"
    return f"{value:g}"


def describe_filter(condition: FilterCondition, names: dict[str, str]) -> str:
    """One filter in words, for example ``Region is one of 'North', 'South'``."""
    column = clean_text(names.get(condition.column, condition.column))
    verb = _OPERATOR_TEXT[condition.operator]
    values = [_value_text(v) for v in condition.values]
    if condition.operator == "between" and len(values) == 2:
        operand = f"{values[0]} and {values[1]}"
    else:
        operand = ", ".join(values)
    return " ".join(part for part in (column, verb, operand) if part)


def describe_time_range(time_range: TimeRange) -> str:
    """The period a question was read as covering, with its dates."""
    description = clean_text(time_range.description)
    if time_range.start and time_range.end:
        span = f"{time_range.start} to {time_range.end}"
    elif time_range.start:
        span = f"from {time_range.start}"
    elif time_range.end:
        span = f"up to {time_range.end}"
    else:
        return description
    return f"{description} ({span})" if description else span


def describe_intent(intent: Intent | None, names: dict[str, str]) -> tuple[str, ...]:
    """The filters and period of a question, one line each (FR-9.1)."""
    if intent is None:
        return ()
    lines = [describe_filter(f, names) for f in intent.filters]
    if intent.time_range is not None:
        lines.append(describe_time_range(intent.time_range))
    return tuple(lines)


# -- download -------------------------------------------------------------------------------------


def _defuse(value: object) -> object:
    """Stop a text cell from being read as a formula by a spreadsheet (OWASP CSV injection)."""
    if isinstance(value, str) and value.startswith(_FORMULA_STARTS):
        return "'" + value
    return value


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """A result as a CSV file that opens correctly in Excel (FR-8.3).

    UTF-8 with a byte order mark, so accented letters and macrons survive. A text cell or
    heading that begins with ``=``, ``+``, ``-``, ``@``, a tab or a carriage return gets a
    leading apostrophe, because a spreadsheet would otherwise run it as a formula: the file
    may have come from someone else. Numbers are never touched, so a negative number stays
    a number.
    """
    safe = frame.copy()
    safe.columns = [_defuse(str(c)) for c in safe.columns]
    for position in range(safe.shape[1]):  # by position: result columns may share a name
        column = safe.iloc[:, position]
        if column.dtype == object or isinstance(column.dtype, pd.StringDtype):
            safe.isetitem(position, column.map(_defuse))
    return safe.to_csv(index=False).encode("utf-8-sig")
