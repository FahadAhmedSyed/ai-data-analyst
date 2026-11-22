"""Render a dataset profile as the schema card shown to the language model (T1.07).

The card is plain text, one line per column, and is a pure function of a
``DatasetProfile``: it never touches the database. Per ADR 0010 it holds names,
types, null shares, distinct counts, min and max, date ranges, and the most
frequent values of text columns with 50 or fewer distinct values. It holds no
sample rows (the context builder, T1.11, adds those, the region rules and the
untrusted-data wrapper).

Everything in it comes from the user's file, so it is treated as untrusted:

* names and values are cut to 80 characters, with line breaks and other control
  characters replaced by spaces, so a cell cannot start a new line or section;
* values are written as JSON strings, so quotes and backslashes are escaped;
* the uploaded file's own name is left out.

The card must fit a token budget (architecture.md: under 3,000 tokens). When the
full card is too long it is rebuilt with less detail, in this order: fewer
frequent values per column, no frequent values, no null shares or distinct
counts, then names and types only. Every column is always listed.

Token counts are estimated at 3 characters per token, which errs high for this
kind of text; it has not been checked against a real tokenizer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from analyst.core.models import ColumnProfile, DatasetProfile
from analyst.ingestion.profiler import CARD_MAX_DISTINCT

MAX_CELL_CHARS = 80
DEFAULT_MAX_TOKENS = 3_000
CHARS_PER_TOKEN = 3


@dataclass(frozen=True)
class _Detail:
    """How much one rendering shows."""

    max_values: int
    counts: bool  # null share and distinct count
    ranges: bool  # min and max, date ranges
    flags: bool
    original_names: bool


# From the full card to the smallest. ``values`` of 0 means no frequent values.
_LEVELS = (
    _Detail(20, counts=True, ranges=True, flags=True, original_names=True),
    _Detail(10, counts=True, ranges=True, flags=True, original_names=True),
    _Detail(5, counts=True, ranges=True, flags=True, original_names=True),
    _Detail(0, counts=True, ranges=True, flags=True, original_names=True),
    _Detail(0, counts=False, ranges=True, flags=True, original_names=True),
    _Detail(0, counts=False, ranges=False, flags=False, original_names=False),
)
_FIRST_LEVEL_WITHOUT_VALUES = 3

_FLAG_TEXT = {
    "ambiguous_date": "day/month order assumed",
    "censored_values": "has censored values such as <0.5, kept as text",
    "mixed_types": "mix of numbers and text or unreadable values",
    "all_missing": "no values",
    "constant": "one value in every row",
}


@dataclass(frozen=True)
class SchemaCard:
    """A rendered card and how it was fitted to the budget."""

    text: str
    approx_tokens: int
    level: int  # 0 is full detail; larger numbers show less
    within_budget: bool


def clean_text(value: str, limit: int = MAX_CELL_CHARS) -> str:
    """Make user text safe to place on one line: no control characters, at most ``limit`` long."""
    flattened = "".join(" " if ch.isspace() or not ch.isprintable() else ch for ch in value)
    collapsed = " ".join(flattened.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def _quoted(value: str) -> str:
    return json.dumps(clean_text(value), ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    """A deliberately high estimate of the tokens in ``text``."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _share(count: int, rows: int) -> str:
    if rows == 0 or count == 0:
        return "0%"
    share = 100 * count / rows
    if share < 1:
        return "under 1%"
    return f"{round(share)}%" if share < 99.5 or count == rows else "over 99%"


def _number(value: float | int) -> str:
    return str(value) if isinstance(value, int) else f"{value:.6g}"


def _range(column: ColumnProfile) -> str | None:
    if column.date_min is not None and column.date_max is not None:
        low, high = column.date_min.isoformat(), column.date_max.isoformat()
        return low if low == high else f"{low} to {high}"
    if column.min_value is not None and column.max_value is not None:
        return f"min {_number(column.min_value)}, max {_number(column.max_value)}"
    return None


def _column_line(column: ColumnProfile, rows: int, detail: _Detail) -> str:
    name = column.sql_name
    original = clean_text(column.display_name)
    if detail.original_names and original.lower() != name:
        name += f" ({json.dumps(original, ensure_ascii=False)})"
    parts = [column.dtype]
    if detail.counts:
        parts += [f"{_share(column.null_count, rows)} missing", f"{column.distinct_count} distinct"]
    if detail.ranges and (span := _range(column)):
        parts.append(span)
    if detail.flags:
        parts += [f"[{_FLAG_TEXT[f]}]" for f in column.flags if f in _FLAG_TEXT]
    listable = (
        detail.max_values > 0
        and column.top_values
        and column.dtype in {"text", "boolean"}
        and 0 < column.distinct_count <= CARD_MAX_DISTINCT
    )
    if listable:
        shown = column.top_values[: detail.max_values]
        label = "values" if len(shown) >= column.distinct_count else f"top {len(shown)} values"
        parts.append(f"{label}: " + ", ".join(_quoted(v.value) for v in shown))
    return f"- {name}: " + "; ".join(parts)


def _render(profile: DatasetProfile, detail: _Detail) -> str:
    lines = [
        f"Table {profile.table_name}: {profile.row_count:,} rows, {profile.column_count} columns."
    ]
    if profile.primary_date_column and profile.latest_date is not None:
        lines.append(
            f"Latest date in the data: {profile.latest_date.isoformat()} "
            f"(column {profile.primary_date_column})."
        )
    lines.append("Columns:")
    lines += [_column_line(c, profile.row_count, detail) for c in profile.columns]
    return "\n".join(lines)


def render_schema_card(
    profile: DatasetProfile,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    include_values: bool = True,
) -> SchemaCard:
    """Render the card, reducing detail until it fits ``max_tokens``.

    Args:
        profile: The dataset profile.
        max_tokens: Budget for the card, using the estimate in ``estimate_tokens``.
        include_values: False for schema-only mode (ADR 0010): no frequent values.

    Returns:
        The card. ``within_budget`` is False only if even names and types are too long.
    """
    start = 0 if include_values else _FIRST_LEVEL_WITHOUT_VALUES
    text, level = "", start
    for level in range(start, len(_LEVELS)):
        text = _render(profile, _LEVELS[level])
        if estimate_tokens(text) <= max_tokens:
            break
    tokens = estimate_tokens(text)
    return SchemaCard(
        text=text, approx_tokens=tokens, level=level, within_budget=tokens <= max_tokens
    )
