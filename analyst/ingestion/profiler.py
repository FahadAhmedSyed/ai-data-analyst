"""Dataset profile and data quality summary, computed in DuckDB (T1.07).

``profile_database`` reads the session table with aggregate queries written by
this application and returns a ``DatasetProfile`` (architecture.md, section 4).
Nothing is copied out of DuckDB except the statistics and, for text columns, the
most frequent values.

Choices that are easy to get wrong:

* **Missing** means NULL in the typed table. Text such as ``N/A`` in a text column
  is a value, as loaded; the normaliser turned blanks into NULL.
* **Duplicate rows** is the number of rows identical to an earlier row, counting
  missing values as equal: rows minus distinct rows.
* **Standard deviation** is the sample standard deviation (n - 1), and quartiles
  use linear interpolation, matching pandas' defaults.
* **Most frequent values** are listed for text and boolean columns when the
  column has 50 or fewer distinct values, or when some value repeats; a column
  where every value is different (identifiers, free text) has none. Ties are
  broken by the value itself so the result is stable.
* **Primary date column** is the date or timestamp column with the fewest missing
  values, leftmost on ties. Its latest value anchors relative dates (ADR 0009).

Wide tables are profiled in chunks of aggregates so memory stays inside the
session cap. If a chunk still runs out of memory, distinct counts for that chunk
are approximate; if the duplicate count does, it uses a 64-bit row hash.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import duckdb

from analyst.core.logging import get_logger
from analyst.core.models import (
    ColumnProfile,
    ColumnSchema,
    DatasetProfile,
    QualityIssue,
    ValueCount,
)
from analyst.database.session_db import SessionDatabase, quote_identifier
from analyst.ingestion.normalize import ColumnNote

log = get_logger(__name__)

# Columns with this many distinct values or fewer have their values listed
# (ADR 0010), and the schema card shows them.
CARD_MAX_DISTINCT = 50
TOP_VALUES = 20
CHUNK_EXPRESSIONS = 20
_QUARTILES = "[0.25, 0.5, 0.75]"
_NUMERIC = frozenset({"integer", "decimal"})
_TEMPORAL = frozenset({"date", "timestamp"})


def _finite(value: Any) -> float | None:
    """A float, or ``None`` when DuckDB returned NULL, NaN or infinity."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _expressions(column: ColumnSchema, *, exact_distinct: bool) -> list[str]:
    """Aggregate expressions for one column, in the order ``_parse`` reads them."""
    q = quote_identifier(column.sql_name)
    distinct = f"count(DISTINCT {q})" if exact_distinct else f"approx_count_distinct({q})"
    expressions = [f"count({q})", distinct]
    if column.dtype in _NUMERIC:
        expressions += [
            f"min({q})",
            f"max({q})",
            f"avg({q})",
            f"stddev_samp({q})",
            f"quantile_cont({q}, {_QUARTILES})",
        ]
    elif column.dtype in _TEMPORAL:
        expressions += [f"min({q})", f"max({q})"]
    return expressions


def _width(column: ColumnSchema) -> int:
    return 2 + (5 if column.dtype in _NUMERIC else 2 if column.dtype in _TEMPORAL else 0)


def _chunks(columns: Sequence[ColumnSchema]) -> list[list[ColumnSchema]]:
    """Group columns so no query has more than about ``CHUNK_EXPRESSIONS`` expressions."""
    groups: list[list[ColumnSchema]] = []
    current: list[ColumnSchema] = []
    used = 0
    for column in columns:
        if current and used + _width(column) > CHUNK_EXPRESSIONS:
            groups.append(current)
            current, used = [], 0
        current.append(column)
        used += _width(column)
    if current:
        groups.append(current)
    return groups


def _aggregate_chunk(
    db: SessionDatabase, columns: Sequence[ColumnSchema]
) -> tuple[list[tuple[Any, ...]], bool]:
    """Statistics for a chunk of columns: one tuple per column, and whether distincts are exact."""
    for exact in (True, False):
        expressions = [e for c in columns for e in _expressions(c, exact_distinct=exact)]
        sql = f"SELECT {', '.join(expressions)} FROM {quote_identifier(db.table_name)}"  # noqa: S608
        try:
            row = db.query_rows(sql)[0]
        except duckdb.OutOfMemoryException:
            if not exact:
                raise
            log.warning("profile_distinct_approximate", extra={"columns": len(columns)})
            continue
        result, start = [], 0
        for column in columns:
            result.append(tuple(row[start : start + _width(column)]))
            start += _width(column)
        return result, exact
    raise AssertionError("unreachable")  # pragma: no cover


def _top_values(db: SessionDatabase, column: ColumnSchema) -> tuple[ValueCount, ...]:
    q = quote_identifier(column.sql_name)
    sql = (
        f"SELECT CAST({q} AS VARCHAR), count(*) FROM {quote_identifier(db.table_name)} "  # noqa: S608
        f"WHERE {q} IS NOT NULL GROUP BY {q} "
        f"ORDER BY count(*) DESC, {q} LIMIT {TOP_VALUES}"
    )
    return tuple(ValueCount(value=str(v), count=int(n)) for v, n in db.query_rows(sql))


def _duplicate_rows(db: SessionDatabase, rows: int) -> int:
    table = quote_identifier(db.table_name)
    try:
        sql = f"SELECT count(*) FROM (SELECT DISTINCT * FROM {table})"  # noqa: S608
        return rows - int(db.query_rows(sql)[0][0])
    except duckdb.OutOfMemoryException:
        log.warning("profile_duplicates_hashed")
        sql = f"SELECT count(DISTINCT hash(__row)) FROM {table} AS __row"  # noqa: S608
        return rows - int(db.query_rows(sql)[0][0])


def _column_profile(
    column: ColumnSchema, stats: tuple[Any, ...], top: tuple[ValueCount, ...], rows: int
) -> ColumnProfile:
    non_null, distinct = int(stats[0]), int(stats[1])
    fields: dict[str, Any] = {}
    if column.dtype in _NUMERIC:
        low, high, mean, std, quartiles = stats[2:7]
        fields = {
            "min_value": low,
            "max_value": high,
            "mean": _finite(mean),
            "std": _finite(std),
        }
        if quartiles is not None:
            fields |= {"q1": quartiles[0], "median": quartiles[1], "q3": quartiles[2]}
    elif column.dtype in _TEMPORAL:
        fields = {"date_min": stats[2], "date_max": stats[3]}
    return ColumnProfile(
        sql_name=column.sql_name,
        display_name=column.display_name,
        dtype=column.dtype,
        flags=column.flags,
        null_count=rows - non_null,
        distinct_count=distinct,
        top_values=top,
        **fields,
    )


def _primary_date_column(columns: Sequence[ColumnProfile]) -> ColumnProfile | None:
    candidates = [c for c in columns if c.dtype in _TEMPORAL and c.date_max is not None]
    return min(candidates, key=lambda c: c.null_count, default=None)  # min keeps the leftmost tie


def profile_database(db: SessionDatabase) -> DatasetProfile:
    """Profile the session's table.

    Raises:
        SessionCorruptedError: If the session is corrupted.
        RuntimeError: If no dataset is loaded.
    """
    started = time.perf_counter()
    with db.lock:
        columns = db.columns
        rows = int(db.query_rows(f"SELECT count(*) FROM {quote_identifier(db.table_name)}")[0][0])  # noqa: S608
        stats: dict[str, tuple[Any, ...]] = {}
        exact = True
        for chunk in _chunks(columns):
            chunk_stats, chunk_exact = _aggregate_chunk(db, chunk)
            exact = exact and chunk_exact
            stats |= {c.sql_name: s for c, s in zip(chunk, chunk_stats, strict=True)}

        profiles: list[ColumnProfile] = []
        for column in columns:
            non_null, distinct = int(stats[column.sql_name][0]), int(stats[column.sql_name][1])
            wants_values = column.dtype in {"text", "boolean"} and non_null > 0
            if wants_values and (distinct <= CARD_MAX_DISTINCT or distinct < non_null):
                top = _top_values(db, column)
            else:
                top = ()
            profiles.append(_column_profile(column, stats[column.sql_name], top, rows))
        duplicates = _duplicate_rows(db, rows) if rows else 0

    primary = _primary_date_column(profiles)
    profile = DatasetProfile(
        table_name=db.table_name,
        source_name=db.source_name,
        sheet_name=db.sheet_name,
        row_count=rows,
        column_count=len(columns),
        columns=tuple(profiles),
        duplicate_rows=duplicates,
        primary_date_column=primary.sql_name if primary else None,
        latest_date=primary.date_max if primary else None,
    )
    log.info(
        "table_profiled",
        extra={
            "rows": rows,
            "columns": len(columns),
            "duplicate_rows": duplicates,
            "exact_distinct": exact,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return profile


# ---------------------------------------------------------------------------
# Quality summary (FR-3.7)
# ---------------------------------------------------------------------------

_KIND_ORDER = (
    "duplicate_rows",
    "all_missing",
    "mixed_types",
    "censored_values",
    "ambiguous_date",
    "missing_values",
    "constant",
)
_TYPE_PHRASES = {
    "integer": "whole numbers",
    "decimal": "numbers",
    "date": "dates",
    "timestamp": "dates and times",
    "boolean": "true/false values",
    "text": "text",
}


def _percent(count: int, rows: int) -> str:
    """A share such as ``3.8%`` or ``20%``, never rounding a partial share to 0% or 100%."""
    if rows == 0 or count == 0:
        return "0%"
    share = 100 * count / rows
    if share < 0.1:
        return "under 0.1%"
    if count < rows and share > 99.9:
        return "over 99.9%"
    return f"{share:.1f}".removesuffix(".0") + "%"


def quality_summary(
    profile: DatasetProfile, notes: Mapping[str, ColumnNote] | None = None
) -> tuple[QualityIssue, ...]:
    """List what a user should know about the data's quality (FR-3.7).

    The data is never changed; this only reports. Messages use the user's original
    column names and counts, never cell values.

    Args:
        profile: The profile from ``profile_database``.
        notes: Per-column notes from ``normalize`` (``TypedTable.notes``). They add
            the number of values read as missing and how dates were interpreted.

    Returns:
        Issues grouped by kind (duplicates first), in column order within each kind.
    """
    notes = notes or {}
    rows = profile.row_count
    issues: list[QualityIssue] = []
    if profile.duplicate_rows:
        n = profile.duplicate_rows
        issues.append(
            QualityIssue(
                kind="duplicate_rows",
                count=n,
                message=f"{n:,} rows ({_percent(n, rows)}) are exact copies of an earlier row.",
            )
        )
    for column in profile.columns:
        name = column.display_name
        note = notes.get(column.sql_name, ColumnNote())
        all_missing = "all_missing" in column.flags or (rows > 0 and column.null_count == rows)
        if all_missing:
            issues.append(
                QualityIssue(
                    kind="all_missing",
                    column=column.sql_name,
                    count=rows,
                    message=f"{name} has no values.",
                )
            )
        if "mixed_types" in column.flags:
            if column.dtype == "text":
                message = f"{name} mixes numbers and text, so it was kept as text."
            elif note.coerced_to_missing:
                message = (
                    f"{note.coerced_to_missing:,} values in {name} could not be read as "
                    f"{_TYPE_PHRASES[column.dtype]} and were treated as missing."
                )
            else:
                message = (
                    f"Some values in {name} could not be read as "
                    f"{_TYPE_PHRASES[column.dtype]} and were treated as missing."
                )
            issues.append(
                QualityIssue(
                    kind="mixed_types",
                    column=column.sql_name,
                    count=note.coerced_to_missing,
                    message=message,
                )
            )
        if "censored_values" in column.flags:
            issues.append(
                QualityIssue(
                    kind="censored_values",
                    column=column.sql_name,
                    message=(
                        f"{name} has censored values such as '<0.5' or 'ND'. They are kept as "
                        "text and not converted to numbers."
                    ),
                )
            )
        if "ambiguous_date" in column.flags:
            detail = f" ({note.detail})" if note.detail else ""
            issues.append(
                QualityIssue(
                    kind="ambiguous_date",
                    column=column.sql_name,
                    message=(
                        f"The day and month order in {name} could not be settled from the "
                        f"values{detail}."
                    ),
                )
            )
        if column.null_count and not all_missing:
            issues.append(
                QualityIssue(
                    kind="missing_values",
                    column=column.sql_name,
                    count=column.null_count,
                    message=(
                        f"{name} is missing in {column.null_count:,} "
                        f"{'row' if column.null_count == 1 else 'rows'} "
                        f"({_percent(column.null_count, rows)})."
                    ),
                )
            )
        if "constant" in column.flags:
            issues.append(
                QualityIssue(
                    kind="constant",
                    column=column.sql_name,
                    message=f"{name} has the same value in every row.",
                )
            )
    return tuple(sorted(issues, key=lambda issue: _KIND_ORDER.index(issue.kind)))
