"""Type inference and column naming (T1.05).

Turns a ``RawTable`` (all text) into a typed table ready for DuckDB, plus a
``ColumnSchema`` per column. Rules, in the order they are tried per column:

1. **Boolean** when every value is true/false/yes/no.
2. **Number** when at least 95% of values parse after removing currency signs,
   thousands separators, percent signs and accounting-style parentheses. Codes
   with leading zeros (``007``, ``0800``) stay text. Values such as ``<0.5`` or
   ``ND`` (censored lab results) keep the whole column as text, flagged
   ``censored_values``, because converting them silently would change results.
3. **Date or timestamp** when at least 95% of values parse with one detected
   layout. Day and month order comes from the values when any part exceeds 12;
   otherwise the region default applies (day first for NZ) and the column is
   flagged ``ambiguous_date``.
4. **Text** otherwise, with surrounding whitespace removed.

Values that fail to parse in an otherwise numeric or date column become
missing, the column is flagged ``mixed_types``, and the count is reported, so
nothing changes without the user being able to see it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd

from analyst.config.domain import get_region
from analyst.config.settings import Settings, get_settings
from analyst.core.logging import get_logger
from analyst.core.models import ColumnFlag, ColumnSchema, ColumnType
from analyst.ingestion.tabular import RawTable

log = get_logger(__name__)

PARSE_THRESHOLD = 0.95
MAX_IDENTIFIER_LENGTH = 63
DATE_SAMPLE_SIZE = 2_000

MISSING_MARKERS = frozenset(
    {
        "na", "n/a", "nan", "null", "none", "-", "--", "?", ".",
        "#n/a", "#null!", "#div/0!", "#value!", "#ref!", "#name?", "#num!",
    }
)  # fmt: skip
BOOLEAN_VALUES = {"true": True, "false": False, "yes": True, "no": False}

# Patterns are plain strings so pandas runs them in Arrow (RE2), which is much
# faster than Python's ``re`` on large columns. Each is used with ``match`` or
# ``fullmatch``, so any top-level alternation is wrapped in a group.
_CENSORED = r"(?i)(?:[<>]=?\s*\d+(?:\.\d+)?|nd|n\.d\.?|bdl|<\s*lod|<\s*lor|<\s*dl)"
_CURRENCY = r"(?i)(?:nz|us|au|a)?\$|€|£|¥"
_THOUSANDS = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
_LEADING_ZERO_CODE = r"0\d+"
_PARENS = r"\(.*\)"
_NUMERIC_DECOR = r"[$€£¥%,()]"
_INTEGER = r"-?\d+"

_ISO = r"\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
_YEAR_MONTH = r"\d{4}-\d{1,2}"
_MONTH_NAME = r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
_FOUR_DIGIT_YEAR = r"\b\d{4}\b"
# Uses a back-reference, which RE2 lacks; it only ever sees the date sample.
_NUMERIC_DATE = re.compile(
    r"^(?P<a>\d{1,2})(?P<sep>[/.-])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{4}|\d{2})"
    r"(?P<time>[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?)?$"
)
_TIME_FORMATS = ("", " %H:%M", " %H:%M:%S", " %I:%M %p", " %I:%M:%S %p", " %I:%M%p", " %I:%M:%S%p")


@dataclass(frozen=True)
class ColumnNote:
    """What inference decided for one column, for display in the profile."""

    detail: str = ""
    coerced_to_missing: int = 0


@dataclass(frozen=True)
class TypedTable:
    """A typed table whose columns are named by SQL identifiers."""

    dataframe: pd.DataFrame
    columns: tuple[ColumnSchema, ...]
    source_name: str
    sheet_name: str | None
    notes: dict[str, ColumnNote] = field(default_factory=dict)


@dataclass(frozen=True)
class _Inferred:
    dtype: ColumnType
    series: pd.Series
    flags: tuple[ColumnFlag, ...] = ()
    note: ColumnNote = ColumnNote()


@dataclass(frozen=True)
class _DateLayout:
    formats: tuple[str, ...]
    description: str
    day_first: bool = True
    ambiguous: bool = False


@dataclass(frozen=True)
class _Distinct:
    """A text column reduced to its distinct non-missing values.

    Parsing runs once per distinct value instead of once per row, which keeps a
    50 MB file inside the load-time target (NFR-6). Shares and counts are
    weighted by the number of rows holding each value.
    """

    index: pd.Index
    codes: np.ndarray  # per row: position in ``values``, or -1 when missing
    values: pd.Series  # distinct stripped values, in order of first appearance
    counts: np.ndarray  # rows holding each value

    @classmethod
    def of(cls, series: pd.Series) -> _Distinct:
        """Strip values, drop blanks and missing markers, and factorise the rest."""
        codes, uniques = pd.factorize(series.str.strip())
        values = pd.Series(uniques, dtype="string")
        missing = (values == "") | values.str.lower().isin(MISSING_MARKERS)
        keep = ~missing.to_numpy(dtype=bool)
        if not keep.all():
            position = np.cumsum(keep) - 1
            position[~keep] = -1
            codes = np.where(codes >= 0, position[codes], -1)
            values = values[keep].reset_index(drop=True)
        counts = np.bincount(codes[codes >= 0], minlength=len(values))
        return cls(series.index, codes, values, counts)

    @property
    def rows(self) -> int:
        """Rows with a value."""
        return int(self.counts.sum())

    def rows_where(self, mask: pd.Series) -> int:
        """Rows whose value satisfies a mask computed on ``values``."""
        return int(self.counts[mask.to_numpy(dtype=bool)].sum())

    def share(self, mask: pd.Series) -> float:
        """Share of rows with a value whose value satisfies ``mask``."""
        return self.rows_where(mask) / self.rows

    def expand(self, converted: pd.Series) -> pd.Series:
        """Map one result per distinct value back onto the rows; missing rows become NA."""
        return pd.Series(converted.array.take(self.codes, allow_fill=True), index=self.index)


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _keywords_to_avoid() -> frozenset[str]:
    """DuckDB keywords that cannot be used as bare column names."""
    with duckdb.connect(":memory:") as con:
        rows = con.execute(
            "SELECT keyword_name FROM duckdb_keywords() "
            "WHERE keyword_category IN ('reserved', 'type_function')"
        ).fetchall()
    return frozenset(r[0] for r in rows)


def to_identifier(name: str) -> str:
    """Turn a display name into a safe, lower-case SQL identifier.

    Accents are folded (``Māori`` becomes ``maori``), symbols become underscores,
    a leading digit gets a ``c_`` prefix, a name with no usable characters becomes
    ``col``, and DuckDB keywords get a ``_col`` suffix. ``Rainfall (mm)`` becomes
    ``rainfall_mm``; ``°C`` becomes ``c``.
    """
    folded = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(ch for ch in folded if not unicodedata.combining(ch))
    ascii_name = ascii_name.encode("ascii", "ignore").decode("ascii").lower()
    ident = re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")
    if not ident:
        ident = "col"
    if ident[0].isdigit():
        ident = f"c_{ident}"
    if ident in _keywords_to_avoid():
        ident = f"{ident}_col"
    return ident[:MAX_IDENTIFIER_LENGTH].rstrip("_")


def make_identifiers(display_names: list[str]) -> list[str]:
    """Unique SQL identifiers for a list of display names, in the same order."""
    result: list[str] = []
    used: set[str] = set()
    for name in display_names:
        base = to_identifier(name)
        candidate, n = base, 1
        while candidate in used:
            n += 1
            suffix = f"_{n}"
            candidate = base[: MAX_IDENTIFIER_LENGTH - len(suffix)] + suffix
        used.add(candidate)
        result.append(candidate)
    return result


# ---------------------------------------------------------------------------
# Per-type inference
# ---------------------------------------------------------------------------


def _try_boolean(col: _Distinct) -> _Inferred | None:
    lowered = col.values.str.lower()
    if not lowered.isin(BOOLEAN_VALUES).all():
        return None
    mapped = lowered.map(BOOLEAN_VALUES).astype("boolean")
    return _Inferred("boolean", col.expand(mapped), note=ColumnNote("true/false values"))


def _clean_numbers(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Strip number decoration. Returns (cleaned text, negative-in-parentheses mask)."""
    negative = values.str.fullmatch(_PARENS)
    text = values.where(~negative, values.str.slice(1, -1))
    text = text.str.replace(_CURRENCY, "", regex=True).str.strip()
    text = text.str.removesuffix("%").str.strip().str.removeprefix("+")
    thousands = text.str.fullmatch(_THOUSANDS)
    text = text.where(~thousands, text.str.replace(",", "", regex=False))
    return text, negative


def _to_float(text: pd.Series, negative: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(text, errors="coerce").astype("Float64")
    return numbers.where(~negative, -numbers)


def _to_int(text: pd.Series, negative: pd.Series) -> pd.Series | None:
    """Exact integer conversion, or ``None`` if a value does not fit in 64 bits."""
    try:
        numbers = text.astype("Int64")
    except (ValueError, TypeError, OverflowError):
        return None
    return numbers.where(~negative, -numbers)


def _try_number(col: _Distinct) -> _Inferred | None:
    values = col.values
    if values.str.fullmatch(_LEADING_ZERO_CODE).any():
        return None
    text, negative = _clean_numbers(values)
    parsed = _to_float(text, negative)
    ok = parsed.notna()
    if col.share(ok) < PARSE_THRESHOLD:
        return None
    if values[~ok].str.fullmatch(_CENSORED).any():
        return None  # kept as censored text by the caller

    flags: list[ColumnFlag] = []
    coerced = col.rows - col.rows_where(ok)
    if coerced:
        flags.append("mixed_types")
    if values.str.contains(_NUMERIC_DECOR, regex=True).any():
        flags.append("numeric_text")
    percent = (
        " (percent signs removed: 12% is stored as 12)" if values.str.endswith("%").any() else ""
    )

    if text[ok].str.fullmatch(_INTEGER).all():
        ints = _to_int(text.where(ok), negative)
        if ints is None:
            return None  # too long for a number; such values are identifiers
        note = ColumnNote(f"whole numbers{percent}", coerced)
        return _Inferred("integer", col.expand(ints), tuple(flags), note)
    note = ColumnNote(f"decimal numbers{percent}", coerced)
    return _Inferred("decimal", col.expand(parsed), tuple(flags), note)


def _date_formats(sample: pd.Series, day_first_default: bool) -> _DateLayout | None:
    """Pick parse formats for a sample of date text, or ``None`` if it is not dates."""
    n = len(sample)
    if sample.str.fullmatch(_ISO).sum() / n >= PARSE_THRESHOLD:
        return _DateLayout(("ISO8601",), "ISO dates (yyyy-mm-dd)")
    if sample.str.fullmatch(_YEAR_MONTH).sum() / n >= PARSE_THRESHOLD:
        return _DateLayout(("%Y-%m",), "year and month (yyyy-mm), first day of each month")

    parts = sample.astype(object).str.extract(_NUMERIC_DATE)
    matched = parts["a"].notna()
    if matched.sum() / n >= PARSE_THRESHOLD:
        seps = parts.loc[matched, "sep"].unique()
        if len(seps) != 1:
            return None
        a_big = bool((parts.loc[matched, "a"].astype(int) > 12).any())
        b_big = bool((parts.loc[matched, "b"].astype(int) > 12).any())
        if a_big and b_big:
            return None
        if a_big or b_big:
            return _numeric_layout(parts, matched, seps[0], day_first=a_big, ambiguous=False)
        return _numeric_layout(parts, matched, seps[0], day_first_default, ambiguous=True)

    named = sample.str.contains(_MONTH_NAME) & sample.str.contains(_FOUR_DIGIT_YEAR)
    if named.sum() / n >= PARSE_THRESHOLD:
        return _DateLayout(("mixed",), "dates with month names", day_first=day_first_default)
    return None


def _numeric_layout(
    parts: pd.DataFrame, matched: pd.Series, sep: str, day_first: bool, *, ambiguous: bool
) -> _DateLayout:
    year_sizes = sorted(parts.loc[matched, "y"].str.len().unique(), reverse=True)
    order = ("%d", "%m") if day_first else ("%m", "%d")
    formats = tuple(
        f"{order[0]}{sep}{order[1]}{sep}{'%Y' if size == 4 else '%y'}{time}"
        for size in year_sizes
        for time in _TIME_FORMATS
    )
    layout = ("dd/mm/yyyy" if day_first else "mm/dd/yyyy").replace("/", sep)
    reason = "assumed from the region default" if ambiguous else "confirmed by the values"
    return _DateLayout(formats, f"{layout}, order {reason}", day_first, ambiguous)


def _swapped(layout: _DateLayout) -> _DateLayout:
    """The same numeric layout with day and month exchanged, as settled by the values."""
    swap = {"%d": "%m", "%m": "%d"}
    formats = tuple(re.sub(r"%[dm]", lambda m: swap[m.group()], f) for f in layout.formats)
    sep = formats[0][2]
    text = ("mm/dd/yyyy" if layout.day_first else "dd/mm/yyyy").replace("/", sep)
    return _DateLayout(formats, f"{text}, order confirmed by the values", not layout.day_first)


def _parse_dates(values: pd.Series, layout: _DateLayout) -> pd.Series:
    """Parse with each format in turn, filling only values still unparsed.

    Times with a UTC offset are converted to UTC and stored without a zone.
    """
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[us]")
    for fmt in layout.formats:
        remaining = result.isna()
        if not remaining.any():
            break
        kwargs: dict[str, bool] = {}
        if fmt == "mixed":
            kwargs["dayfirst"] = layout.day_first
        if fmt == "ISO8601":
            kwargs["utc"] = True
        parsed = pd.to_datetime(values[remaining], format=fmt, errors="coerce", **kwargs)
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        result.loc[remaining] = parsed.astype("datetime64[us]")
    return result


def _try_date(col: _Distinct, day_first_default: bool) -> _Inferred | None:
    layout = _date_formats(col.values.head(DATE_SAMPLE_SIZE), day_first_default)
    if layout is None:
        return None
    parsed = _parse_dates(col.values, layout)
    if layout.ambiguous and parsed.isna().any():
        # A value beyond the sample can settle the order (for example 01/13/2024).
        other = _swapped(layout)
        reparsed = _parse_dates(col.values, other)
        if col.rows_where(reparsed.notna()) > col.rows_where(parsed.notna()):
            layout, parsed = other, reparsed
    ok = parsed.notna()
    if col.share(ok) < PARSE_THRESHOLD:
        return None

    flags: list[ColumnFlag] = []
    if layout.ambiguous:
        flags.append("ambiguous_date")
    coerced = col.rows - col.rows_where(ok)
    if coerced:
        flags.append("mixed_types")
    valid = parsed[ok]
    has_time = bool((valid != valid.dt.normalize()).any())
    return _Inferred(
        "timestamp" if has_time else "date",
        col.expand(parsed),
        tuple(flags),
        ColumnNote(layout.description, coerced),
    )


def _as_text(series: pd.Series, col: _Distinct) -> _Inferred:
    text = series.str.strip().where(lambda s: s != "")
    censored = col.values.str.fullmatch(_CENSORED)
    if censored.any():
        count = col.rows_where(censored)
        detail = f"text: {count:,} censored values such as '<0.5' kept as written"
        return _Inferred("text", text, ("censored_values",), ColumnNote(detail))
    if col.values.str.fullmatch(_LEADING_ZERO_CODE).any():
        return _Inferred("text", text, (), ColumnNote("text: codes with leading zeros"))
    cleaned, negative = _clean_numbers(col.values)
    numeric_share = col.share(_to_float(cleaned, negative).notna())
    if 0.5 <= numeric_share < 1:
        detail = f"text: {numeric_share:.0%} of values are numbers"
        return _Inferred("text", text, ("mixed_types",), ColumnNote(detail))
    return _Inferred("text", text, (), ColumnNote("text"))


def infer_column(series: pd.Series, *, day_first_default: bool = True) -> _Inferred:
    """Infer the type of one text column and convert it."""
    col = _Distinct.of(series.astype("string"))
    if col.values.empty:
        missing = pd.Series(pd.NA, index=series.index, dtype="string")
        return _Inferred("text", missing, ("all_missing",), ColumnNote("no values"))

    inferred = (
        _try_boolean(col)
        or _try_number(col)
        or _try_date(col, day_first_default)
        or _as_text(series.astype("string"), col)
    )
    if len(col.values) == 1 and col.rows > 1:
        inferred = replace(inferred, flags=(*inferred.flags, "constant"))
    return inferred


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def normalize(raw: RawTable, *, settings: Settings | None = None) -> TypedTable:
    """Infer types for every column of ``raw`` and rename columns to SQL identifiers."""
    settings = settings or get_settings()
    day_first = get_region(settings.default_region).day_first
    display_names = [str(c) for c in raw.dataframe.columns]
    identifiers = make_identifiers(display_names)

    data: dict[str, pd.Series] = {}
    schemas: list[ColumnSchema] = []
    notes: dict[str, ColumnNote] = {}
    for display, ident in zip(display_names, identifiers, strict=True):
        inferred = infer_column(raw.dataframe[display], day_first_default=day_first)
        data[ident] = inferred.series.reset_index(drop=True)
        schemas.append(
            ColumnSchema(
                sql_name=ident, display_name=display, dtype=inferred.dtype, flags=inferred.flags
            )
        )
        notes[ident] = inferred.note

    table = pd.DataFrame(data)
    log.info(
        "table_typed",
        extra={
            "columns": len(schemas),
            "types": {t: sum(s.dtype == t for s in schemas) for t in {s.dtype for s in schemas}},
            "flagged": sum(bool(s.flags) for s in schemas),
        },
    )
    return TypedTable(
        dataframe=table,
        columns=tuple(schemas),
        source_name=raw.source_name,
        sheet_name=raw.sheet_name,
        notes=notes,
    )
