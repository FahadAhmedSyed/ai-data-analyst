"""Typed contracts passed between pipeline stages (architecture.md, section 4).

Two kinds of model live here:

* **LLM output models** (``Intent``, ``GenerationResult``, ``ModelExplanation`` and
  their parts) are sent to OpenAI as structured-output schemas. They must stay
  compatible with OpenAI's strict JSON schema mode: no free-form dicts, no
  open-ended ``Any``. ``tests/unit/test_models.py`` enforces this.
* **Internal models** carry data between stages inside the process and may hold
  richer types such as pandas DataFrames.

Model output is untrusted. These models only check shape; code in later stages
checks meaning (for example, that every column in an intent exists).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

# ---------------------------------------------------------------------------
# Shared literals
# ---------------------------------------------------------------------------

ColumnType = Literal["integer", "decimal", "text", "boolean", "date", "timestamp"]

ColumnFlag = Literal[
    "ambiguous_date",  # day and month order could not be determined
    "censored_values",  # values such as "<0.5" or "ND" are present
    "mixed_types",  # kept as text because too few values parsed as one type
    "numeric_text",  # parsed from text with separators, currency or percent signs
    "all_missing",  # every value is missing
    "constant",  # a single distinct value
]

Aggregation = Literal["sum", "avg", "min", "max", "count", "count_distinct", "median", "none"]

FilterOperator = Literal[
    "=",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not_in",
    "between",
    "is_null",
    "is_not_null",
    "contains",
    "month_in",
    "year_in",
]

Operation = Literal[
    "aggregate", "count", "list", "rank", "trend", "compare", "distribution", "other"
]

OutputShape = Literal["scalar", "table", "time_series", "distribution"]

ChartType = Literal["line", "bar", "scatter", "histogram", "none"]

ResponseStatus = Literal["answered", "clarification", "failed"]

ValidationStatus = Literal["ok", "correctable", "rejected"]

QualityKind = Literal[
    "duplicate_rows",
    "all_missing",
    "mixed_types",
    "censored_values",
    "ambiguous_date",
    "missing_values",
    "constant",
]

ValidationCategory = Literal[
    "ok",
    "parse_error",
    "multiple_statements",
    "forbidden_statement",
    "unknown_table",
    "forbidden_table",
    "table_function",
    "disallowed_function",
    "unknown_column",
    "too_complex",
]


class _Strict(BaseModel):
    """Base for LLM output models: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class _Internal(BaseModel):
    """Base for internal models: immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Dataset schema and profile (ingestion and profiler)
# ---------------------------------------------------------------------------


class ColumnSchema(_Internal):
    """A column as loaded: its safe SQL name, the user's original name, and its type."""

    sql_name: str = Field(min_length=1, max_length=63)
    display_name: str = Field(min_length=1)
    dtype: ColumnType
    flags: tuple[ColumnFlag, ...] = ()


class ValueCount(_Internal):
    """One frequent value of a column and how often it occurs."""

    value: str
    count: NonNegativeInt


class ColumnProfile(ColumnSchema):
    """A column plus the statistics computed by the profiler."""

    null_count: NonNegativeInt
    distinct_count: NonNegativeInt
    min_value: float | int | str | None = None
    max_value: float | int | str | None = None
    mean: float | None = None
    std: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    top_values: tuple[ValueCount, ...] = ()
    date_min: date | datetime | None = None
    date_max: date | datetime | None = None

    def null_fraction(self, row_count: int) -> float:
        """Share of missing values, between 0 and 1."""
        return self.null_count / row_count if row_count else 0.0


class QualityIssue(_Internal):
    """One finding in the data quality summary. The message never contains cell values."""

    kind: QualityKind
    column: str | None = Field(default=None, description="SQL name; None for whole-table findings")
    count: NonNegativeInt = Field(default=0, description="Rows or values affected, where counted")
    message: str


class DatasetProfile(_Internal):
    """Everything known about the loaded table, used by the UI and the prompts."""

    table_name: str = Field(min_length=1, max_length=63)
    source_name: str
    sheet_name: str | None = None
    row_count: NonNegativeInt
    column_count: NonNegativeInt
    columns: tuple[ColumnProfile, ...]
    duplicate_rows: NonNegativeInt = 0
    primary_date_column: str | None = None
    latest_date: date | datetime | None = None

    def column(self, sql_name: str) -> ColumnProfile | None:
        """Return a column by its SQL name, or ``None``."""
        return next((c for c in self.columns if c.sql_name == sql_name), None)

    @property
    def column_names(self) -> frozenset[str]:
        """SQL names of all columns."""
        return frozenset(c.sql_name for c in self.columns)

    @property
    def display_names(self) -> dict[str, str]:
        """Map from SQL name to the user's original column name."""
        return {c.sql_name: c.display_name for c in self.columns}


# ---------------------------------------------------------------------------
# LLM output: intent and generation (ADR 0006)
# ---------------------------------------------------------------------------


class Measure(_Strict):
    """A column and how it is aggregated."""

    column: str
    aggregation: Aggregation


class FilterCondition(_Strict):
    """One filter. ``values`` holds zero, one or several operands depending on the operator."""

    column: str
    operator: FilterOperator
    values: list[str | float]


class TimeRange(_Strict):
    """A resolved time window, ISO dates, plus how it was derived."""

    start: str | None = Field(description="Inclusive ISO date, or null if open")
    end: str | None = Field(description="Inclusive ISO date, or null if open")
    description: str = Field(description="How the range was derived, in plain words")


class SortKey(_Strict):
    """One sort key. ``column`` may be an output alias as well as a source column."""

    column: str
    direction: Literal["asc", "desc"]


class Intent(_Strict):
    """The structured meaning of a question, stated before any SQL is written."""

    operation: Operation
    measures: list[Measure]
    dimensions: list[str]
    filters: list[FilterCondition]
    time_range: TimeRange | None
    sort: list[SortKey]
    limit: int | None
    output: OutputShape

    def source_columns(self) -> set[str]:
        """Columns that must exist in the dataset (sort keys may be aliases, so are excluded)."""
        cols = {m.column for m in self.measures if m.column != "*"}
        cols.update(self.dimensions)
        cols.update(f.column for f in self.filters)
        return cols


class Ambiguity(_Strict):
    """A term in the question that could be read more than one way."""

    term: str
    readings: list[str]
    changes_result: bool = Field(description="True if the readings give different answers")


class ChartSpec(_Strict):
    """A suggested chart. Used only if its columns exist in the result and fit the type."""

    type: ChartType
    x: str | None
    y: list[str]
    color: str | None


class GenerationResult(_Strict):
    """Output of LLM call 1. Field order matters: intent comes before SQL."""

    resolved_question: str = Field(description="The question rewritten to stand alone")
    intent: Intent
    assumptions: list[str]
    ambiguities: list[Ambiguity]
    needs_clarification: bool
    clarification_question: str | None
    sql: str | None
    explanation: str = Field(description="What the SQL does, in one or two sentences")
    chart: ChartSpec | None

    def problems(self) -> list[str]:
        """Inconsistencies the generator must handle; an empty list means usable."""
        issues: list[str] = []
        if self.needs_clarification and not (self.clarification_question or "").strip():
            issues.append("needs_clarification is true but no clarification_question was given")
        if not self.needs_clarification and not (self.sql or "").strip():
            issues.append("no SQL was given and no clarification was requested")
        return issues


# ---------------------------------------------------------------------------
# LLM output: explanation (ADR 0017)
# ---------------------------------------------------------------------------


class KeyNumber(_Strict):
    """A number the explanation relies on, and the result column it came from."""

    value: float
    column: str
    label: str


class ModelExplanation(_Strict):
    """Output of LLM call 2, before the grounding check."""

    answer_markdown: str
    key_numbers: list[KeyNumber]
    caveats: list[str]


# ---------------------------------------------------------------------------
# Validation and execution
# ---------------------------------------------------------------------------


class ValidationResult(_Internal):
    """Verdict of the SQL validator (security.md, section 4)."""

    status: ValidationStatus
    category: ValidationCategory
    reason: str = ""
    safe_sql: str | None = Field(
        default=None, description="Query wrapped with the row cap; set only when status is ok"
    )

    @property
    def ok(self) -> bool:
        """True when the query may be executed."""
        return self.status == "ok"


class QueryResult(BaseModel):
    """A successful query. Failures raise ``QueryExecutionError`` instead."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    dataframe: pd.DataFrame
    row_count: NonNegativeInt
    truncated: bool = False
    duration_ms: NonNegativeFloat = 0.0

    @property
    def is_empty(self) -> bool:
        """True when the query returned no rows."""
        return self.row_count == 0

    def summary(self, max_rows: int = 20) -> ResultSummary:
        """Compact, serialisable summary kept in conversation memory."""
        preview = self.dataframe.head(max_rows)
        rows = [
            {str(k): _jsonable(v) for k, v in record.items()}
            for record in preview.to_dict(orient="records")
        ]
        return ResultSummary(
            columns=tuple(str(c) for c in self.dataframe.columns),
            row_count=self.row_count,
            truncated=self.truncated,
            preview_rows=tuple(rows),
        )


def _jsonable(value: Any) -> Any:
    """Convert pandas and numpy scalars to plain Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Conversation and response
# ---------------------------------------------------------------------------


class ResultSummary(_Internal):
    """What memory keeps of a result: its shape and a small preview."""

    columns: tuple[str, ...]
    row_count: NonNegativeInt
    truncated: bool = False
    preview_rows: tuple[dict[str, Any], ...] = ()


class Usage(_Internal):
    """LLM usage and timing for one question."""

    llm_calls: NonNegativeInt = 0
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFloat = 0.0
    latency_ms: NonNegativeFloat = 0.0

    def __add__(self, other: Usage) -> Usage:
        """Combine usage from several calls."""
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
        )


class Turn(_Internal):
    """One question and its outcome, as stored in conversation memory."""

    question: str
    status: ResponseStatus
    resolved_question: str | None = None
    intent: Intent | None = None
    sql: str | None = None
    assumptions: tuple[str, ...] = ()
    answer_text: str = ""
    result_summary: ResultSummary | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AnalystResponse(BaseModel):
    """Everything the UI shows for one question."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    status: ResponseStatus
    question: str
    message: str = Field(description="The answer, the clarifying question, or the failure message")
    resolved_question: str | None = None
    assumptions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    sql: str | None = None
    result: QueryResult | None = None
    chart: ChartSpec | None = None
    grounded: bool | None = None
    fallback_used: bool = False
    usage: Usage = Field(default_factory=Usage)
    turn: Turn
