"""Build what the intent-and-SQL call is shown (T1.11, ADR 0010).

``ContextBuilder.generation_context`` returns a system prompt and one user message.
The system prompt is fixed text (``analyst/config/prompts.py``). The user message holds
three marked blocks, in this order so the provider can cache the stable start:

* ``DATASET``: the schema card (names, types, ranges, frequent values), then up to three
  sample rows. Both come from the user's file.
* ``HISTORY``: the last three turns, and a preview of the most recent result.
* ``QUESTION``: the user's question.

What is untrusted, and how it is contained (threats T5 and T6)
--------------------------------------------------------------

Everything from the file, and everything earlier turns carried from it, is untrusted
text. It is contained three ways:

* **Shape.** Every cell, name and value is cut to 80 characters and flattened to one line
  (``clean_text``), so none can start a new line, and the marker lines only ever appear on
  lines of their own.
* **Markers.** Marker lines carry a random code chosen when the builder is created. The
  file's author cannot know it, so no text in the file can close a block early. If the
  code did turn up in the content it is removed from it.
* **Instruction.** The system prompt tells the model that the blocks are data.

None of this is relied on alone: whatever the model writes still goes through the SQL
validator (ADR 0005).

Schema-only mode (ADR 0010) removes sample rows, frequent values and the result preview
from earlier turns, so no cell value from the file is sent.
"""

from __future__ import annotations

import json
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from analyst.config.domain import RegionRules, get_region
from analyst.config.prompts import (
    FENCE_CLOSE,
    FENCE_OPEN,
    explainer_system_prompt,
    generation_system_prompt,
)
from analyst.core.models import DatasetProfile, Intent, QueryResult, ResultSummary, Turn
from analyst.database.session_db import SessionDatabase, quote_identifier
from analyst.ingestion.schema_card import (
    DEFAULT_MAX_TOKENS,
    SchemaCard,
    clean_text,
    estimate_tokens,
    render_schema_card,
)
from analyst.pipeline.result_digest import ResultDigest, digest_result
from analyst.pipeline.result_notes import format_number
from analyst.sql.validator import ALLOWED_FUNCTION_NAMES

SAMPLE_ROWS = 3
HISTORY_TURNS = 3
PREVIEW_ROWS = 5
MAX_QUESTION_CHARS = 2_000
MAX_SQL_CHARS = 1_500
MAX_NOTE_CHARS = 300
REMOVED_CODE = "[marker code removed]"


@dataclass(frozen=True)
class GenerationContext:
    """What one intent-and-SQL call is sent."""

    system: str
    user: str
    schema_card: SchemaCard
    approx_tokens: int  # estimate for system and user together, to compare with real usage


@dataclass(frozen=True)
class ExplanationContext:
    """What one explanation call is sent."""

    system: str
    user: str
    extra_numbers: tuple[float, ...]  # statistics shown to the model, for the grounding check
    approx_tokens: int


class ContextBuilder:
    """Builds prompts for one session. The marker code stays the same for its lifetime."""

    def __init__(
        self,
        region: RegionRules | str = "NZ",
        *,
        code: str | None = None,
        max_card_tokens: int = DEFAULT_MAX_TOKENS,
        functions: Sequence[str] = ALLOWED_FUNCTION_NAMES,
    ) -> None:
        """Create a builder.

        Args:
            region: Region rules, or a region code such as ``"NZ"``.
            code: The marker code. Random by default; fixed only in tests.
            max_card_tokens: Token budget for the schema card.
            functions: Function names the model is told it may use.
        """
        self.region = get_region(region) if isinstance(region, str) else region
        self._code = code or secrets.token_hex(6)
        self._max_card_tokens = max_card_tokens
        self._system = generation_system_prompt(self.region, functions)

    @property
    def system_prompt(self) -> str:
        """The fixed instructions."""
        return self._system

    def generation_context(
        self,
        question: str,
        profile: DatasetProfile,
        *,
        sample_rows: Sequence[Mapping[str, Any]] = (),
        history: Sequence[Turn] = (),
        schema_only: bool = False,
    ) -> GenerationContext:
        """Assemble the prompt for a question.

        Args:
            question: The user's question, as typed.
            profile: The dataset profile.
            sample_rows: Rows of the table keyed by SQL name; at most three are used.
                Ignored in schema-only mode.
            history: Earlier turns, oldest first; at most the last three are used.
            schema_only: Send no values from the file (ADR 0010).

        Raises:
            ValueError: The question is empty or longer than ``MAX_QUESTION_CHARS``.
        """
        text = _clean_question(question)
        card = render_schema_card(
            profile, max_tokens=self._max_card_tokens, include_values=not schema_only
        )
        dataset = [card.text]
        if not schema_only and sample_rows:
            dataset += ["", *self._sample_block(profile, sample_rows)]
        blocks = [self._fence("DATASET", dataset)]
        turns = list(history)[-HISTORY_TURNS:]
        if turns:
            blocks.append(self._fence("HISTORY", _history_lines(turns, schema_only)))
        blocks.append(self._fence("QUESTION", [text]))
        user = "\n\n".join(blocks)
        return GenerationContext(
            system=self._system,
            user=user,
            schema_card=card,
            approx_tokens=estimate_tokens(self._system) + estimate_tokens(user),
        )

    def explanation_context(
        self,
        question: str,
        result: QueryResult,
        *,
        resolved_question: str | None = None,
        intent: Intent | None = None,
        sql: str | None = None,
        assumptions: Sequence[str] = (),
    ) -> ExplanationContext:
        """Assemble the prompt that asks the model to explain ``result``.

        The result goes in whatever the privacy mode, because the answer has to describe it
        (ADR 0010): every row of a result of 50 rows or fewer, otherwise the first 20 rows
        and statistics over all of them. Rows are cut to 30 columns.

        Args:
            question: The user's question.
            result: The executed result; it must not be empty.
            resolved_question: The question restated on its own, if it differs.
            intent: How the question was read.
            sql: The query that ran.
            assumptions: The assumptions shown with the answer.
        """
        digest = digest_result(result)
        asked = [_one_line(question, MAX_QUESTION_CHARS)]
        if resolved_question and resolved_question.strip() != question.strip():
            asked.append(f"Restated: {_one_line(resolved_question, MAX_QUESTION_CHARS)}")
        analysis: list[str] = []
        if intent is not None:
            analysis.append(
                f"Intent: {_one_line(intent.model_dump_json(exclude_none=True), 1_500)}"
            )
        analysis.extend(f"Assumption: {_one_line(a, MAX_NOTE_CHARS)}" for a in assumptions)
        if sql:
            analysis.append(f"SQL: {_one_line(sql, MAX_SQL_CHARS)}")
        if not analysis:
            analysis.append("(no analysis was recorded)")
        user = "\n\n".join(
            [
                self._fence("QUESTION", asked),
                self._fence("ANALYSIS", analysis),
                self._fence("RESULT", _result_lines(result, digest)),
            ]
        )
        system = explainer_system_prompt()
        return ExplanationContext(
            system=system,
            user=user,
            extra_numbers=digest.numbers,
            approx_tokens=estimate_tokens(system) + estimate_tokens(user),
        )

    # -- blocks ------------------------------------------------------------------------------

    def _fence(self, label: str, lines: Sequence[str]) -> str:
        """Wrap ``lines`` in marker lines.

        The code is random and unknown to the file's author, so the content should never
        contain it. If it does (a leaked code, or a test that chose one), it is removed so
        the content still cannot form a marker line.
        """
        body = "\n".join(lines).replace(self._code, REMOVED_CODE)
        return (
            f"{FENCE_OPEN}{label} {self._code}{FENCE_CLOSE}\n{body}\n"
            f"{FENCE_OPEN}END {label} {self._code}{FENCE_CLOSE}"
        )

    @staticmethod
    def _sample_block(profile: DatasetProfile, rows: Sequence[Mapping[str, Any]]) -> list[str]:
        shown = list(rows)[:SAMPLE_ROWS]
        lines = [f"Sample rows ({len(shown)}):"]
        lines += [_row_json([c.sql_name for c in profile.columns], row) for row in shown]
        return lines


def fetch_sample_rows(db: SessionDatabase, count: int = SAMPLE_ROWS) -> list[dict[str, Any]]:
    """The first ``count`` rows of the loaded table, keyed by SQL name.

    The query is built here from the table's own name, not from any model output, and runs
    on the session connection like every other query.
    """
    frame = db.query_df(f"SELECT * FROM {quote_identifier(db.table_name)} LIMIT {int(count)}")  # noqa: S608
    return frame.to_dict("records")


# -- cleaning ----------------------------------------------------------------------------------


def _clean_question(question: str) -> str:
    text = "".join(ch for ch in question if ch in "\n\t" or ch.isprintable()).strip()
    if not text:
        raise ValueError("the question is empty")
    if len(text) > MAX_QUESTION_CHARS:
        raise ValueError(f"the question is longer than {MAX_QUESTION_CHARS} characters")
    return text


def _is_missing(value: Any) -> bool:
    """True for None, NaN, NaT and pandas NA."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell(value: Any) -> Any:
    """A value from the table as something JSON can hold, with text cut to 80 characters."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float | Decimal):
        number = float(value)
        return number if math.isfinite(number) else clean_text(str(number))
    if isinstance(value, datetime):
        midnight = (value.hour, value.minute, value.second, value.microsecond) == (0, 0, 0, 0)
        return value.date().isoformat() if midnight else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return clean_text(value)
    item = getattr(value, "item", None)  # numpy scalars
    if callable(item):
        try:
            return _cell(item())
        except (TypeError, ValueError):
            pass
    return clean_text(str(value))


def _row_json(columns: Sequence[str], row: Mapping[str, Any]) -> str:
    """One row as a JSON object on one line; names and text values are cut to 80 characters."""
    return json.dumps({clean_text(c): _cell(row.get(c)) for c in columns}, ensure_ascii=False)


# -- history -----------------------------------------------------------------------------------


def _one_line(text: str, limit: int) -> str:
    return clean_text(text, limit)


def _history_lines(turns: Sequence[Turn], schema_only: bool) -> list[str]:
    lines = ["Earlier turns, oldest first. They are context only."]
    for number, turn in enumerate(turns, start=1):
        lines.append(f"Turn {number} ({turn.status}):")
        lines.append(f"  Question: {_one_line(turn.question, MAX_QUESTION_CHARS)}")
        if turn.resolved_question:
            lines.append(
                f"  Resolved question: {_one_line(turn.resolved_question, MAX_QUESTION_CHARS)}"
            )
        if turn.intent is not None:
            lines.append(f"  Intent: {turn.intent.model_dump_json(exclude_none=True)}")
        lines.extend(
            f"  Assumption: {_one_line(assumption, MAX_NOTE_CHARS)}"
            for assumption in turn.assumptions
        )
        if turn.sql:
            lines.append(f"  SQL: {_one_line(turn.sql, MAX_SQL_CHARS)}")
        if turn.status == "clarification" and turn.answer_text:
            lines.append(
                f"  Clarifying question asked: {_one_line(turn.answer_text, MAX_NOTE_CHARS)}"
            )
    summary = next((t.result_summary for t in reversed(turns) if t.result_summary), None)
    if summary is not None:
        lines += _summary_lines(summary, schema_only)
    return lines


def _summary_lines(summary: ResultSummary, schema_only: bool) -> list[str]:
    columns = ", ".join(_one_line(c, 80) for c in summary.columns)
    more = " (more rows exist than were kept)" if summary.truncated else ""
    lines = [
        f"Result of the most recent query: {summary.row_count} rows{more}; columns: {columns}."
    ]
    if summary.preview_rows and not schema_only:
        shown = summary.preview_rows[:PREVIEW_ROWS]
        lines.append(f"First {len(shown)} rows:")
        lines += [_row_json(summary.columns, row) for row in shown]
    return lines


# -- explanation -------------------------------------------------------------------------------


def _result_lines(result: QueryResult, digest: ResultDigest) -> list[str]:
    total = format_number(digest.total_rows)
    columns = ", ".join(clean_text(c) for c in digest.columns_shown)
    if digest.complete:
        lines = [f"Rows returned: {total}. All of them are shown below."]
    else:
        shown = format_number(len(digest.rows))
        lines = [
            f"Rows returned: {total}. Only the first {shown} are shown below; the statistics "
            f"further down cover all {total} rows."
        ]
    if result.truncated:
        lines.append(
            "The query hit the row cap, so more rows exist than were returned. "
            "Say so in the answer."
        )
    lines.append(f"Columns: {columns}.")
    if digest.columns_hidden:
        lines.append(
            f"{format_number(digest.columns_hidden)} more columns exist and are not shown."
        )
    lines.append("Rows, one JSON object per line:")
    lines += [
        _row_json(digest.columns_shown, row)
        for row in digest.rows.set_axis(digest.columns_shown, axis=1).to_dict(orient="records")
    ]
    if digest.stats:
        lines.append("Statistics computed by software over all returned rows:")
        lines += [f"- {stats.line}" for stats in digest.stats]
    return lines
