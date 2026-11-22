"""Ask the model to fix a query that could not be used (T1.13, architecture section 7).

``Corrector.correct`` makes one LLM call on the conversation the generator started: the
model is shown its own answer again, then a ``PROBLEM`` block saying why the query could not
be used, then a one-line request that depends on the kind of problem. It answers in the same
format as the first call (``GenerationResult``), so a correction can change the intent and
the assumptions as well as the SQL when the problem shows they were wrong.

The functions that build a ``Problem`` decide what the model is told:

* ``validation_problem``: the validator's own reason, which names identifiers the model wrote
  and the closest real ones, and never a value from the data. Only a *correctable* verdict
  can be sent; a rejected one (a query that changes data or reads files) is never retried.
* ``execution_problem``: the database's message, cleaned, and for type problems the types of
  the columns the query uses. The message can quote values from the file, so it is fenced
  like the rest of the file's text (see ``ContextBuilder.correction_message``), and in
  schema-only mode (ADR 0010) quoted text is removed and, for the categories that can echo a
  cell, the message is withheld altogether.
* ``intent_problem``: the intent names columns that are not in the table.

Nothing here runs SQL or decides how many corrections are allowed; ``QueryLoop`` does that.
Neither the SQL nor the database's message is ever logged (threat T10).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from analyst.config.prompts import correction_request, correction_title
from analyst.core.errors import QueryExecutionError
from analyst.core.logging import get_logger
from analyst.core.models import DatasetProfile, GenerationResult, Usage, ValidationResult
from analyst.ingestion.schema_card import clean_text
from analyst.llm.client import CallBudget, LLMClient, Message
from analyst.pipeline.generator import MAX_MESSAGE_CHARS, Generation, clean_result
from analyst.sql.context import ContextBuilder

log = get_logger(__name__)

MAX_ENGINE_CHARS = 400
MAX_TYPED_COLUMNS = 12

ProblemStage = Literal["intent", "validation", "execution"]

# A query that was too heavy is corrected once, however it failed: a second try would only
# make the user wait for the same failure.
HEAVY = frozenset({"timeout", "out_of_memory"})
# Categories whose database message can quote a value from the file, and so is not sent in
# schema-only mode.
_MAY_QUOTE_VALUES = frozenset({"conversion", "other"})
# Categories for which the types of the columns the query uses help the model.
_NEEDS_TYPES = frozenset({"binder", "conversion"})
_QUOTED = re.compile(r"'[^']*'")
_NO_SQL_MESSAGE = "The question could not be answered with this table."


@dataclass(frozen=True)
class Problem:
    """Why a query could not be used, as the model is told it."""

    stage: ProblemStage
    category: str  # a validator or executor category, or ``intent_columns``
    details: tuple[str, ...]  # what the checker or database said, one line each

    @property
    def title(self) -> str:
        """One sentence naming the problem."""
        return correction_title(self.category)

    @property
    def request(self) -> str:
        """What the model is asked to do about it."""
        return correction_request(self.category)

    @property
    def heavy(self) -> bool:
        """True when the query was too slow or too big, which is corrected only once."""
        return self.category in HEAVY


@dataclass(frozen=True)
class Correction:
    """The model's answer to one correction request."""

    result: GenerationResult  # text for the user already cleaned; SQL as written
    sql: str | None  # None when the model gave no query
    message: str  # why there is no query (its question or explanation); empty with SQL
    messages: tuple[Message, ...]  # the conversation sent, to continue from
    usage: Usage


# -- what the model is told ----------------------------------------------------------------------


def intent_problem(generation: Generation) -> Problem | None:
    """The problem with a generation whose intent names columns that do not exist, or None."""
    text = generation.intent_problem
    return None if text is None else Problem("intent", "intent_columns", (text,))


def validation_problem(validation: ValidationResult) -> Problem:
    """The problem with a query the validator found fixable.

    Raises:
        ValueError: The verdict is not ``correctable``. A rejected query is never sent back.
    """
    if validation.status != "correctable":
        msg = "only a correctable validation result can be sent to the corrector"
        raise ValueError(msg)
    details = (validation.reason,) if validation.reason else ()
    return Problem("validation", validation.category, details)


def execution_problem(
    error: QueryExecutionError, sql: str, profile: DatasetProfile, *, schema_only: bool = False
) -> Problem:
    """The problem with a validated query that failed to run."""
    details: list[str] = []
    engine = _engine_text(error, schema_only)
    if engine:
        details.append(engine)
    if error.category in _NEEDS_TYPES:
        used = _columns_used(sql, profile)
        if used:
            details.append("Columns the query uses, with their types: " + ", ".join(used) + ".")
    return Problem("execution", error.category, tuple(details))


def _engine_text(error: QueryExecutionError, schema_only: bool) -> str:
    message = error.engine_message
    if schema_only:
        if error.category in _MAY_QUOTE_VALUES:
            return ""
        message = _QUOTED.sub("'…'", message)
    return clean_text(message, MAX_ENGINE_CHARS)


def _columns_used(sql: str, profile: DatasetProfile) -> list[str]:
    """``name (type)`` for each column of the table that the query's text mentions."""
    used = []
    for column in profile.columns:
        pattern = rf"(?<!\w){re.escape(column.sql_name)}(?!\w)"
        if re.search(pattern, sql, re.IGNORECASE):
            used.append(f"{column.sql_name} ({column.dtype})")
    return used[:MAX_TYPED_COLUMNS]


# -- the call ------------------------------------------------------------------------------------


class Corrector:
    """Makes the correction call for one session, with the session's context builder."""

    def __init__(self, client: LLMClient, builder: ContextBuilder | None = None) -> None:
        """Create a corrector.

        Args:
            client: The model client (``FakeLLMClient`` in tests).
            builder: The session's context builder, so the marker code is the same one the
                generator used; a new one is made if none is given.
        """
        self._client = client
        self._builder = builder or ContextBuilder()

    def correct(
        self,
        messages: Sequence[Message],
        previous: GenerationResult,
        problem: Problem,
        *,
        budget: CallBudget | None = None,
    ) -> Correction:
        """Ask the model for a corrected answer. One call; the caller decides about retries.

        Args:
            messages: The conversation so far, ending with a user message: the generator's
                ``Generation.opening`` for the first correction, ``Correction.messages``
                for a later one.
            previous: The answer that had the problem, which is added to the conversation.
            problem: What was wrong with it.
            budget: The question's call budget; the call is charged to it.

        Raises:
            LLMError: The call failed, or the answer was unusable (``invalid_output``).
                ``usage`` holds the cost when the call reached the model.
            BudgetExceededError: No call was left.
        """
        request = self._builder.correction_message(problem.title, problem.details, problem.request)
        sent = (
            *messages,
            Message("assistant", previous.model_dump_json()),
            Message("user", request),
        )
        response = self._client.complete("correct", sent, GenerationResult, budget=budget)
        result = clean_result(response.parsed)
        sql = (result.sql or "").strip() or None
        log.info(
            "correction",
            extra={
                "stage": problem.stage,
                "category": problem.category,
                "returned_sql": sql is not None,
                "input_tokens": response.usage.input_tokens,
            },
        )
        return Correction(
            result=result,
            sql=sql,
            message="" if sql else _no_sql_message(result),
            messages=sent,
            usage=response.usage,
        )


def _no_sql_message(result: GenerationResult) -> str:
    """What to tell the user when the model, asked to fix a query, gave none."""
    text = (result.clarification_question or "") if result.needs_clarification else ""
    return clean_text(text or result.explanation, MAX_MESSAGE_CHARS) or _NO_SQL_MESSAGE
