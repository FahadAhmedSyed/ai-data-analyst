"""Ask the model what a question means and which query answers it (T1.12, ADR 0006, ADR 0011).

``Generator.generate`` makes LLM call 1: one structured answer holding the question restated on
its own, its intent, the assumptions made, any ambiguity, and the SQL. It does not run or
validate the SQL; that is what the validator and executor are for. What it adds around the
call is the code that decides what the answer is:

* ``sql``: a query to validate and run.
* ``clarification``: the user is asked one question (ADR 0011). The gate is in code: a
  question is only asked if some ambiguity would change the result. If the model asks without
  one, it is told to take the most likely reading and write the SQL, once; if it still asks,
  the question is passed on rather than failing the user.
* ``refused``: the model declined, with no SQL and no question, because the request was to
  change data, read files or ignore the rules (benchmark case I02). The user is told why and
  nothing runs. Code owns this status, so a model cannot turn it into SQL.

An answer that is inconsistent (it asks a question but gives none, or gives nothing at all) is
asked for again once. A second failure raises ``LLMError("invalid_output")`` with the cost of
both calls, because there is no useful fallback for a question that was never understood.

Text written for the user (the restatement, assumptions, the clarifying question and the
refusal) is cleaned with ``sanitize_text`` so that links and images cannot reach the screen.
The SQL is left as written: it is shown as code and is judged by the validator.

``Generation.unknown_columns`` lists intent columns that do not exist in the table, with the
closest real names, so the orchestrator can send them to the corrector before validation
(architecture section 7).
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from analyst.core.errors import BudgetExceededError, LLMError
from analyst.core.logging import get_logger
from analyst.core.models import (
    Ambiguity,
    DatasetProfile,
    GenerationResult,
    Intent,
    Turn,
    Usage,
)
from analyst.ingestion.schema_card import clean_text
from analyst.llm.client import CallBudget, LLMClient, Message
from analyst.pipeline.safe_text import sanitize_text
from analyst.sql.context import ContextBuilder

log = get_logger(__name__)

MAX_ATTEMPTS = 2
MAX_MESSAGE_CHARS = 600
MAX_SUGGESTIONS = 3

Kind = Literal["sql", "clarification", "refused"]

_RETRY_UNUSABLE = "Your last answer could not be used. Answer again in the format requested."
_RETRY_INCONSISTENT = "Your answer could not be used: {problem}. Answer again in the same format."
_RETRY_UNNEEDED = (
    "You asked for clarification, but none of the ambiguities you listed would change the "
    "result. Use the most likely reading, state it in assumptions, and write the SQL."
)


@dataclass(frozen=True)
class UnknownColumn:
    """An intent column that is not in the table, and the real columns closest to its name."""

    name: str
    suggestions: tuple[str, ...]


@dataclass(frozen=True)
class Generation:
    """What call 1 produced, and what code made of it."""

    kind: Kind
    result: GenerationResult  # text for the user already cleaned; SQL as written
    sql: str | None  # set only when ``kind`` is ``sql``
    message: str  # the clarifying question or the refusal; empty for ``sql``
    unknown_columns: tuple[UnknownColumn, ...]
    clarification_overridden: bool  # the model asked, but nothing needed asking
    opening: tuple[Message, ...]  # system and user messages, for the corrector to continue
    usage: Usage
    attempts: int  # model calls made
    approx_tokens: int  # the builder's estimate of the prompt, to compare with real usage

    @property
    def intent_problem(self) -> str | None:
        """A sentence for the corrector when the intent names columns that do not exist."""
        if not self.unknown_columns:
            return None
        parts = []
        for column in self.unknown_columns:
            hint = f" (did you mean {', '.join(column.suggestions)}?)" if column.suggestions else ""
            parts.append(f"{column.name}{hint}")
        return "the intent names columns that are not in the table: " + "; ".join(parts)


class Generator:
    """Makes LLM call 1 for one session, with the session's context builder."""

    def __init__(self, client: LLMClient, builder: ContextBuilder | None = None) -> None:
        """Create a generator.

        Args:
            client: The model client (``FakeLLMClient`` in tests).
            builder: The session's context builder, so the marker code and the system prompt
                stay the same all session; a new one is made if none is given.
        """
        self._client = client
        self._builder = builder or ContextBuilder()

    def generate(
        self,
        question: str,
        profile: DatasetProfile,
        *,
        sample_rows: Sequence[Mapping[str, Any]] = (),
        history: Sequence[Turn] = (),
        carried_assumptions: Sequence[str] = (),
        schema_only: bool = False,
        budget: CallBudget | None = None,
    ) -> Generation:
        """Understand ``question`` and propose a query for it.

        Args:
            question: The user's question.
            profile: The dataset profile.
            sample_rows: Rows of the table, for the prompt (ignored in schema-only mode).
            history: Earlier turns, oldest first.
            carried_assumptions: Assumptions from answered turns that still apply.
            schema_only: Send no values from the file (ADR 0010).
            budget: The question's call budget; each model call is charged to it.

        Raises:
            ValueError: The question is empty or too long (``ContextBuilder``).
            LLMError: The call failed, or two answers in a row were unusable. ``usage`` holds
                the cost of every call that reached the model.
            BudgetExceededError: No call was left for the first attempt.
        """
        context = self._builder.generation_context(
            question,
            profile,
            sample_rows=sample_rows,
            history=history,
            carried_assumptions=carried_assumptions,
            schema_only=schema_only,
        )
        opening = (Message("system", context.system), Message("user", context.user))
        messages = list(opening)
        usage = Usage()
        attempts = 0
        last_problem = "no usable answer"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            final = attempt == MAX_ATTEMPTS
            try:
                response = self._client.complete(
                    "generate", messages, GenerationResult, budget=budget
                )
            except BudgetExceededError:
                if attempt == 1:
                    raise
                raise LLMError("invalid_output", "no call left to ask again", usage=usage) from None
            except LLMError as error:
                attempts += 1
                usage += error.usage or Usage()
                if error.category == "invalid_output" and not final:
                    messages = [*opening, Message("user", _RETRY_UNUSABLE)]
                    continue
                raise LLMError(
                    error.category, error.detail, user_message=error.user_message, usage=usage
                ) from error
            attempts += 1
            usage += response.usage
            result = clean_result(response.parsed)
            kind, overridden, retry = _classify(result, final=final)
            if kind is None:
                last_problem = retry
                messages = [
                    *opening,
                    Message("assistant", response.parsed.model_dump_json()),
                    Message("user", retry),
                ]
                continue
            generation = self._build(
                kind, overridden, result, profile, opening, usage, attempts, context.approx_tokens
            )
            log.info(
                "generation",
                extra={
                    "kind": kind,
                    "attempts": attempts,
                    "clarification_overridden": overridden,
                    "unknown_columns": len(generation.unknown_columns),
                    "approx_tokens": context.approx_tokens,
                    "input_tokens": response.usage.input_tokens,
                },
            )
            return generation
        raise LLMError("invalid_output", last_problem, usage=usage)

    @staticmethod
    def _build(
        kind: Kind,
        overridden: bool,
        result: GenerationResult,
        profile: DatasetProfile,
        opening: tuple[Message, ...],
        usage: Usage,
        attempts: int,
        approx_tokens: int,
    ) -> Generation:
        sql = (result.sql or "").strip() or None
        if kind == "clarification":
            message = clean_text(result.clarification_question or "", MAX_MESSAGE_CHARS)
        elif kind == "refused":
            message = clean_text(result.explanation, MAX_MESSAGE_CHARS)
        else:
            message = ""
        return Generation(
            kind=kind,
            result=result,
            sql=sql if kind == "sql" else None,
            message=message,
            unknown_columns=unknown_columns(result.intent, profile) if kind == "sql" else (),
            clarification_overridden=overridden,
            opening=opening,
            usage=usage,
            attempts=attempts,
            approx_tokens=approx_tokens,
        )


# -- reading the answer --------------------------------------------------------------------------


def clean_result(result: GenerationResult) -> GenerationResult:
    """The answer with every text the user will read made safe. The SQL is not touched."""
    return result.model_copy(
        update={
            "resolved_question": sanitize_text(result.resolved_question),
            "assumptions": [a for a in (sanitize_text(a) for a in result.assumptions) if a],
            "ambiguities": [
                Ambiguity(
                    term=sanitize_text(a.term),
                    readings=[sanitize_text(r) for r in a.readings],
                    changes_result=a.changes_result,
                )
                for a in result.ambiguities
            ],
            "clarification_question": (
                sanitize_text(result.clarification_question)
                if result.clarification_question is not None
                else None
            ),
            "explanation": sanitize_text(result.explanation),
        }
    )


def _classify(result: GenerationResult, *, final: bool) -> tuple[Kind | None, bool, str]:
    """What the answer is: ``(kind, overridden, retry_message)``; ``kind`` None means ask again."""
    sql = (result.sql or "").strip()
    if result.needs_clarification:
        if not (result.clarification_question or "").strip():
            problem = "needs_clarification is true but clarification_question is empty"
            return None, False, _RETRY_INCONSISTENT.format(problem=problem)
        if any(a.changes_result for a in result.ambiguities):
            return "clarification", False, ""
        # Nothing the model listed would change the result, so asking is not warranted.
        if sql:
            return "sql", True, ""
        if not final:
            return None, False, _RETRY_UNNEEDED
        return "clarification", False, ""
    if sql:
        return "sql", False, ""
    if result.explanation.strip():
        return "refused", False, ""
    problem = "there is no SQL, no clarification and no explanation"
    return None, False, _RETRY_INCONSISTENT.format(problem=problem)


# -- the intent against the table ----------------------------------------------------------------


def unknown_columns(intent: Intent, profile: DatasetProfile) -> tuple[UnknownColumn, ...]:
    """Intent columns that are not columns of the table, with the closest real names.

    A name matches if it is a SQL name, or the same with different capitalisation. The
    original header is not accepted: the prompt asks for SQL names, and the SQL has to use them.
    """
    known = {c.sql_name for c in profile.columns}
    folded = {name.casefold(): name for name in known}
    originals = {c.display_name.casefold(): c.sql_name for c in profile.columns}
    missing: list[UnknownColumn] = []
    for name in sorted(intent.source_columns()):
        if name in known or name.casefold() in folded:
            continue
        missing.append(UnknownColumn(name, _closest(name, folded, originals)))
    return tuple(missing)


def _closest(name: str, folded: dict[str, str], originals: dict[str, str]) -> tuple[str, ...]:
    """SQL names whose SQL name or original header is close to ``name``, best first."""
    pool = {**originals, **folded}
    picks: list[str] = []
    for match in difflib.get_close_matches(
        name.casefold(), list(pool), n=2 * MAX_SUGGESTIONS, cutoff=0.6
    ):
        if pool[match] not in picks:
            picks.append(pool[match])
    return tuple(picks[:MAX_SUGGESTIONS])
