"""Validate, run and, when it is fixable, correct a query within the question's limits (T1.13).

``QueryLoop.run`` takes the generator's answer and returns either a result or the reason
there is none. It is the loop in the question flow (architecture section 6):

1. If the intent names columns that are not in the table, that is corrected first. This is
   a soft problem: if it cannot be corrected (no correction or call left, a failed call) the
   query is checked as written, because the validator and the database judge the SQL itself.
2. The query goes to the validator. ``rejected`` ends the turn with a message about safety
   and is never sent to the model. ``correctable`` is a problem to correct.
3. An accepted query runs. A ``QueryExecutionError`` is a problem to correct.
4. A corrected query goes back to step 2, so every attempt is validated again.

Limits (FR-4.4): at most ``max_corrections`` corrections (settings, default 2), one of them
for a query that was too heavy (timeout or memory), and a call left over for the explainer
(``reserve_calls``): a question that spent every call on corrections would get no model
explanation. The outcome records how it ended; ``usage`` is the cost of the corrections,
including calls that reached the model and gave an unusable answer.

What reaches the user is a message in the outcome, written here from the category and never
from the model's SQL, the validator's reason or the database's message, so none of those can
carry text from the file to the screen. The one exception is ``gave_up``, where the message is
the model's own explanation, cleaned like any text it writes for the user.

A ``SessionCorruptedError`` from the executor is not caught: the session is unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from analyst.config.settings import Settings, get_settings
from analyst.core.errors import BudgetExceededError, LLMError, QueryExecutionError
from analyst.core.logging import get_logger, sql_hash
from analyst.core.models import DatasetProfile, GenerationResult, QueryResult, Usage
from analyst.database.executor import QueryExecutor
from analyst.llm.client import CallBudget
from analyst.pipeline.corrector import (
    Corrector,
    Problem,
    ProblemStage,
    execution_problem,
    intent_problem,
    validation_problem,
)
from analyst.pipeline.generator import Generation
from analyst.sql.validator import SqlValidator

log = get_logger(__name__)

DEFAULT_RESERVE_CALLS = 1

RunStatus = Literal["ok", "rejected", "unfixed", "gave_up", "budget", "llm_error"]

_REJECTED_MESSAGES = {
    "forbidden_statement": "This tool only reads data, so a query that changes data or "
    "settings was not run.",
    "forbidden_table": "The query asked for something other than the uploaded table, so it "
    "was not run.",
    "table_function": "The query asked for something other than the uploaded table, so it "
    "was not run.",
    "disallowed_function": "The query used a function that this tool does not allow, so it "
    "was not run.",
    "multiple_statements": "The query was not a single read-only query, so it was not run.",
    "parse_error": "The query could not be read, so it was not run. Try rephrasing the question.",
    "too_complex": "The query was too large to check safely, so it was not run. Try asking "
    "something simpler.",
}
_REJECTED_FALLBACK = "The query was refused by the safety checks, so it was not run."
_UNFIXED_MESSAGE = BudgetExceededError.default_messages["corrections"]
_BUDGET_MESSAGE = BudgetExceededError.default_messages["llm_calls"]


@dataclass(frozen=True)
class Attempt:
    """One query tried, and how it fared. ``category`` is ``ok`` when it ran."""

    sql: str
    stage: ProblemStage
    category: str


@dataclass(frozen=True)
class RunOutcome:
    """How a question's query ended."""

    status: RunStatus
    result: QueryResult | None  # set only when ``status`` is ``ok``
    sql: str  # the last query tried, as the model wrote it
    generation: GenerationResult  # the answer in force: the generator's, or the last correction's
    message: str  # for the user; empty when ``status`` is ``ok``
    attempts: tuple[Attempt, ...]
    corrections: int  # correction calls made
    usage: Usage  # cost of the corrections only

    @property
    def ok(self) -> bool:
        """True when the query ran and ``result`` holds its rows."""
        return self.status == "ok"


class QueryLoop:
    """Runs a question's query, correcting it when that can help."""

    def __init__(
        self,
        validator: SqlValidator,
        executor: QueryExecutor,
        corrector: Corrector,
        *,
        settings: Settings | None = None,
        reserve_calls: int = DEFAULT_RESERVE_CALLS,
    ) -> None:
        """Create a loop for one session.

        Args:
            validator: The session's validator.
            executor: The session's executor.
            corrector: The session's corrector.
            settings: Where ``max_corrections`` comes from; the application settings by default.
            reserve_calls: Model calls to leave in the question's budget for later stages.
        """
        self._validator = validator
        self._executor = executor
        self._corrector = corrector
        self._max_corrections = (settings or get_settings()).max_corrections
        self._reserve = reserve_calls

    def run(
        self,
        generation: Generation,
        profile: DatasetProfile,
        *,
        budget: CallBudget | None = None,
        schema_only: bool = False,
    ) -> RunOutcome:
        """Run ``generation``'s query, with corrections.

        Args:
            generation: The generator's answer; its ``kind`` must be ``sql``.
            profile: The dataset profile, for column types in a correction.
            budget: The question's call budget; each correction is charged to it.
            schema_only: Send no values from the file (ADR 0010).

        Raises:
            ValueError: ``generation`` holds no query to run.
            SessionCorruptedError: The session's table changed (from the executor).
        """
        if generation.kind != "sql" or not generation.sql:
            msg = "only a generation that holds SQL can be run"
            raise ValueError(msg)
        state = _State(generation)
        problem = intent_problem(generation)
        if problem is not None:
            state.attempt(problem.stage, problem.category)
        while True:
            if problem is None:
                found = self._check(state, profile, schema_only)
                if isinstance(found, RunOutcome):
                    return _logged(found)
                problem = found
            stop = self._stop_reason(state, problem, budget)
            if stop is not None and problem.stage != "intent":
                return _logged(state.end(*stop))
            step = "skip" if stop is not None else self._correct(state, problem, budget)
            if isinstance(step, RunOutcome):
                return _logged(step)
            if step != "retry":
                problem = None  # a new query to check, or a soft problem given up on

    def _check(
        self, state: _State, profile: DatasetProfile, schema_only: bool
    ) -> Problem | RunOutcome:
        """Validate and run the query in force: the problem found, or how the run ended."""
        sql = state.sql
        state.error_message = ""
        validation = self._validator.validate(sql)
        if validation.status == "rejected":
            state.attempt("validation", validation.category)
            message = _REJECTED_MESSAGES.get(validation.category, _REJECTED_FALLBACK)
            return state.end("rejected", message)
        if not validation.ok:
            problem = validation_problem(validation)
            state.attempt(problem.stage, problem.category)
            return problem
        try:
            result = self._executor.execute(validation)
        except QueryExecutionError as error:
            problem = execution_problem(error, sql, profile, schema_only=schema_only)
            state.attempt(problem.stage, problem.category)
            state.error_message = error.user_message
            return problem
        state.attempt("execution", "ok")
        return state.end("ok", "", result)

    def _stop_reason(
        self, state: _State, problem: Problem, budget: CallBudget | None
    ) -> tuple[RunStatus, str] | None:
        """Why this problem cannot be corrected, or None if it can."""
        if state.corrections >= self._max_corrections or (problem.heavy and state.heavy_corrected):
            return "unfixed", state.error_message or _UNFIXED_MESSAGE
        if budget is not None and budget.remaining <= self._reserve:
            return "budget", _BUDGET_MESSAGE
        return None

    def _correct(
        self, state: _State, problem: Problem, budget: CallBudget | None
    ) -> RunOutcome | Literal["fixed", "retry", "skip"]:
        """Make one correction. ``fixed``: a new query is in force. ``retry``: the answer was
        unusable and the same problem stands. ``skip``: a soft problem could not be corrected.
        """
        state.corrections += 1
        state.heavy_corrected = state.heavy_corrected or problem.heavy
        try:
            correction = self._corrector.correct(
                state.messages, state.result, problem, budget=budget
            )
        except LLMError as error:
            state.usage += error.usage or Usage()
            if problem.stage == "intent":
                return "skip"
            if error.category == "invalid_output":
                return "retry"
            return state.end("llm_error", error.user_message)
        state.usage += correction.usage
        state.messages = correction.messages
        state.result = correction.result
        if correction.sql is None:
            return state.end("gave_up", correction.message)
        state.sql = correction.sql
        return "fixed"


def _logged(outcome: RunOutcome) -> RunOutcome:
    log.info(
        "query_run",
        extra={
            "status": outcome.status,
            "attempts": len(outcome.attempts),
            "corrections": outcome.corrections,
            "last_category": outcome.attempts[-1].category,
        },
    )
    return outcome


class _State:
    """What one question's loop has done so far: the query in force and its history."""

    def __init__(self, generation: Generation) -> None:
        assert generation.sql is not None  # noqa: S101 - ``run`` checked it
        self.sql: str = generation.sql
        self.result = generation.result
        self.messages = generation.opening
        self.attempts: list[Attempt] = []
        self.corrections = 0
        self.usage = Usage()
        self.heavy_corrected = False
        self.error_message = ""  # the user text for the last run failure, if that was the problem

    def attempt(self, stage: ProblemStage, category: str) -> None:
        self.attempts.append(Attempt(self.sql, stage, category))
        log.info(
            "query_attempt",
            extra={"stage": stage, "category": category, "sql_hash": sql_hash(self.sql)},
        )

    def end(self, status: RunStatus, message: str, result: QueryResult | None = None) -> RunOutcome:
        return RunOutcome(
            status=status,
            result=result,
            sql=self.sql,
            generation=self.result,
            message=message,
            attempts=tuple(self.attempts),
            corrections=self.corrections,
            usage=self.usage,
        )
