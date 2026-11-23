"""One question in, one answer out: the orchestrator (T1.17, ADR 0003).

``Analyst.ask`` runs the question flow of architecture section 6 in plain Python, one
dataset and one conversation at a time:

1. **Generate** (LLM call 1): the question is restated, read as an intent, and answered with
   SQL, a clarifying question or a refusal (``Generator``). The last three turns and the
   assumptions carried forward from earlier answers are part of the prompt, so follow-ups
   such as "and for the north?" or "make it monthly" resolve against them.
2. **Run** the query (``QueryLoop``): validated, executed, and corrected up to twice.
3. **Explain** the result (LLM call 2, ``Explainer``): every number in it is checked against
   the result, and the template answer is used if the check fails.
4. **Chart** the result (``recommend_chart``): the model's suggestion if it fits, else by rule.
5. **Record** the turn in memory.

What the user gets is always an ``AnalystResponse`` and never an exception: a clarifying
question is a ``clarification``, and every failure (a refused request, a query that could
not be fixed, a failed model call, a spent budget, a damaged session, a bug) is a ``failed``
answer with a message written without the SQL, the database's message or any value from the
file. Nothing here is shared between sessions: the database, the profile, the memory and the
context builder (and so the marker code, security.md section 3) belong to one ``Analyst``.

One ``CallBudget`` per question (``MAX_LLM_CALLS``, 5) is shared by every stage. Generation
takes one call, each correction one (at most ``MAX_CORRECTIONS``, 2), and the loop always
leaves one call for the explainer, which makes at most two.

``Usage`` on the response adds the cost of every call that reached the model, including
calls that gave an unusable answer. Its ``latency_ms`` is the wall-clock time of the whole
question (ADR 0013 measures p95 question latency), not the sum of the model calls.

Logs carry the status, counts, categories and timings of a question, never its text.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from analyst.config.settings import Settings, get_settings
from analyst.core.errors import (
    AnalystError,
    BudgetExceededError,
    LLMError,
    SessionCorruptedError,
)
from analyst.core.logging import get_logger
from analyst.core.models import (
    AnalystResponse,
    ChartSpec,
    DatasetProfile,
    GenerationResult,
    QueryResult,
    Turn,
    Usage,
)
from analyst.database.executor import QueryExecutor
from analyst.database.session_db import SessionDatabase
from analyst.llm.client import CallBudget, LLMClient
from analyst.memory.conversation import ConversationMemory
from analyst.pipeline.corrector import Corrector
from analyst.pipeline.explainer import Explainer
from analyst.pipeline.generator import Generator
from analyst.pipeline.query_loop import QueryLoop, RunOutcome
from analyst.sql.context import (
    MAX_QUESTION_CHARS,
    ContextBuilder,
    clean_question,
    fetch_sample_rows,
)
from analyst.sql.validator import SqlValidator
from analyst.visualization.recommender import recommend_chart

log = get_logger(__name__)

BAD_QUESTION = f"Please type a question about your data, up to {MAX_QUESTION_CHARS:,} characters."

ProgressStage = Literal["understanding", "running", "explaining"]
"""Where a question has got to, for the interface's progress line (FR-9.2)."""
ProgressCallback = Callable[[ProgressStage], None]


@dataclass(frozen=True)
class _Understood:
    """What is known of a question's reading, for the response and turn if it then fails."""

    result: GenerationResult | None = None
    sql: str | None = None


@dataclass
class _Work:
    """What one question has used and learned so far, kept outside the stages so that a
    stage that raises still leaves its cost and what was understood behind."""

    budget: CallBudget
    progress: ProgressCallback | None = None
    usage: Usage = field(default_factory=Usage)
    understood: _Understood = field(default_factory=_Understood)
    notes: dict[str, Any] = field(default_factory=dict)  # counts and categories for the log

    def tell(self, stage: ProgressStage) -> None:
        """Tell the interface a stage has begun. Progress cannot fail a question."""
        if self.progress is None:
            return
        try:
            self.progress(stage)
        except Exception as error:
            log.error("progress_failed", extra={"error_type": type(error).__name__})


class Analyst:
    """Answers questions about one loaded dataset."""

    def __init__(
        self,
        db: SessionDatabase,
        profile: DatasetProfile,
        client: LLMClient,
        *,
        memory: ConversationMemory | None = None,
        settings: Settings | None = None,
        schema_only: bool = False,
    ) -> None:
        """Create an analyst for the table loaded in ``db``.

        Args:
            db: The session's database, with its table loaded.
            profile: The profile of that table.
            client: The model client (``FakeLLMClient`` in tests).
            memory: The session's conversation; a new, empty one if none is given.
            settings: Limits and region; the application settings by default.
            schema_only: Send no values from the file to the model (ADR 0010). It can be
                changed between questions; turns that ran before the change may still
                hold values the user or the model quoted.
        """
        self._settings = settings or get_settings()
        self._db = db
        self._profile = profile
        self._memory = memory if memory is not None else ConversationMemory()  # empty is falsy
        self.schema_only = schema_only
        builder = ContextBuilder(self._settings.default_region)
        self._generator = Generator(client, builder)
        self._loop = QueryLoop(
            SqlValidator.for_database(db, max_rows=self._settings.max_result_rows),
            QueryExecutor(db, self._settings),
            Corrector(client, builder),
            settings=self._settings,
        )
        self._explainer = Explainer(client, builder)
        self._samples: list[dict[str, Any]] | None = None

    @property
    def memory(self) -> ConversationMemory:
        """The conversation, for the interface to reset."""
        return self._memory

    def ask(self, question: str, on_progress: ProgressCallback | None = None) -> AnalystResponse:
        """Answer ``question``. Never raises for a failure of the question itself.

        ``on_progress`` is told when each stage that takes noticeable time begins, so the
        interface can show where a question is (FR-9.2). A failure in it never fails the
        question.
        """
        started = time.perf_counter()
        try:
            text = clean_question(question)
        except ValueError:
            # Nothing was asked, so nothing is recorded in the conversation.
            return _failed(question.strip()[:MAX_QUESTION_CHARS], BAD_QUESTION, _Understood())
        self._db.touch()
        work = _Work(CallBudget.from_settings(self._settings), on_progress)
        try:
            response = self._answer(text, work)
        except LLMError as error:
            work.usage += error.usage or Usage()
            response = _failed(text, error.user_message, work.understood)
        except (BudgetExceededError, SessionCorruptedError) as error:
            response = _failed(text, error.user_message, work.understood)
        except Exception as error:  # a bug must not reach the user as a stack trace
            log.error("question_crashed", extra={"error_type": type(error).__name__})
            response = _failed(text, AnalystError.fallback_message, work.understood)
        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = work.usage.model_copy(update={"latency_ms": elapsed_ms})
        response = response.model_copy(update={"usage": usage})
        self._memory.record(response.turn)
        log.info(
            "question",
            extra={
                "status": response.status,
                "llm_calls": usage.llm_calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": usage.cost_usd,
                "latency_ms": round(elapsed_ms, 1),
                "schema_only": self.schema_only,
                **work.notes,
            },
        )
        return response

    # -- the stages ----------------------------------------------------------------------------

    def _answer(self, question: str, work: _Work) -> AnalystResponse:
        work.tell("understanding")
        generation = self._generator.generate(
            question,
            self._profile,
            sample_rows=self._sample_rows(),
            history=self._memory.recent(),
            carried_assumptions=self._memory.carried_assumptions(),
            schema_only=self.schema_only,
            budget=work.budget,
        )
        work.usage += generation.usage
        work.notes["kind"] = generation.kind
        work.understood = _Understood(result=generation.result)
        if generation.kind == "clarification":
            return _clarification(question, generation.result, generation.message)
        if generation.kind == "refused":
            return _failed(question, generation.message, work.understood)
        work.tell("running")
        outcome = self._loop.run(
            generation, self._profile, budget=work.budget, schema_only=self.schema_only
        )
        work.usage += outcome.usage
        work.notes.update(
            run_status=outcome.status,
            corrections=outcome.corrections,
            attempts=len(outcome.attempts),
        )
        work.understood = _Understood(result=outcome.generation, sql=outcome.sql)
        if outcome.result is None:
            return _failed(question, outcome.message, work.understood)
        return self._explained(question, outcome, outcome.result, work)

    def _explained(
        self, question: str, outcome: RunOutcome, result: QueryResult, work: _Work
    ) -> AnalystResponse:
        reading = outcome.generation
        work.tell("explaining")
        explanation = self._explainer.explain(
            question,
            result,
            resolved_question=reading.resolved_question,
            intent=reading.intent,
            sql=outcome.sql,
            assumptions=reading.assumptions,
            budget=work.budget,
        )
        work.usage += explanation.usage
        chart = _chart(result, reading.chart)
        work.notes.update(
            explanation=explanation.source,
            fallback_reason=explanation.fallback_reason,
            chart=chart.type if chart else None,
            rows=result.row_count,
            truncated=result.truncated,
        )
        assumptions = tuple(reading.assumptions)
        turn = Turn(
            question=question,
            status="answered",
            resolved_question=reading.resolved_question,
            intent=reading.intent,
            sql=outcome.sql,
            assumptions=assumptions,
            answer_text=explanation.answer,
            result_summary=result.summary(),
        )
        return AnalystResponse(
            status="answered",
            question=question,
            message=explanation.answer,
            resolved_question=reading.resolved_question,
            assumptions=assumptions,
            caveats=explanation.caveats,
            sql=outcome.sql,
            result=result,
            chart=chart,
            grounded=True,  # checked against the result, or written by code from it
            fallback_used=explanation.fallback_used,
            turn=turn,
        )

    def _sample_rows(self) -> list[dict[str, Any]]:
        """Three rows of the table for the prompt; none in schema-only mode.

        Read once: the table of one ``Analyst`` does not change.
        """
        if self.schema_only:
            return []
        if self._samples is None:
            self._samples = fetch_sample_rows(self._db)
        return self._samples


# -- building responses --------------------------------------------------------------------------


def _clarification(question: str, reading: GenerationResult, message: str) -> AnalystResponse:
    """The model asked a question that would change the result; nothing ran."""
    turn = Turn(
        question=question,
        status="clarification",
        resolved_question=reading.resolved_question,
        intent=reading.intent,
        answer_text=message,
    )
    return AnalystResponse(
        status="clarification",
        question=question,
        message=message,
        resolved_question=reading.resolved_question,
        turn=turn,
    )


def _failed(question: str, message: str, understood: _Understood) -> AnalystResponse:
    """A question that could not be answered.

    The query is shown when there was one (the user can read what was tried); the turn
    keeps none, so a follow-up is not built on a query that did not work.
    """
    reading = understood.result
    resolved = reading.resolved_question if reading else None
    turn = Turn(
        question=question,
        status="failed",
        resolved_question=resolved,
        intent=reading.intent if reading else None,
        answer_text=message,
    )
    return AnalystResponse(
        status="failed",
        question=question,
        message=message,
        resolved_question=resolved,
        sql=understood.sql,
        turn=turn,
    )


def _chart(result: QueryResult, suggestion: ChartSpec | None) -> ChartSpec | None:
    """The chart for a result. A chart is decoration, so a failure here costs only the chart."""
    try:
        return recommend_chart(result, suggestion)
    except Exception as error:
        log.error("chart_failed", extra={"error_type": type(error).__name__})
        return None
