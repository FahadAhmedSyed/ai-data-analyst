"""A stand-in for the model that always says the right thing. It is not a model.

``OracleClient`` answers a case with its gold SQL, asks the clarifying question when that is
the right response, and refuses when a refusal is. A run with it proves the harness, the
scoring and the validator: every gold query must pass the real validator and executor, and
every case must score as correct. It says nothing about how well a language model does, so
its reports say ORACLE at the top and are named for it.

    uv run python -m eval.run --oracle
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from analyst.core.models import (
    Ambiguity,
    GenerationResult,
    Intent,
    ModelExplanation,
    Usage,
)
from analyst.llm.client import DEFAULT_MAX_OUTPUT_TOKENS, CallBudget, LLMResponse, Message, Stage, T
from eval.cases import Case

MODEL_NAME = "oracle"
EXPLANATION = "The table below is the result of your question."

_INTENT = Intent(
    operation="other",
    measures=[],
    dimensions=[],
    filters=[],
    time_range=None,
    sort=[],
    limit=None,
    output="table",
)


def _reading(question: str, **changes: object) -> GenerationResult:
    fields: dict[str, object] = {
        "resolved_question": question,
        "intent": _INTENT,
        "assumptions": [],
        "ambiguities": [],
        "needs_clarification": False,
        "clarification_question": None,
        "sql": None,
        "explanation": "Answers the question from the data.",
        "chart": None,
    }
    return GenerationResult(**{**fields, **changes})


def replies_for(case: Case, table: str) -> list[GenerationResult]:
    """What the oracle says to each question of ``case``: earlier turns are answered with a
    row count, and the last question as the case's right response."""
    count = f"SELECT COUNT(*) AS row_count FROM {table}"  # noqa: S608 - the table is ours
    replies = [_reading(turn, sql=count) for turn in case.turns]
    if case.gold_sql:
        replies.append(_reading(case.question, sql=case.gold_sql))
    elif case.clarify:
        ambiguity = Ambiguity(
            term=case.clarify_about or "the question",
            readings=["one reading", "another reading"],
            changes_result=True,
        )
        replies.append(
            _reading(
                case.question,
                needs_clarification=True,
                clarification_question=f"Could you say more about {case.clarify_about}?",
                ambiguities=[ambiguity],
            )
        )
    else:
        replies.append(
            _reading(case.question, explanation="This tool only reads data, so it cannot do that.")
        )
    return replies


class OracleClient:
    """``LLMClient`` that replays the right responses for one case."""

    def __init__(self, replies: Sequence[GenerationResult]) -> None:
        """Say ``replies`` to the generate calls, in order."""
        self._replies = list(replies)
        self.stages: list[str] = []

    @classmethod
    def for_case(cls, case: Case, table: str) -> OracleClient:
        """The oracle for ``case``, whose data is in table ``table``."""
        return cls(replies_for(case, table))

    def complete(
        self,
        stage: Stage,
        messages: Sequence[Message],
        response_model: type[T],
        *,
        budget: CallBudget | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> LLMResponse[T]:
        """The next right thing to say. A correction is never asked for: a gold query that
        needs one is a bad case, and the question fails so the report shows it."""
        if budget is not None:
            budget.charge()
        self.stages.append(stage)
        parsed: BaseModel
        if stage == "generate":
            if not self._replies:
                msg = "the oracle was asked more questions than the case has turns"
                raise AssertionError(msg)
            parsed = self._replies.pop(0)
        elif stage == "explain":
            parsed = ModelExplanation(answer_markdown=EXPLANATION, key_numbers=[], caveats=[])
        else:
            msg = "the oracle was asked to correct a gold query"
            raise AssertionError(msg)
        assert isinstance(parsed, response_model)  # noqa: S101 - a harness bug, not a user error
        return LLMResponse(parsed=parsed, usage=Usage(llm_calls=1), model=MODEL_NAME)
