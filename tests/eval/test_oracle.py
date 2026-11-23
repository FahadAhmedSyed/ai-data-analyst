"""The oracle: a stand-in that says the right thing. It must follow the client contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from analyst.core.errors import BudgetExceededError
from analyst.core.models import GenerationResult, ModelExplanation
from analyst.llm.client import CallBudget, LLMClient, Message
from eval.cases import Case
from eval.oracle import EXPLANATION, MODEL_NAME, OracleClient, replies_for

ANSWER = Case(
    id="S90", dataset="sales", category="simple_aggregation", question="How many orders?",
    gold_sql="SELECT COUNT(*) AS orders FROM sales", compare="scalar",
)  # fmt: skip
ASK = replace(
    ANSWER, id="A90", category="ambiguous", gold_sql=None, clarify=True,
    expected_status="clarification", clarify_about="what recent means",
)  # fmt: skip
REFUSE = replace(ANSWER, id="I90", category="injection", gold_sql=None, expected_status="failed")
FOLLOW = replace(ANSWER, id="T90", category="follow_up", turns=("First?", "Second?"))
MESSAGES = [Message("system", "s"), Message("user", "u")]


def test_the_oracle_is_a_client() -> None:
    assert isinstance(OracleClient([]), LLMClient)
    assert MODEL_NAME == "oracle"


def test_an_answer_case_gets_its_gold_sql() -> None:
    (reply,) = replies_for(ANSWER, "sales")
    assert reply.sql == ANSWER.gold_sql
    assert not reply.needs_clarification
    assert reply.resolved_question == "How many orders?"


def test_a_clarification_case_gets_a_question_back_about_what_it_is_about() -> None:
    (reply,) = replies_for(ASK, "sales")
    assert reply.needs_clarification
    assert reply.sql is None
    assert "what recent means" in (reply.clarification_question or "")
    assert reply.ambiguities[0].changes_result
    assert reply.ambiguities[0].term == "what recent means"
    assert reply.problems() == []


def test_a_refusal_case_gets_a_refusal() -> None:
    (reply,) = replies_for(REFUSE, "sales")
    assert reply.sql is None and not reply.needs_clarification
    assert "only reads data" in reply.explanation


def test_earlier_turns_are_answered_with_a_row_count_and_the_last_question_as_the_case_says() -> (
    None
):
    first, second, last = replies_for(FOLLOW, "sales")
    assert first.sql == second.sql == "SELECT COUNT(*) AS row_count FROM sales"
    assert (first.resolved_question, second.resolved_question) == ("First?", "Second?")
    assert last.sql == FOLLOW.gold_sql


def test_the_oracle_says_each_reply_in_turn_and_the_same_explanation_every_time() -> None:
    client = OracleClient.for_case(FOLLOW, "sales")
    generated = [client.complete("generate", MESSAGES, GenerationResult) for _ in range(3)]
    assert [g.parsed.resolved_question for g in generated] == [
        "First?",
        "Second?",
        "How many orders?",
    ]
    explained = client.complete("explain", MESSAGES, ModelExplanation)
    assert explained.parsed.answer_markdown == EXPLANATION
    assert explained.parsed.key_numbers == [] and explained.parsed.caveats == []
    assert explained.model == MODEL_NAME
    assert explained.usage.llm_calls == 1 and explained.usage.cost_usd == 0
    assert client.stages == ["generate"] * 3 + ["explain"]


def test_it_charges_the_budget_like_any_client() -> None:
    budget = CallBudget(1)
    client = OracleClient.for_case(ANSWER, "sales")
    client.complete("generate", MESSAGES, GenerationResult, budget=budget)
    with pytest.raises(BudgetExceededError):
        client.complete("explain", MESSAGES, ModelExplanation, budget=budget)


def test_asked_more_questions_than_the_case_has_it_says_so() -> None:
    client = OracleClient.for_case(ANSWER, "sales")
    client.complete("generate", MESSAGES, GenerationResult)
    with pytest.raises(AssertionError, match="more questions than the case has turns"):
        client.complete("generate", MESSAGES, GenerationResult)


def test_it_is_never_asked_to_correct_a_gold_query() -> None:
    with pytest.raises(AssertionError, match="correct a gold query"):
        OracleClient.for_case(ANSWER, "sales").complete("correct", MESSAGES, GenerationResult)
