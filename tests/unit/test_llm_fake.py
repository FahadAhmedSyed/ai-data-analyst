"""The scripted fake client (T1.10): it must honour the same contract as the real one."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from analyst.core.errors import BudgetExceededError, LLMError
from analyst.core.models import Usage
from analyst.llm import FakeCall, FakeLLMClient
from analyst.llm.client import DEFAULT_MAX_OUTPUT_TOKENS, CallBudget, LLMResponse, Message


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class Other(BaseModel):
    text: str


ASK = [Message("user", "question")]


def test_replays_the_script_in_order():
    fake = FakeLLMClient([Answer(value=1), Answer(value=2)])
    assert fake.complete("generate", ASK, Answer).parsed == Answer(value=1)
    assert fake.complete("explain", ASK, Answer).parsed == Answer(value=2)
    assert fake.remaining == 0
    fake.assert_exhausted()


def test_records_every_call_with_what_was_asked():
    fake = FakeLLMClient([Answer(value=1), Answer(value=2)])
    messages = [Message("system", "rules"), Message("user", "q")]
    fake.complete("generate", messages, Answer, max_output_tokens=321)
    fake.complete("correct", ASK, Answer)
    first, second = fake.calls
    assert first == FakeCall("generate", tuple(messages), Answer, 321)
    assert second.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert fake.stages == ["generate", "correct"]


def test_response_carries_usage_and_model():
    fake = FakeLLMClient([Answer(value=1)], model="m-1")
    response = fake.complete("generate", ASK, Answer)
    assert isinstance(response, LLMResponse)
    assert response.model == "m-1"
    assert response.usage == Usage(llm_calls=1, input_tokens=100, output_tokens=50)


def test_usage_can_be_set():
    usage = Usage(llm_calls=1, input_tokens=7, output_tokens=3, cost_usd=0.5, latency_ms=9.0)
    fake = FakeLLMClient([Answer(value=1)], usage=usage)
    assert fake.complete("generate", ASK, Answer).usage == usage


def test_a_dict_is_validated_against_the_requested_model():
    fake = FakeLLMClient([{"value": 5}])
    assert fake.complete("generate", ASK, Answer).parsed == Answer(value=5)


@pytest.mark.parametrize("bad", [{"value": "x"}, {}, {"value": 1, "extra": 2}])
def test_a_dict_that_does_not_fit_is_invalid_output(bad):
    fake = FakeLLMClient([bad])
    with pytest.raises(LLMError) as raised:
        fake.complete("generate", ASK, Answer)
    assert raised.value.category == "invalid_output"


def test_a_string_is_parsed_as_raw_json():
    fake = FakeLLMClient(['{"value": 9}', "not json at all", '{"value": "x"}'])
    assert fake.complete("generate", ASK, Answer).parsed.value == 9
    for _ in range(2):
        with pytest.raises(LLMError) as raised:
            fake.complete("generate", ASK, Answer)
        assert raised.value.category == "invalid_output"


def test_an_exception_in_the_script_is_raised():
    fake = FakeLLMClient([LLMError("timeout"), RuntimeError("boom"), Answer(value=1)])
    with pytest.raises(LLMError) as raised:
        fake.complete("generate", ASK, Answer)
    assert raised.value.category == "timeout"
    with pytest.raises(RuntimeError, match="boom"):
        fake.complete("generate", ASK, Answer)
    assert fake.complete("generate", ASK, Answer).parsed.value == 1


def test_a_callable_sees_the_call_and_returns_the_answer():
    seen: list[FakeCall] = []

    def respond(call: FakeCall) -> Answer:
        seen.append(call)
        return Answer(value=len(call.messages))

    fake = FakeLLMClient([respond])
    assert fake.complete("generate", [*ASK, *ASK], Answer).parsed.value == 2
    assert seen == [fake.calls[0]]


def test_a_callable_may_return_a_dict_a_string_or_an_exception():
    fake = FakeLLMClient(
        [lambda _: {"value": 3}, lambda _: '{"value": 4}', lambda _: LLMError("rate_limited")]
    )
    assert fake.complete("generate", ASK, Answer).parsed.value == 3
    assert fake.complete("generate", ASK, Answer).parsed.value == 4
    with pytest.raises(LLMError):
        fake.complete("generate", ASK, Answer)


def test_a_callable_can_answer_by_stage():
    def respond(call: FakeCall) -> dict[str, int]:
        return {"value": {"generate": 1, "explain": 2}[call.stage]}

    fake = FakeLLMClient([respond, respond])
    assert fake.complete("explain", ASK, Answer).parsed.value == 2
    assert fake.complete("generate", ASK, Answer).parsed.value == 1


def test_a_model_of_the_wrong_type_is_a_test_bug():
    fake = FakeLLMClient([Other(text="x")])
    with pytest.raises(TypeError, match=r"Other.*Answer"):
        fake.complete("generate", ASK, Answer)


def test_an_unsupported_script_item_is_a_test_bug():
    fake = FakeLLMClient([42])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="unsupported"):
        fake.complete("generate", ASK, Answer)


def test_running_past_the_end_of_the_script_fails_loudly():
    fake = FakeLLMClient([Answer(value=1)])
    fake.complete("generate", ASK, Answer)
    with pytest.raises(AssertionError, match="unscripted explain call"):
        fake.complete("explain", ASK, Answer)
    assert len(fake.calls) == 2  # the unscripted call is still recorded


def test_assert_exhausted_fails_when_answers_were_never_asked_for():
    fake = FakeLLMClient([Answer(value=1), Answer(value=2)])
    fake.complete("generate", ASK, Answer)
    with pytest.raises(AssertionError, match="1 scripted"):
        fake.assert_exhausted()


def test_items_can_be_added_later():
    fake = FakeLLMClient()
    fake.add(Answer(value=1), {"value": 2})
    assert fake.remaining == 2
    assert fake.complete("generate", ASK, Answer).parsed.value == 1


def test_the_budget_is_charged_per_call_and_enforced_before_the_script_is_used():
    fake = FakeLLMClient([Answer(value=1), Answer(value=2)])
    budget = CallBudget(1)
    fake.complete("generate", ASK, Answer, budget=budget)
    with pytest.raises(BudgetExceededError):
        fake.complete("generate", ASK, Answer, budget=budget)
    assert fake.remaining == 1  # the refused call did not consume an answer
    assert len(fake.calls) == 1


def test_a_scripted_failure_still_counts_against_the_budget():
    fake = FakeLLMClient([LLMError("api_error")])
    budget = CallBudget(2)
    with pytest.raises(LLMError):
        fake.complete("generate", ASK, Answer, budget=budget)
    assert budget.used == 1


def test_messages_are_copied_so_later_changes_do_not_alter_the_record():
    fake = FakeLLMClient([Answer(value=1)])
    messages = [Message("user", "a")]
    fake.complete("generate", messages, Answer)
    messages.append(Message("user", "b"))
    assert fake.calls[0].messages == (Message("user", "a"),)


def test_a_stage_in_the_script_must_match_the_call():
    fake = FakeLLMClient([("generate", Answer(value=1)), ("explain", {"value": 2})])
    assert fake.complete("generate", ASK, Answer).parsed.value == 1
    with pytest.raises(AssertionError, match=r"stage 'correct'.*expected 'explain'"):
        fake.complete("correct", ASK, Answer)


def test_a_stage_can_be_paired_with_an_exception_or_a_callable():
    fake = FakeLLMClient([("generate", LLMError("timeout")), ("explain", lambda _: {"value": 4})])
    with pytest.raises(LLMError):
        fake.complete("generate", ASK, Answer)
    assert fake.complete("explain", ASK, Answer).parsed.value == 4
