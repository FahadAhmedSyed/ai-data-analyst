"""The generator (T1.12): call 1 and what code makes of its answer.

The model is a ``FakeLLMClient`` that says exactly what each test scripts.
"""

from __future__ import annotations

import logging

import pytest

from analyst.core.errors import BudgetExceededError, LLMError
from analyst.core.models import (
    Ambiguity,
    ChartSpec,
    FilterCondition,
    GenerationResult,
    Intent,
    Measure,
    SortKey,
    Usage,
)
from analyst.llm.client import CallBudget
from analyst.llm.fake import FakeLLMClient
from analyst.pipeline.generator import (
    MAX_ATTEMPTS,
    MAX_MESSAGE_CHARS,
    Generator,
    UnknownColumn,
    unknown_columns,
)
from analyst.sql.context import ContextBuilder
from tests.unit.test_context import CODE, HISTORY, INTENT, PROFILE, ROWS

QUESTION = "Total revenue by region?"
SQL = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region"


def answer(**changes) -> GenerationResult:
    fields = {
        "resolved_question": "What is the total revenue for each region?",
        "intent": INTENT,
        "assumptions": ["Revenue means the revenue column."],
        "ambiguities": [],
        "needs_clarification": False,
        "clarification_question": None,
        "sql": SQL,
        "explanation": "Adds up revenue for each region.",
        "chart": ChartSpec(type="bar", x="region", y=["total_revenue"], color=None),
    }
    return GenerationResult(**{**fields, **changes})


def asks(question="Do you mean revenue or quantity?", **changes) -> GenerationResult:
    fields = {
        "needs_clarification": True,
        "clarification_question": question,
        "sql": None,
        "ambiguities": [
            Ambiguity(term="best", readings=["revenue", "quantity"], changes_result=True)
        ],
        **changes,
    }
    return answer(**fields)


def refuses(explanation="This tool only reads data, so it cannot change it.") -> GenerationResult:
    return answer(sql=None, explanation=explanation, chart=None)


def generator(*script, usage: Usage | None = None) -> tuple[Generator, FakeLLMClient]:
    client = FakeLLMClient(list(script), usage=usage)
    return Generator(client, ContextBuilder("NZ", code=CODE)), client


def run(gen: Generator, **kwargs):
    kwargs.setdefault("sample_rows", ROWS)
    return gen.generate(QUESTION, PROFILE, **kwargs)


# -- the call -------------------------------------------------------------------------------------


def test_a_usable_answer_is_returned_with_its_sql() -> None:
    gen, client = generator(answer())
    out = run(gen)
    assert out.kind == "sql"
    assert out.sql == SQL
    assert out.message == ""
    assert out.attempts == 1
    assert out.clarification_overridden is False
    assert out.unknown_columns == ()
    assert out.intent_problem is None
    assert len(client.calls) == 1


def test_the_call_is_the_generate_stage_with_the_builders_prompt() -> None:
    gen, client = generator(answer())
    out = run(gen)
    (call,) = client.calls
    assert call.stage == "generate"
    assert call.response_model is GenerationResult
    expected = ContextBuilder("NZ", code=CODE).generation_context(
        QUESTION, PROFILE, sample_rows=ROWS
    )
    assert [m.role for m in call.messages] == ["system", "user"]
    assert call.messages[0].content == expected.system
    assert call.messages[1].content == expected.user
    assert out.opening == call.messages
    assert out.approx_tokens == expected.approx_tokens


def test_history_and_schema_only_mode_reach_the_prompt() -> None:
    gen, client = generator(answer(), answer())
    run(gen, history=HISTORY)
    full = client.calls[0].messages[1].content
    assert "Earlier turns, oldest first." in full
    assert "Sample rows (3):" in full
    run(gen, schema_only=True)
    schema_only = client.calls[1].messages[1].content
    assert "Sample rows" not in schema_only
    assert '"revenue": 120.5' not in schema_only


def test_an_empty_question_is_refused_before_any_call() -> None:
    gen, client = generator(answer())
    with pytest.raises(ValueError, match="empty"):
        gen.generate("   ", PROFILE)
    assert client.calls == []


def test_the_usage_of_the_call_is_reported() -> None:
    usage = Usage(llm_calls=1, input_tokens=2500, output_tokens=300, cost_usd=0.01)
    gen, _ = generator(answer(), usage=usage)
    assert run(gen).usage == usage


def test_each_call_is_charged_to_the_budget() -> None:
    budget = CallBudget(5)
    gen, _ = generator(answer())
    run(gen, budget=budget)
    assert budget.used == 1


def test_the_sql_is_stripped_but_otherwise_left_as_written() -> None:
    sql = "  SELECT *\nFROM sales -- a comment  \n"
    out = run(generator(answer(sql=sql))[0])
    assert out.sql == "SELECT *\nFROM sales -- a comment"


def test_the_chart_suggestion_and_intent_pass_through() -> None:
    out = run(generator(answer())[0])
    assert out.result.chart == ChartSpec(type="bar", x="region", y=["total_revenue"], color=None)
    assert out.result.intent == INTENT


# -- clarification (ADR 0011) -------------------------------------------------------------------


def test_a_question_that_would_change_the_result_is_asked() -> None:
    out = run(generator(asks())[0])
    assert out.kind == "clarification"
    assert out.message == "Do you mean revenue or quantity?"
    assert out.sql is None
    assert out.clarification_overridden is False
    assert out.attempts == 1


def test_a_question_asked_with_sql_still_runs_nothing() -> None:
    out = run(generator(asks(sql=SQL))[0])
    assert out.kind == "clarification"
    assert out.sql is None


def test_one_real_ambiguity_among_several_is_enough_to_ask() -> None:
    mixed = asks(
        ambiguities=[
            Ambiguity(term="sales", readings=["revenue"], changes_result=False),
            Ambiguity(term="best", readings=["revenue", "orders"], changes_result=True),
        ]
    )
    assert run(generator(mixed)[0]).kind == "clarification"


def test_asking_when_no_ambiguity_matters_runs_the_sql_instead() -> None:
    unneeded = answer(
        needs_clarification=True,
        clarification_question="Which revenue column?",
        ambiguities=[Ambiguity(term="revenue", readings=["revenue"], changes_result=False)],
    )
    out = run(generator(unneeded)[0])
    assert out.kind == "sql"
    assert out.sql == SQL
    assert out.clarification_overridden is True
    assert out.message == ""


def test_an_unneeded_question_without_sql_is_asked_again_once_with_the_reason() -> None:
    unneeded = asks(
        ambiguities=[Ambiguity(term="best", readings=["revenue"], changes_result=False)]
    )
    gen, client = generator(unneeded, answer())
    out = run(gen)
    assert out.kind == "sql"
    assert out.attempts == 2
    assert out.clarification_overridden is False
    retry = client.calls[1].messages
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[:2] == client.calls[0].messages
    assert "none of the ambiguities" in retry[3].content
    assert "most likely reading" in retry[3].content


def test_asked_twice_without_cause_the_question_is_passed_on_not_failed() -> None:
    unneeded = asks(
        ambiguities=[Ambiguity(term="best", readings=["revenue"], changes_result=False)]
    )
    gen, client = generator(unneeded, unneeded)
    out = run(gen)
    assert out.kind == "clarification"
    assert out.attempts == 2
    assert client.remaining == 0


def test_asking_with_no_ambiguities_listed_counts_as_unneeded() -> None:
    bare = asks(ambiguities=[])
    out = run(generator(bare, answer())[0])
    assert out.kind == "sql"


def test_needs_clarification_without_a_question_is_asked_again() -> None:
    broken = answer(needs_clarification=True, clarification_question="  ", sql=None)
    gen, client = generator(broken, asks())
    out = run(gen)
    assert out.kind == "clarification"
    assert out.attempts == 2
    assert "clarification_question is empty" in client.calls[1].messages[-1].content


# -- a refusal (case I02) -------------------------------------------------------------------------


def test_no_sql_and_no_question_is_a_refusal_with_the_models_reason() -> None:
    gen, client = generator(refuses())
    out = run(gen)
    assert out.kind == "refused"
    assert out.sql is None
    assert out.message == "This tool only reads data, so it cannot change it."
    assert out.attempts == 1
    assert out.unknown_columns == ()
    assert len(client.calls) == 1


def test_a_refusal_with_sql_is_not_a_refusal() -> None:
    assert run(generator(answer(explanation="This tool only reads data."))[0]).kind == "sql"


def test_nothing_at_all_is_asked_again_and_then_fails() -> None:
    empty = answer(sql=None, explanation="   ")
    gen, client = generator(empty, empty)
    with pytest.raises(LLMError) as raised:
        run(gen)
    assert raised.value.category == "invalid_output"
    assert "no SQL, no clarification and no explanation" in raised.value.detail
    assert raised.value.usage == Usage(llm_calls=2, input_tokens=200, output_tokens=100)
    assert len(client.calls) == MAX_ATTEMPTS


def test_a_second_try_can_recover_from_an_empty_answer() -> None:
    empty = answer(sql=None, explanation="")
    out = run(generator(empty, answer())[0])
    assert out.kind == "sql"
    assert out.attempts == 2
    assert out.usage == Usage(llm_calls=2, input_tokens=200, output_tokens=100)


# -- calls that fail ------------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["api_error", "timeout", "rate_limited", "not_configured"])
def test_a_failed_call_is_raised_at_once_with_its_message(category: str) -> None:
    gen, client = generator(LLMError(category, "detail"), answer())
    with pytest.raises(LLMError) as raised:
        run(gen)
    assert raised.value.category == category
    assert raised.value.detail == "detail"
    assert client.remaining == 1


def test_an_unusable_answer_is_asked_for_again_once() -> None:
    billed = Usage(llm_calls=1, input_tokens=900, output_tokens=0, cost_usd=0.002)
    gen, client = generator(LLMError("invalid_output", "refusal", usage=billed), answer())
    out = run(gen)
    assert out.kind == "sql"
    assert out.attempts == 2
    assert out.usage == billed + Usage(llm_calls=1, input_tokens=100, output_tokens=50)
    assert [m.role for m in client.calls[1].messages] == ["system", "user", "user"]
    assert "could not be used" in client.calls[1].messages[-1].content


def test_two_unusable_answers_raise_with_the_cost_of_both() -> None:
    billed = Usage(llm_calls=1, input_tokens=900, output_tokens=0, cost_usd=0.002)
    error = LLMError("invalid_output", "incomplete:max_output_tokens", usage=billed)
    gen, _ = generator(error, error)
    with pytest.raises(LLMError) as raised:
        run(gen)
    assert raised.value.category == "invalid_output"
    assert raised.value.detail == "incomplete:max_output_tokens"
    assert raised.value.usage == billed + billed
    assert raised.value.user_message == error.user_message


def test_a_failure_on_the_second_call_keeps_the_first_calls_cost() -> None:
    billed = Usage(llm_calls=1, input_tokens=900, output_tokens=0, cost_usd=0.002)
    gen, _ = generator(LLMError("invalid_output", usage=billed), LLMError("timeout"))
    with pytest.raises(LLMError) as raised:
        run(gen)
    assert raised.value.category == "timeout"
    assert raised.value.usage == billed


def test_a_spent_budget_raises_before_any_call() -> None:
    gen, client = generator(answer())
    with pytest.raises(BudgetExceededError):
        run(gen, budget=CallBudget(0))
    assert client.calls == []


def test_a_budget_for_one_call_cannot_pay_for_a_second_attempt() -> None:
    empty = answer(sql=None, explanation="")
    gen, client = generator(empty, answer())
    with pytest.raises(LLMError) as raised:
        run(gen, budget=CallBudget(1))
    assert raised.value.category == "invalid_output"
    assert raised.value.usage == Usage(llm_calls=1, input_tokens=100, output_tokens=50)
    assert len(client.calls) == 1


# -- text for the user is made safe -------------------------------------------------------------


def test_text_for_the_user_loses_links_and_images() -> None:
    dirty = asks(
        "Do you mean [revenue](https://evil.test/a) or ![q](https://evil.test/p?d=North)?",
        resolved_question="Total revenue ![x](https://evil.test/i) by region",
        assumptions=["Revenue is in [dollars](https://evil.test/d)", "![](https://evil.test/z)"],
        ambiguities=[
            Ambiguity(
                term="best [x](https://evil.test)",
                readings=["revenue www.evil.test", "quantity"],
                changes_result=True,
            )
        ],
        explanation="See https://evil.test/e",
    )
    out = run(generator(dirty)[0])
    assert out.message == "Do you mean revenue or q?"
    assert out.result.resolved_question == "Total revenue x by region"
    assert out.result.assumptions == ["Revenue is in dollars"]
    assert out.result.ambiguities[0].term == "best x"
    assert out.result.ambiguities[0].readings == ["revenue [link removed]", "quantity"]
    assert out.result.explanation == "See [link removed]"
    assert "evil.test" not in out.result.model_dump_json()


def test_a_refusal_is_cleaned_flattened_and_cut() -> None:
    long = "I only read data. [more](https://evil.test)\n" + "x" * 2000
    out = run(generator(refuses(long))[0])
    assert "evil.test" not in out.message
    assert "\n" not in out.message
    assert MAX_MESSAGE_CHARS == 600
    assert len(out.message) <= 600
    assert out.message.startswith("I only read data. more x")


def test_a_very_long_clarifying_question_is_cut_and_flattened() -> None:
    long = "Do you mean revenue\nor quantity? " + "x" * 2000
    out = run(generator(asks(long))[0])
    assert out.kind == "clarification"
    assert "\n" not in out.message
    assert len(out.message) <= 600
    assert out.message.startswith("Do you mean revenue or quantity? xxx")


def test_the_sql_is_not_sanitised() -> None:
    sql = "SELECT 'https://example.test/[a](b)' AS link FROM sales"
    assert run(generator(answer(sql=sql))[0]).sql == sql


# -- intent columns against the table -------------------------------------------------------------


def intent_with(*columns: str, dimensions=(), sort=()) -> Intent:
    return Intent(
        operation="aggregate",
        measures=[Measure(column=c, aggregation="sum") for c in columns],
        dimensions=list(dimensions),
        filters=[],
        time_range=None,
        sort=list(sort),
        limit=None,
        output="table",
    )


def test_every_column_of_the_intent_exists() -> None:
    assert unknown_columns(INTENT, PROFILE) == ()


def test_a_misspelt_column_is_reported_with_the_closest_real_names() -> None:
    (missing,) = unknown_columns(intent_with("reveneu"), PROFILE)
    assert missing == UnknownColumn("reveneu", ("revenue",))


def test_the_users_own_header_is_recognised_and_mapped_to_the_sql_name() -> None:
    (missing,) = unknown_columns(intent_with("Order Date"), PROFILE)
    assert missing.name == "Order Date"
    assert missing.suggestions[0] == "order_date"


def test_a_header_unlike_its_sql_name_is_still_mapped_to_it() -> None:
    from tests.unit.test_context import col, make_profile

    profile = make_profile([col("amt", "decimal", "Total Sales Value"), col("region", "text")])
    (missing,) = unknown_columns(intent_with("total sales value"), profile)
    assert missing.suggestions == ("amt",)


def test_a_different_capitalisation_is_accepted() -> None:
    assert unknown_columns(intent_with("Revenue", dimensions=["REGION"]), PROFILE) == ()


def test_columns_in_dimensions_and_filters_are_checked_too() -> None:
    intent = intent_with("revenue", dimensions=["regon"]).model_copy(
        update={"filters": [FilterCondition(column="orderdate", operator=">", values=["2024"])]}
    )
    names = [m.name for m in unknown_columns(intent, PROFILE)]
    assert names == ["orderdate", "regon"]


def test_count_star_and_sort_aliases_are_not_columns() -> None:
    intent = intent_with("*", sort=[SortKey(column="total_revenue", direction="desc")])
    assert unknown_columns(intent, PROFILE) == ()


def test_a_column_with_no_close_name_has_no_suggestion() -> None:
    (missing,) = unknown_columns(intent_with("zzzzzz"), PROFILE)
    assert missing.suggestions == ()


def test_at_most_three_suggestions_are_given() -> None:
    from tests.unit.test_context import col, make_profile

    wide = make_profile([col(f"sales_{n}", "integer") for n in "abcde"])
    (missing,) = unknown_columns(intent_with("sales_"), wide)
    assert len(missing.suggestions) == 3


def test_the_generation_reports_unknown_columns_for_the_corrector() -> None:
    bad = answer(intent=intent_with("reveneu", dimensions=["regon"]))
    out = run(generator(bad)[0])
    assert out.kind == "sql"
    assert [m.name for m in out.unknown_columns] == ["regon", "reveneu"]
    assert out.intent_problem == (
        "the intent names columns that are not in the table: "
        "regon (did you mean region?); reveneu (did you mean revenue?)"
    )


def test_a_column_without_a_close_match_is_named_without_a_hint() -> None:
    out = run(generator(answer(intent=intent_with("zzzzzz")))[0])
    assert out.intent_problem == "the intent names columns that are not in the table: zzzzzz"


def test_unknown_columns_are_not_looked_for_when_nothing_will_run() -> None:
    gen, _ = generator(
        asks(intent=intent_with("reveneu")),
        refuses().model_copy(update={"intent": intent_with("x")}),
    )
    assert run(gen).unknown_columns == ()
    assert run(gen).unknown_columns == ()


# -- logs -----------------------------------------------------------------------------------------


def test_logs_carry_counts_and_the_token_estimate_never_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    gen, _ = generator(answer(explanation="private-explanation"))
    out = run(gen)
    (record,) = [r for r in caplog.records if r.getMessage() == "generation"]
    assert record.kind == "sql"
    assert record.attempts == 1
    assert record.clarification_overridden is False
    assert record.approx_tokens == out.approx_tokens
    assert record.input_tokens == 100
    text = " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)
    for forbidden in ("SUM(revenue)", "private-explanation", "Total revenue", "region"):
        assert forbidden not in text


def test_an_overridden_question_is_logged_as_such(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    unneeded = answer(
        needs_clarification=True,
        clarification_question="Which revenue column?",
        ambiguities=[],
    )
    run(generator(unneeded)[0])
    (record,) = [r for r in caplog.records if r.getMessage() == "generation"]
    assert record.clarification_overridden is True


def test_a_generator_makes_its_own_builder_when_none_is_given() -> None:
    client = FakeLLMClient([answer()])
    out = Generator(client).generate(QUESTION, PROFILE)
    assert out.kind == "sql"
    assert "<<<DATASET " in client.calls[0].messages[1].content
