"""The orchestrator (T1.17): every path of the question flow, on a real session database.

The model is a ``FakeLLMClient`` whose script names the stage of each call, so a test also
fails if the stages are called in the wrong order. The validator, executor, corrector,
explainer and chart recommender are the real ones.
"""

from __future__ import annotations

import logging
import re

import pytest

from analyst.config.settings import Settings
from analyst.core.errors import BudgetExceededError, LLMError, SessionCorruptedError
from analyst.core.models import ChartSpec, ModelExplanation, Usage
from analyst.database.session_db import SessionDatabase
from analyst.llm.fake import FakeLLMClient
from analyst.memory.conversation import ConversationMemory
from analyst.pipeline import analyst as analyst_module
from analyst.pipeline.analyst import BAD_QUESTION, Analyst
from analyst.sql.context import MAX_QUESTION_CHARS
from tests.unit.test_context import PROFILE
from tests.unit.test_executor import sales_table
from tests.unit.test_generator import QUESTION, SQL, answer, asks, refuses

FIXED = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY 1"
TYPO = "SELECT region, SUM(revnue) AS total_revenue FROM sales GROUP BY region"
TYPO_2 = "SELECT regon, SUM(revenue) AS total_revenue FROM sales GROUP BY regon"
NO_ROWS = "SELECT region, revenue FROM sales WHERE region = 'Nowhere'"
SAID = "Revenue is shown for each of the 4 regions."
TEMPLATE = "Here is the result of the query: 4 rows and 2 columns."
ONE_CALL = Usage(llm_calls=1, input_tokens=100, output_tokens=50)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.fixture
def db():
    with SessionDatabase(settings()) as database:
        database.load(sales_table())
        yield database


def explains(text: str = SAID, **kw) -> ModelExplanation:
    return ModelExplanation(answer_markdown=text, key_numbers=[], caveats=[], **kw)


def build(db, *script, usage: Usage | None = None, **kw) -> tuple[Analyst, FakeLLMClient]:
    client = FakeLLMClient(list(script), usage=usage)
    return Analyst(db, PROFILE, client, **kw), client


def happy(**changes) -> tuple:
    """The two calls of a question that goes straight through."""
    return ("generate", answer(**changes)), ("explain", explains())


# -- an answered question -------------------------------------------------------------------------


def test_a_question_that_goes_straight_through_is_answered(db) -> None:
    ana, client = build(db, *happy())
    out = ana.ask(QUESTION)
    assert out.status == "answered"
    assert out.question == QUESTION
    assert out.message == SAID
    assert out.resolved_question == "What is the total revenue for each region?"
    assert out.assumptions == ("Revenue means the revenue column.",)
    assert out.sql == SQL
    assert out.result is not None
    assert out.result.row_count == 4
    assert out.chart == ChartSpec(type="bar", x="region", y=["total_revenue"], color=None)
    assert out.grounded is True
    assert out.fallback_used is False
    assert out.caveats == ()
    assert client.stages == ["generate", "explain"]
    client.assert_exhausted()


def test_the_explainer_is_told_how_the_question_was_read_and_what_ran(db) -> None:
    ana, client = build(db, *happy())
    ana.ask(QUESTION)
    prompt = client.calls[1].messages[1].content
    assert "Restated: What is the total revenue for each region?" in prompt
    assert '"operation":"aggregate"' in prompt
    assert "Assumption: Revenue means the revenue column." in prompt
    assert f"SQL: {SQL}" in prompt
    assert "Rows returned: 4." in prompt


def test_the_explainer_is_told_about_the_corrected_answer_not_the_first(db) -> None:
    fixed = answer(
        sql=FIXED,
        resolved_question="Revenue by region, sorted.",
        assumptions=["Sorted by region."],
    )
    ana, client = build(
        db, ("generate", answer(sql=TYPO)), ("correct", fixed), ("explain", explains())
    )
    ana.ask(QUESTION)
    prompt = client.calls[2].messages[1].content
    assert "Restated: Revenue by region, sorted." in prompt
    assert "Assumption: Sorted by region." in prompt
    assert "Revenue means the revenue column." not in prompt
    assert f"SQL: {FIXED}" in prompt


def test_the_turn_is_recorded_for_the_next_question(db) -> None:
    ana, _ = build(db, *happy())
    out = ana.ask(QUESTION)
    assert ana.memory.turns == (out.turn,)
    turn = out.turn
    assert turn.status == "answered"
    assert turn.question == QUESTION
    assert turn.resolved_question == out.resolved_question
    assert turn.intent == answer().intent
    assert turn.sql == SQL
    assert turn.assumptions == ("Revenue means the revenue column.",)
    assert turn.answer_text == SAID
    assert turn.result_summary == out.result.summary()  # type: ignore[union-attr]
    assert ana.memory.carried_assumptions() == ("Revenue means the revenue column.",)


def test_the_cost_of_every_call_is_added_up(db) -> None:
    ana, _ = build(db, *happy(), usage=Usage(llm_calls=1, input_tokens=100, cost_usd=0.01))
    out = ana.ask(QUESTION)
    assert out.usage.llm_calls == 2
    assert out.usage.input_tokens == 200
    assert out.usage.cost_usd == pytest.approx(0.02)


def test_latency_is_the_wall_clock_time_of_the_question_not_the_sum_of_the_calls(db) -> None:
    slow = Usage(llm_calls=1, latency_ms=60_000)
    ana, _ = build(db, *happy(), usage=slow)
    out = ana.ask(QUESTION)
    assert 0 < out.usage.latency_ms < 10_000  # two calls that report a minute each


def test_the_models_chart_suggestion_is_used_when_it_fits_and_a_rule_when_there_is_none(db) -> None:
    ana, _ = build(db, *happy(chart=None))
    out = ana.ask(QUESTION)
    assert out.chart is not None
    assert out.chart.type == "bar"
    assert out.chart.x == "region"
    unfit = ChartSpec(type="line", x="nothing", y=["revenue"], color=None)
    ana, _ = build(db, *happy(chart=unfit))
    other = ana.ask(QUESTION).chart
    assert other is not None
    assert other.x != "nothing"


def test_the_models_suggestion_wins_over_the_rule_when_it_fits(db) -> None:
    table_only = ChartSpec(type="none", x=None, y=[], color=None)
    ana, _ = build(db, *happy(chart=table_only))
    out = ana.ask(QUESTION)
    assert out.chart == table_only  # the rule would have drawn a bar chart


def test_the_question_is_cleaned_of_unprintable_characters(db) -> None:
    ana, client = build(db, *happy())
    out = ana.ask(f"  {QUESTION}\x00\x07  ")
    assert out.question == QUESTION
    assert QUESTION in client.calls[0].messages[1].content
    assert "\x00" not in client.calls[0].messages[1].content


def test_the_longest_question_is_accepted(db) -> None:
    long = "a" * MAX_QUESTION_CHARS
    ana, _ = build(db, *happy())
    assert ana.ask(long).status == "answered"


# -- a clarifying question ------------------------------------------------------------------------


def test_a_question_that_needs_asking_is_a_clarification_and_nothing_runs(db) -> None:
    ana, client = build(db, ("generate", asks()))
    out = ana.ask("Which is best?")
    assert out.status == "clarification"
    assert out.message == "Do you mean revenue or quantity?"
    assert out.result is None
    assert out.chart is None
    assert out.sql is None
    assert out.grounded is None
    assert out.resolved_question == "What is the total revenue for each region?"
    assert out.usage.llm_calls == 1
    assert client.stages == ["generate"]
    turn = out.turn
    assert (turn.status, turn.sql, turn.assumptions) == ("clarification", None, ())
    assert turn.answer_text == "Do you mean revenue or quantity?"
    assert ana.memory.carried_assumptions() == ()  # nothing was settled


def test_the_clarifying_question_is_part_of_the_next_prompt(db) -> None:
    ana, client = build(db, ("generate", asks()), *happy())
    ana.ask("Which is best?")
    ana.ask("By revenue")
    prompt = client.calls[1].messages[1].content
    assert "Turn 1 (clarification):" in prompt
    assert "Clarifying question asked: Do you mean revenue or quantity?" in prompt


# -- a refusal ------------------------------------------------------------------------------------


def test_a_request_to_change_data_is_a_failed_answer_with_the_models_explanation(db) -> None:
    ana, client = build(db, ("generate", refuses("This tool only reads data.")))
    out = ana.ask("Delete all rows")
    assert out.status == "failed"
    assert out.message == "This tool only reads data."
    assert out.sql is None
    assert out.result is None
    assert client.stages == ["generate"]
    assert out.turn.status == "failed"
    assert out.turn.sql is None
    assert out.turn.resolved_question == "What is the total revenue for each region?"
    assert out.turn.intent == answer().intent
    assert out.turn.answer_text == "This tool only reads data."
    assert ana.memory.carried_assumptions() == ()


# -- queries that fail ----------------------------------------------------------------------------


def test_a_destructive_query_is_refused_and_never_corrected(db) -> None:
    ana, client = build(db, ("generate", answer(sql="DROP TABLE sales")))
    out = ana.ask("Drop the table")
    assert out.status == "failed"
    assert out.message == (
        "This tool only reads data, so a query that changes data or settings was not run."
    )
    assert out.sql == "DROP TABLE sales"  # shown as code, so the user can see what was refused
    assert out.turn.sql is None  # but a follow-up is not built on it
    assert client.stages == ["generate"]
    assert out.usage.llm_calls == 1
    assert db.row_count == 600  # the table is untouched


def test_a_fixable_query_is_corrected_and_the_correction_is_what_is_answered(db) -> None:
    fixed = answer(
        sql=FIXED,
        resolved_question="Revenue by region, sorted by region.",
        assumptions=["Sorted by region."],
    )
    ana, client = build(
        db, ("generate", answer(sql=TYPO)), ("correct", fixed), ("explain", explains())
    )
    out = ana.ask(QUESTION)
    assert out.status == "answered"
    assert out.sql == FIXED
    assert out.resolved_question == "Revenue by region, sorted by region."
    assert out.assumptions == ("Sorted by region.",)
    assert out.turn.sql == FIXED
    assert out.usage.llm_calls == 3
    assert client.stages == ["generate", "correct", "explain"]
    assert "FROM sales GROUP BY region ORDER BY 1" in client.calls[2].messages[1].content


def test_a_query_that_cannot_be_fixed_is_a_failed_answer_with_what_was_tried(db) -> None:
    ana, client = build(
        db,
        ("generate", answer(sql=TYPO)),
        ("correct", answer(sql=TYPO_2)),
        ("correct", answer(sql=TYPO, resolved_question="Revenue by region, tidied up.")),
    )
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert out.message == "The query couldn't be fixed within the allowed number of attempts."
    assert out.sql == TYPO
    assert out.resolved_question == "Revenue by region, tidied up."  # the last answer in force
    assert out.turn.resolved_question == "Revenue by region, tidied up."
    assert out.turn.answer_text == out.message
    assert out.usage.llm_calls == 3
    assert client.stages == ["generate", "correct", "correct"]
    assert out.turn.status == "failed"
    assert out.turn.sql is None
    assert ana.memory.carried_assumptions() == ()


def test_a_correction_never_takes_the_call_kept_for_the_explanation(db) -> None:
    ana, client = build(db, ("generate", answer(sql=TYPO)), settings=settings(max_llm_calls=2))
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert out.message == "This question needed more steps than allowed. Try asking it more simply."
    assert client.stages == ["generate"]  # the second call was kept back


def test_all_five_calls_fit_in_a_question_that_needs_everything(db) -> None:
    bad = ModelExplanation(answer_markdown="Revenue is 987654.", key_numbers=[], caveats=[])
    ana, client = build(
        db,
        ("generate", answer(sql=TYPO)),
        ("correct", answer(sql=TYPO_2)),
        ("correct", answer(sql=FIXED)),
        ("explain", bad),
        ("explain", explains()),
    )
    out = ana.ask(QUESTION)
    assert out.status == "answered"
    assert out.message == SAID
    assert out.fallback_used is False
    assert out.usage.llm_calls == 5
    assert client.stages == ["generate", "correct", "correct", "explain", "explain"]
    client.assert_exhausted()


# -- the explanation ------------------------------------------------------------------------------


def test_an_explanation_that_fails_the_check_twice_is_replaced_by_the_template(db) -> None:
    bad = ModelExplanation(answer_markdown="Revenue is 987654.", key_numbers=[], caveats=[])
    ana, client = build(db, ("generate", answer()), ("explain", bad), ("explain", bad))
    out = ana.ask(QUESTION)
    assert out.status == "answered"
    assert out.message == TEMPLATE
    assert out.fallback_used is True
    assert out.grounded is True  # written by code from the result
    assert out.result is not None
    assert out.usage.llm_calls == 3
    assert client.stages == ["generate", "explain", "explain"]


def test_an_empty_result_is_answered_by_code_and_costs_no_explanation_call(db) -> None:
    ana, client = build(db, ("generate", answer(sql=NO_ROWS)))
    out = ana.ask("Revenue in Nowhere?")
    assert out.status == "answered"
    assert out.result is not None
    assert out.result.is_empty
    assert out.message.startswith("The query ran but returned no rows.")
    assert out.fallback_used is False
    assert client.stages == ["generate"]
    assert out.usage.llm_calls == 1


def test_a_spent_budget_gives_the_template_answer_not_a_failure(db) -> None:
    ana, client = build(db, ("generate", answer()), settings=settings(max_llm_calls=1))
    out = ana.ask(QUESTION)
    assert out.status == "answered"
    assert out.message == TEMPLATE
    assert out.fallback_used is True
    assert client.stages == ["generate"]


def test_a_result_cut_at_the_row_cap_says_so(db) -> None:
    ana, _ = build(
        db,
        ("generate", answer(sql="SELECT order_id FROM sales", chart=None)),
        ("explain", explains("The first 2 order ids are shown.")),
        settings=settings(max_result_rows=2),
    )
    out = ana.ask("List the orders")
    assert out.status == "answered"
    assert out.result is not None
    assert (out.result.row_count, out.result.truncated) == (2, True)
    assert out.caveats == ("The result was cut at 2 rows, so there are more rows than are shown.",)


# -- failures that are not the query's ------------------------------------------------------------


def test_a_failed_first_call_is_a_failed_answer_that_counts_its_cost(db) -> None:
    boom = LLMError("timeout", "slow", usage=Usage(llm_calls=1, input_tokens=7))
    ana, _ = build(db, ("generate", boom))
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert out.message == "The language model took too long to respond. Please try again."
    assert out.usage.llm_calls == 1
    assert out.usage.input_tokens == 7
    assert out.turn.status == "failed"


def test_two_unusable_first_answers_are_a_failed_answer_that_counts_both_calls(db) -> None:
    unusable = LLMError("invalid_output", "no", usage=Usage(llm_calls=1, input_tokens=5))
    ana, _ = build(db, ("generate", unusable), ("generate", unusable))
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert out.message == "The language model returned an answer in an unexpected form."
    assert out.usage.llm_calls == 2
    assert out.usage.input_tokens == 10


def test_an_unconfigured_model_is_a_failed_answer(db) -> None:
    ana, _ = build(db, ("generate", LLMError("not_configured", "no key")))
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert "isn't set up yet" in out.message


def test_a_budget_error_from_a_stage_is_a_failed_answer(db, monkeypatch) -> None:
    ana, _ = build(db)

    def refuse(*args, **kwargs):
        raise BudgetExceededError("llm_calls")

    monkeypatch.setattr(ana._generator, "generate", refuse)
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert out.message == "This question needed more steps than allowed. Try asking it more simply."


def test_a_damaged_session_is_a_failed_answer_that_keeps_what_was_understood(
    db, monkeypatch
) -> None:
    ana, _ = build(db, ("generate", answer()))

    def corrupted(*args, **kwargs):
        raise SessionCorruptedError("changed")

    monkeypatch.setattr(ana._loop, "run", corrupted)
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert out.message.startswith("Your data changed unexpectedly")
    assert out.resolved_question == "What is the total revenue for each region?"
    assert out.usage.llm_calls == 1


def test_a_bug_is_a_failed_answer_with_a_plain_message_and_a_log_without_its_text(
    db, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    ana, _ = build(db)

    def bug(*args, **kwargs):
        raise RuntimeError("secret-value-4711 leaked into an error message")

    monkeypatch.setattr(ana._generator, "generate", bug)
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert out.message == "Something went wrong. Please try again."
    assert "secret" not in out.message
    (record,) = [r for r in caplog.records if r.getMessage() == "question_crashed"]
    assert record.error_type == "RuntimeError"
    assert "secret-value-4711" not in " ".join(
        f"{r.getMessage()} {vars(r)}" for r in caplog.records
    )
    assert ana.memory.last_turn is not None
    assert ana.memory.last_turn.status == "failed"


def test_a_failing_chart_costs_the_chart_and_nothing_else(db, monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO)

    def broken(*args, **kwargs):
        raise ValueError("no chart for you")

    monkeypatch.setattr(analyst_module, "recommend_chart", broken)
    ana, _ = build(db, *happy())
    out = ana.ask(QUESTION)
    assert out.status == "answered"
    assert out.chart is None
    assert out.message == SAID
    (record,) = [r for r in caplog.records if r.getMessage() == "chart_failed"]
    assert record.error_type == "ValueError"


# -- questions that are not questions -------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "\x00\x07", "a" * (MAX_QUESTION_CHARS + 1)])
def test_a_missing_or_oversized_question_is_a_failed_answer_without_any_call(db, text) -> None:
    ana, client = build(db)
    out = ana.ask(text)
    assert out.status == "failed"
    assert out.message == BAD_QUESTION
    assert out.message == "Please type a question about your data, up to 2,000 characters."
    assert out.usage == Usage()
    assert len(out.question) <= MAX_QUESTION_CHARS
    assert client.calls == []
    assert len(ana.memory) == 0  # nothing was asked, so nothing is remembered


# -- memory and follow-ups ------------------------------------------------------------------------


def test_the_prompt_carries_earlier_turns_and_assumptions(db) -> None:
    ana, client = build(db, *happy(), *happy())
    ana.ask(QUESTION)
    ana.ask("And by month?")
    first = client.calls[0].messages[1].content
    second = client.calls[2].messages[1].content
    assert "Earlier turns" not in first
    assert "Turn 1 (answered):" in second
    assert f"Question: {QUESTION}" in second
    assert "  - Revenue means the revenue column." in second
    assert "Assumptions already shown to the user" in second


def test_an_assumption_stays_after_its_turn_leaves_the_window(db) -> None:
    script = []
    for number in range(1, 6):
        script += [
            ("generate", answer(assumptions=[f"Assumption number {number}."])),
            ("explain", explains()),
        ]
    ana, client = build(db, *script)
    for number in range(1, 6):
        ana.ask(f"Question {number}?")
    prompt = client.calls[8].messages[1].content  # the fifth question
    assert "Turn 1" in prompt  # turns 2 to 4 are in the window, so they number 1 to 3 ...
    assert "Question: Question 1?" not in prompt  # ... and the first question is out of it
    assert "  - Assumption number 1." in prompt  # but its assumption still applies
    assert "  - Assumption number 4." in prompt


def test_a_failed_question_settles_nothing(db) -> None:
    ana, client = build(db, ("generate", answer(sql="DROP TABLE sales")), *happy())
    ana.ask("Drop it")
    ana.ask(QUESTION)
    prompt = client.calls[1].messages[1].content
    assert "Turn 1 (failed):" in prompt
    assert "Assumptions already shown" not in prompt
    assert "SQL: DROP" not in prompt


def test_the_conversation_can_be_reset_by_the_interface(db) -> None:
    memory = ConversationMemory()
    ana, client = build(db, *happy(), *happy(), memory=memory)
    assert ana.memory is memory
    ana.ask(QUESTION)
    memory.reset()
    ana.ask(QUESTION)
    assert "Earlier turns" not in client.calls[2].messages[1].content


# -- privacy --------------------------------------------------------------------------------------


def test_sample_rows_are_sent_unless_the_session_is_schema_only(db) -> None:
    ana, client = build(db, *happy(), *happy())
    ana.ask(QUESTION)
    assert "Sample rows (3):" in client.calls[0].messages[1].content
    ana.schema_only = True  # switched between questions
    ana.ask(QUESTION)
    second = client.calls[2].messages[1].content
    assert "Sample rows" not in second
    assert '"region": "' not in second


def test_the_sample_rows_are_read_from_the_table_once(db, monkeypatch) -> None:
    reads: list[int] = []
    real = analyst_module.fetch_sample_rows

    def counting(database):
        reads.append(1)
        return real(database)

    monkeypatch.setattr(analyst_module, "fetch_sample_rows", counting)
    ana, _ = build(db, *happy(), *happy())
    ana.ask(QUESTION)
    ana.ask(QUESTION)
    assert reads == [1]


def test_the_sample_rows_are_not_read_at_all_in_schema_only_mode(db, monkeypatch) -> None:
    def never(database):
        raise AssertionError("the table was read")

    monkeypatch.setattr(analyst_module, "fetch_sample_rows", never)
    ana, _ = build(db, *happy(), schema_only=True)
    assert ana.ask(QUESTION).status == "answered"


def test_the_limits_come_from_the_settings_given(db) -> None:
    ana, client = build(db, ("generate", answer(sql=TYPO)), settings=settings(max_corrections=0))
    out = ana.ask(QUESTION)
    assert out.status == "failed"
    assert client.stages == ["generate"]  # no correction was allowed


def test_in_schema_only_mode_a_correction_sends_no_cell_value(db) -> None:
    convert = "SELECT region, CAST(region AS INTEGER) AS r FROM sales"
    ana, client = build(
        db,
        ("generate", answer(sql=convert)),
        ("correct", answer(sql=FIXED)),
        ("explain", explains()),
        schema_only=True,
    )
    out = ana.ask(QUESTION)
    assert out.status == "answered"
    problem = client.calls[1].messages[-1].content
    assert "PROBLEM" in problem
    for value in ("North", "South", "East", "West"):
        assert value not in problem


def test_the_result_is_explained_in_schema_only_mode_too(db) -> None:
    ana, client = build(db, *happy(), schema_only=True)
    ana.ask(QUESTION)
    assert "North" in client.calls[1].messages[1].content  # ADR 0010: the answer needs it


def test_every_stage_of_a_question_uses_the_same_marker_code(db) -> None:
    ana, client = build(
        db, ("generate", answer(sql=TYPO)), ("correct", answer(sql=FIXED)), ("explain", explains())
    )
    ana.ask(QUESTION)
    codes = [
        set(re.findall(r"<<<(?:END )?[A-Z]+ ([0-9a-f]{12})>>>", call.messages[-1].content))
        for call in client.calls
    ]
    assert all(len(found) == 1 for found in codes)
    assert codes[0] == codes[1] == codes[2]


def test_two_analysts_do_not_share_a_marker_code(db) -> None:
    first, one = build(db, *happy())
    second, two = build(db, *happy())
    first.ask(QUESTION)
    second.ask(QUESTION)
    pattern = r"<<<QUESTION ([0-9a-f]{12})>>>"
    code_one = re.search(pattern, one.calls[0].messages[1].content)
    code_two = re.search(pattern, two.calls[0].messages[1].content)
    assert code_one and code_two
    assert code_one.group(1) != code_two.group(1)


# -- the session ----------------------------------------------------------------------------------


def test_the_regions_rules_come_from_the_settings(db, monkeypatch) -> None:
    seen: list[object] = []
    real = analyst_module.ContextBuilder

    def spy(region, *args, **kwargs):
        seen.append(region)
        return real(region, *args, **kwargs)

    monkeypatch.setattr(analyst_module, "ContextBuilder", spy)
    build(db, settings=settings(default_region="NZ"))
    assert seen == ["NZ"]


def test_asking_counts_as_activity_even_when_nothing_runs(db, monkeypatch) -> None:
    touched: list[int] = []
    monkeypatch.setattr(db, "touch", lambda: touched.append(1))
    ana, _ = build(db, ("generate", asks()), schema_only=True)  # no sample read touches it
    ana.ask("Which is best?")
    assert touched == [1]


def test_a_bad_question_is_not_activity(db, monkeypatch) -> None:
    touched: list[int] = []
    monkeypatch.setattr(db, "touch", lambda: touched.append(1))
    ana, _ = build(db)
    ana.ask("")
    assert touched == []


# -- logs -----------------------------------------------------------------------------------------


def test_the_log_line_carries_counts_and_categories_never_text(
    db, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    canary_sql = "SELECT region, SUM(revenue) AS zebra_alias FROM sales GROUP BY region"
    ana, _ = build(
        db,
        ("generate", answer(sql=canary_sql, chart=None, resolved_question="zebra restated")),
        ("explain", explains("Revenue is shown for each of the 4 regions, zebra.")),
        usage=Usage(llm_calls=1, input_tokens=100, output_tokens=50, cost_usd=0.01),
        schema_only=True,
    )
    ana.ask("ZEBRA-QUESTION about revenue")
    (record,) = [r for r in caplog.records if r.getMessage() == "question"]
    assert record.status == "answered"
    assert record.kind == "sql"
    assert record.run_status == "ok"
    assert record.corrections == 0
    assert record.attempts == 1
    assert record.llm_calls == 2
    assert (record.input_tokens, record.output_tokens) == (200, 100)
    assert record.cost_usd == pytest.approx(0.02)
    assert record.explanation == "model"
    assert record.fallback_reason is None
    assert record.chart == "bar"
    assert record.rows == 4
    assert record.truncated is False
    assert record.schema_only is True
    assert record.latency_ms > 0
    text = " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)
    for forbidden in ("zebra", "ZEBRA", "SUM(revenue)", "North", "Revenue is shown"):
        assert forbidden not in text


def test_a_failed_question_logs_its_status_and_how_far_it_got(
    db, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    ana, _ = build(db, ("generate", answer(sql="DROP TABLE sales")))
    ana.ask(QUESTION)
    (record,) = [r for r in caplog.records if r.getMessage() == "question"]
    assert (record.status, record.kind, record.run_status) == ("failed", "sql", "rejected")
