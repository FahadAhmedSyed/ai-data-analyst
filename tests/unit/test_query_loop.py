"""The query loop (T1.13): validate, run and correct within the question's limits.

The validator and the executor are the real ones on a real session database, so the
verdicts and the database's messages are what the model will really be sent. Failures that
cannot be produced on a small table (a timeout, running out of memory, a corrupted session)
come from a stub executor. The model is a ``FakeLLMClient``.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from analyst.config.settings import Settings
from analyst.core.errors import (
    BudgetExceededError,
    LLMError,
    QueryExecutionError,
    SessionCorruptedError,
)
from analyst.core.models import Measure, QueryResult, Usage, ValidationResult
from analyst.database.executor import QueryExecutor
from analyst.database.session_db import SessionDatabase
from analyst.llm.client import CallBudget, Message
from analyst.llm.fake import FakeLLMClient
from analyst.pipeline.corrector import Corrector
from analyst.pipeline.generator import Generation, unknown_columns
from analyst.pipeline.query_loop import Attempt, QueryLoop
from analyst.sql.context import ContextBuilder
from analyst.sql.validator import SqlValidator
from tests.unit.test_context import CODE, INTENT, PROFILE
from tests.unit.test_executor import sales_table
from tests.unit.test_generator import SQL, answer

OPENING = (Message("system", "SYSTEM"), Message("user", "USER"))
FIXED = "SELECT region, SUM(revenue) AS total_revenue FROM sales GROUP BY region ORDER BY 1"
TYPO = "SELECT region, SUM(revnue) AS total_revenue FROM sales GROUP BY region"
TYPO_2 = "SELECT regon, SUM(revenue) AS total_revenue FROM sales GROUP BY regon"
CONVERT = "SELECT region, CAST(region AS INTEGER) AS r FROM sales"
ONE_CALL = Usage(llm_calls=1, input_tokens=100, output_tokens=50)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.fixture(scope="module")
def db():
    with SessionDatabase(settings()) as database:
        database.load(sales_table())
        yield database


class StubExecutor:
    """An executor that does what the test says: raise the error or return the result."""

    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self.executed: list[str | None] = []

    def execute(self, validation: ValidationResult) -> QueryResult:
        self.executed.append(validation.safe_sql)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def rows() -> QueryResult:
    return QueryResult(dataframe=pd.DataFrame({"n": [1]}), row_count=1)


def failure(category: str, message: str = "") -> QueryExecutionError:
    return QueryExecutionError(category, "detail", engine_message=message)  # type: ignore[arg-type]


def make(db, *script, executor=None, **kw) -> tuple[QueryLoop, FakeLLMClient]:
    client = FakeLLMClient(list(script))
    corrector = Corrector(client, ContextBuilder("NZ", code=CODE))
    loop_settings = kw.pop("settings", settings())
    loop = QueryLoop(
        SqlValidator.for_database(db),
        executor or QueryExecutor(db, settings()),
        corrector,
        settings=loop_settings,
        **kw,
    )
    return loop, client


def generation(sql: str = SQL, **changes) -> Generation:
    result = answer(sql=sql, **changes)
    return Generation(
        kind="sql",
        result=result,
        sql=sql,
        message="",
        unknown_columns=unknown_columns(result.intent, PROFILE),
        clarification_overridden=False,
        opening=OPENING,
        usage=Usage(),
        attempts=1,
        approx_tokens=10,
    )


def run(loop: QueryLoop, gen: Generation | None = None, **kw):
    return loop.run(gen or generation(), PROFILE, **kw)


# -- a query that works ---------------------------------------------------------------------------


def test_a_good_query_runs_without_the_model(db) -> None:
    loop, client = make(db)
    first = generation()
    out = run(loop, first)
    assert out.ok
    assert out.status == "ok"
    assert out.message == ""
    assert out.sql == SQL
    assert out.result is not None
    assert out.result.row_count == 4
    assert list(out.result.dataframe.columns) == ["region", "total_revenue"]
    assert out.attempts == (Attempt(SQL, "execution", "ok"),)
    assert out.corrections == 0
    assert out.usage == Usage()
    assert out.generation is first.result
    assert client.calls == []


def test_a_generation_that_holds_no_sql_cannot_be_run(db) -> None:
    loop, _ = make(db)
    asking = Generation(**{**generation().__dict__, "kind": "clarification", "sql": None})
    with pytest.raises(ValueError, match="holds SQL"):
        run(loop, asking)
    refused = Generation(**{**generation().__dict__, "kind": "refused"})  # SQL is still set
    with pytest.raises(ValueError, match="holds SQL"):
        run(loop, refused)
    empty = Generation(**{**generation().__dict__, "sql": None})
    with pytest.raises(ValueError, match="holds SQL"):
        run(loop, empty)


# -- a fixable validation problem -----------------------------------------------------------------


def test_an_unknown_column_is_corrected_and_the_fixed_query_runs(db) -> None:
    fixed = answer(
        sql=FIXED,
        resolved_question="Revenue by region, sorted by region.",
        assumptions=["Sorted by region."],
    )
    loop, client = make(db, fixed)
    out = run(loop, generation(TYPO))
    assert out.status == "ok"
    assert out.sql == FIXED
    assert out.result is not None
    assert out.result.row_count == 4
    assert out.attempts == (
        Attempt(TYPO, "validation", "unknown_column"),
        Attempt(FIXED, "execution", "ok"),
    )
    assert out.corrections == 1
    assert out.usage == ONE_CALL
    assert out.generation == fixed  # the corrected answer is the one in force
    (call,) = client.calls
    assert call.stage == "correct"
    problem = call.messages[-1].content
    assert "Unknown column 'revnue'. Did you mean: revenue?" in problem
    assert call.messages[:2] == OPENING


def test_a_second_correction_continues_the_same_conversation(db) -> None:
    loop, client = make(db, answer(sql=TYPO_2), answer(sql=FIXED))
    out = run(loop, generation(TYPO))
    assert out.status == "ok"
    assert out.corrections == 2
    assert out.usage == ONE_CALL + ONE_CALL
    assert [a.category for a in out.attempts] == ["unknown_column", "unknown_column", "ok"]
    sent = client.calls[1].messages
    assert [m.role for m in sent] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert sent[:4] == client.calls[0].messages
    assert "Unknown column 'regon'" in sent[5].content


def test_corrections_stop_at_the_limit_and_the_query_is_not_run(db) -> None:
    loop, client = make(db, answer(sql=TYPO_2), answer(sql=TYPO), answer(sql=FIXED))
    out = run(loop, generation(TYPO))
    assert out.status == "unfixed"
    assert out.ok is False
    assert out.result is None
    assert out.message == "The query couldn't be fixed within the allowed number of attempts."
    assert out.corrections == 2
    assert len(client.calls) == 2
    assert client.remaining == 1  # the third answer was never asked for
    assert out.sql == TYPO  # the last query tried
    assert len(out.attempts) == 3
    assert out.usage == ONE_CALL + ONE_CALL


def test_the_limit_comes_from_the_settings(db) -> None:
    one, client = make(db, answer(sql=TYPO_2), settings=settings(max_corrections=1))
    out = run(one, generation(TYPO))
    assert out.status == "unfixed"
    assert out.corrections == 1
    assert len(client.calls) == 1
    none, client = make(db, settings=settings(max_corrections=0))
    out = run(none, generation(TYPO))
    assert out.status == "unfixed"
    assert out.corrections == 0
    assert client.calls == []


def test_the_default_limit_is_two_corrections(db) -> None:
    client = FakeLLMClient([answer(sql=TYPO_2), answer(sql=TYPO), answer(sql=FIXED)])
    loop = QueryLoop(
        SqlValidator.for_database(db),
        QueryExecutor(db, settings()),
        Corrector(client, ContextBuilder("NZ", code=CODE)),
    )
    out = run(loop, generation(TYPO))
    assert (out.status, out.corrections, len(client.calls)) == ("unfixed", 2, 2)


# -- queries that are never corrected -------------------------------------------------------------

REJECTED = [
    ("DROP TABLE sales", "forbidden_statement", "This tool only reads data"),
    ("SELECT getenv('HOME')", "disallowed_function", "a function that this tool does not allow"),
    ("SELECT * FROM read_csv('x.csv')", "table_function", "something other than the uploaded"),
    ('SELECT * FROM "a b"', "forbidden_table", "something other than the uploaded"),
    ("SELECT 1; SELECT 2", "multiple_statements", "not a single read-only query"),
    ("SELEC FROM", "parse_error", "could not be read"),
]


@pytest.mark.parametrize(("sql", "category", "wording"), REJECTED)
def test_a_rejected_query_ends_the_turn_and_is_never_sent_to_the_model(
    db, sql: str, category: str, wording: str
) -> None:
    loop, client = make(db, answer(sql=FIXED))
    out = run(loop, generation(sql))
    assert out.status == "rejected"
    assert out.result is None
    assert wording in out.message
    assert out.attempts == (Attempt(sql, "validation", category),)
    assert out.corrections == 0
    assert client.calls == []
    assert client.remaining == 1
    assert sql not in out.message


class StubValidator:
    """A validator that always gives the verdict it was made with."""

    def __init__(self, category: str) -> None:
        self._category = category

    def validate(self, sql: str) -> ValidationResult:
        return ValidationResult(status="rejected", category=self._category, reason="x")  # type: ignore[arg-type]


def refusing(db, category: str) -> QueryLoop:
    return QueryLoop(
        StubValidator(category),  # type: ignore[arg-type]
        QueryExecutor(db, settings()),
        Corrector(FakeLLMClient()),
        settings=settings(),
    )


def test_a_query_nested_too_deeply_to_check_is_rejected_with_its_own_message(db) -> None:
    out = run(refusing(db, "too_complex"))
    assert out.status == "rejected"
    assert "too large to check safely" in out.message


def test_a_category_without_its_own_message_gets_the_general_one(db) -> None:
    out = run(refusing(db, "unknown_table"))
    assert out.message == "The query was refused by the safety checks, so it was not run."


def test_a_query_that_is_merely_too_long_is_corrected(db) -> None:
    long_sql = SQL + " " * 10_001
    loop, client = make(db, answer(sql=FIXED))
    out = run(loop, generation(long_sql))
    assert out.status == "ok"
    assert out.attempts[0] == Attempt(long_sql, "validation", "too_complex")
    assert "too large or too deeply nested" in client.calls[0].messages[-1].content


def test_a_correction_that_turns_out_to_be_rejected_ends_the_turn(db) -> None:
    loop, client = make(db, answer(sql="DROP TABLE sales"), answer(sql=FIXED))
    out = run(loop, generation(TYPO))
    assert out.status == "rejected"
    assert out.corrections == 1
    assert out.sql == "DROP TABLE sales"
    assert [a.category for a in out.attempts] == ["unknown_column", "forbidden_statement"]
    assert len(client.calls) == 1
    assert out.usage == ONE_CALL


# -- failures when the query runs -----------------------------------------------------------------


def test_a_conversion_failure_is_corrected_with_the_database_message_and_types(db) -> None:
    loop, client = make(db, answer(sql=FIXED))
    out = run(loop, generation(CONVERT))
    assert out.status == "ok"
    assert [a.category for a in out.attempts] == ["conversion", "ok"]
    assert out.attempts[0].stage == "execution"
    sent = client.calls[0].messages[-1].content
    assert "The database could not convert a value to the type the query needed." in sent
    assert "Could not convert string '" in sent
    assert "Columns the query uses, with their types: region (text)." in sent


def test_a_type_mismatch_the_validator_cannot_see_is_a_binder_failure(db) -> None:
    loop, client = make(db, answer(sql=FIXED))
    out = run(loop, generation("SELECT SUM(region) AS s FROM sales"))
    assert out.status == "ok"
    assert out.attempts[0].category == "binder"
    sent = client.calls[0].messages[-1].content
    assert "No function matches" in sent
    assert "region (text)" in sent


def test_in_schema_only_mode_no_cell_value_reaches_the_model(db) -> None:
    loop, client = make(db, answer(sql=FIXED))
    out = run(loop, generation(CONVERT), schema_only=True)
    assert out.status == "ok"
    sent = client.calls[0].messages[-1].content
    for value in ("North", "South", "East", "West"):
        assert value not in sent
    assert "Could not convert" not in sent  # withheld; the title and the types remain
    assert "region (text)" in sent


def test_an_unfixed_run_failure_tells_the_user_what_failed(db) -> None:
    loop, _ = make(db, answer(sql=CONVERT), answer(sql=CONVERT))
    out = run(loop, generation(CONVERT))
    assert out.status == "unfixed"
    assert out.message == "The query tried to treat a value as the wrong type."
    assert out.corrections == 2


def test_a_later_validation_failure_replaces_the_run_failure_text(db) -> None:
    loop, _ = make(db, answer(sql=TYPO), answer(sql=TYPO))
    out = run(loop, generation(CONVERT))
    assert out.status == "unfixed"
    assert out.message == "The query couldn't be fixed within the allowed number of attempts."


@pytest.mark.parametrize("category", ["timeout", "out_of_memory"])
def test_a_heavy_query_is_corrected_once(db, category: str) -> None:
    stub = StubExecutor(failure(category), failure(category), rows())
    loop, client = make(db, answer(sql=FIXED), answer(sql=FIXED), executor=stub)
    out = run(loop, generation())
    assert out.status == "unfixed"
    assert out.corrections == 1  # the limit is two, but a second try would only make them wait
    assert len(client.calls) == 1
    assert client.remaining == 1
    assert out.message == QueryExecutionError(category).user_message  # type: ignore[arg-type]
    assert "simpler query" in client.calls[0].messages[-1].content


def test_a_timeout_then_memory_exhaustion_is_still_one_heavy_failure(db) -> None:
    stub = StubExecutor(failure("timeout"), failure("out_of_memory"))
    loop, client = make(db, answer(sql=FIXED), executor=stub)
    out = run(loop)
    assert out.status == "unfixed"
    assert out.corrections == 1
    assert out.message == "The query needed more memory than is available."
    assert len(client.calls) == 1


def test_a_heavy_failure_does_not_use_up_the_other_correction(db) -> None:
    stub = StubExecutor(failure("timeout"), failure("binder", "Binder Error: x"), rows())
    loop, client = make(db, answer(sql=FIXED), answer(sql=FIXED), executor=stub)
    out = run(loop)
    assert out.status == "ok"
    assert out.corrections == 2
    assert [a.category for a in out.attempts] == ["timeout", "binder", "ok"]
    assert len(client.calls) == 2


def test_a_heavy_failure_stays_used_up_after_another_kind_of_correction(db) -> None:
    stub = StubExecutor(
        failure("timeout"), failure("binder", "Binder Error: x"), failure("timeout")
    )
    loop, client = make(
        db,
        answer(sql=FIXED),
        answer(sql=FIXED),
        answer(sql=FIXED),
        executor=stub,
        settings=settings(max_corrections=3),
    )
    out = run(loop)
    assert out.status == "unfixed"
    assert out.corrections == 2  # the third try would be a second correction for a heavy query
    assert out.message == "The query took too long and was stopped."
    assert len(client.calls) == 2


def test_a_first_timeout_is_corrected_and_the_simpler_query_runs(db) -> None:
    stub = StubExecutor(failure("timeout"), rows())
    loop, client = make(db, answer(sql=FIXED), executor=stub)
    out = run(loop)
    assert out.status == "ok"
    assert stub.executed[1] is not None
    assert "FROM sales" in stub.executed[1]
    assert len(client.calls) == 1


def test_another_execution_failure_is_corrected_with_its_message(db) -> None:
    stub = StubExecutor(failure("other", "Out of Range Error: overflow"), rows())
    loop, client = make(db, answer(sql=FIXED), executor=stub)
    out = run(loop)
    assert out.status == "ok"
    assert out.attempts[0].category == "other"
    assert "Out of Range Error: overflow" in client.calls[0].messages[-1].content


def test_a_corrupted_session_is_not_caught(db) -> None:
    stub = StubExecutor(SessionCorruptedError("changed"))
    loop, client = make(db, executor=stub)
    with pytest.raises(SessionCorruptedError):
        run(loop)
    assert client.calls == []


# -- the model's side -----------------------------------------------------------------------------


def test_a_correction_with_no_sql_ends_the_turn_with_the_models_reason(db) -> None:
    giving_up = answer(sql=None, explanation="The table has no discount column.", chart=None)
    loop, client = make(db, giving_up)
    out = run(loop, generation(TYPO))
    assert out.status == "gave_up"
    assert out.message == "The table has no discount column."
    assert out.result is None
    assert out.sql == TYPO  # the last query tried; there is no newer one
    assert out.generation == giving_up
    assert out.corrections == 1
    assert out.usage == ONE_CALL
    assert len(client.calls) == 1


def test_a_failed_correction_call_ends_the_turn_and_counts_its_cost(db) -> None:
    boom = LLMError("timeout", "slow", usage=Usage(llm_calls=1, input_tokens=7))
    loop, _ = make(db, boom, answer(sql=FIXED))
    out = run(loop, generation(TYPO))
    assert out.status == "llm_error"
    assert out.message == "The language model took too long to respond. Please try again."
    assert out.usage == Usage(llm_calls=1, input_tokens=7)
    assert out.corrections == 1
    assert out.sql == TYPO


def test_a_failed_call_without_usage_costs_nothing(db) -> None:
    loop, _ = make(db, LLMError("rate_limited", "429"))
    out = run(loop, generation(TYPO))
    assert out.status == "llm_error"
    assert out.usage == Usage()


def test_an_unusable_correction_is_asked_for_again_while_corrections_remain(db) -> None:
    unusable = LLMError("invalid_output", "no", usage=Usage(llm_calls=1, input_tokens=5))
    loop, client = make(db, unusable, answer(sql=FIXED))
    out = run(loop, generation(TYPO))
    assert out.status == "ok"
    assert out.corrections == 2  # the unusable answer used one
    assert out.usage == Usage(llm_calls=2, input_tokens=105, output_tokens=50)
    assert client.calls[1].messages == client.calls[0].messages  # the same request again
    assert [a.category for a in out.attempts] == ["unknown_column", "ok"]  # not checked twice


def test_an_unusable_correction_does_not_run_the_same_query_again(db) -> None:
    stub = StubExecutor(failure("binder", "Binder Error: x"), rows())
    unusable = LLMError("invalid_output", "no")
    loop, _ = make(db, unusable, answer(sql=FIXED), executor=stub)
    out = run(loop)
    assert out.status == "ok"
    assert len(stub.executed) == 2  # the failing query once, the fixed one once
    assert [a.category for a in out.attempts] == ["binder", "ok"]


def test_two_unusable_corrections_leave_the_query_unfixed(db) -> None:
    unusable = LLMError("invalid_output", "no", usage=Usage(llm_calls=1, input_tokens=5))
    loop, client = make(db, unusable, unusable, answer(sql=FIXED))
    out = run(loop, generation(TYPO))
    assert out.status == "unfixed"
    assert out.corrections == 2
    assert out.usage == Usage(llm_calls=2, input_tokens=10)
    assert client.remaining == 1


# -- the call budget ------------------------------------------------------------------------------


def test_corrections_leave_one_call_for_the_explainer(db) -> None:
    budget = CallBudget(3)
    budget.charge()  # the generator's call
    loop, client = make(db, answer(sql=TYPO_2), answer(sql=FIXED))
    out = run(loop, generation(TYPO), budget=budget)
    assert out.status == "budget"
    assert out.message == "This question needed more steps than allowed. Try asking it more simply."
    assert out.corrections == 1  # one was made, the next would have taken the last call
    assert budget.used == 2
    assert budget.remaining == 1
    assert len(client.calls) == 1


def test_a_budget_with_room_is_charged_for_each_correction(db) -> None:
    budget = CallBudget(5)
    budget.charge()
    loop, _ = make(db, answer(sql=TYPO_2), answer(sql=FIXED))
    out = run(loop, generation(TYPO), budget=budget)
    assert out.status == "ok"
    assert budget.used == 3


def test_the_reserve_can_be_changed(db) -> None:
    budget = CallBudget(2)
    budget.charge()
    loop, _ = make(db, answer(sql=FIXED), reserve_calls=0)
    out = run(loop, generation(TYPO), budget=budget)
    assert out.status == "ok"
    assert budget.used == 2
    blocked, client = make(db, answer(sql=FIXED), reserve_calls=1)
    other = CallBudget(2)
    other.charge()
    assert run(blocked, generation(TYPO), budget=other).status == "budget"
    assert client.calls == []


def test_a_spent_budget_cannot_be_overspent_by_a_correction(db) -> None:
    budget = CallBudget(1)
    budget.charge()
    loop, client = make(db, answer(sql=FIXED), reserve_calls=0)
    out = run(loop, generation(TYPO), budget=budget)
    assert out.status == "budget"
    assert client.calls == []
    assert BudgetExceededError("llm_calls").user_message == out.message


# -- the intent -----------------------------------------------------------------------------------

BAD_INTENT = INTENT.model_copy(update={"measures": [Measure(column="revnue", aggregation="sum")]})


def test_an_intent_with_unknown_columns_is_corrected_before_the_query_is_checked(db) -> None:
    fixed = answer(sql=FIXED)
    loop, client = make(db, fixed)
    out = run(loop, generation(intent=BAD_INTENT))
    assert out.status == "ok"
    assert out.attempts == (
        Attempt(SQL, "intent", "intent_columns"),
        Attempt(FIXED, "execution", "ok"),
    )
    assert out.generation == fixed
    assert out.corrections == 1
    sent = client.calls[0].messages[-1].content
    assert "The intent names columns that are not in the table." in sent
    assert "revnue (did you mean revenue?)" in sent


def test_an_intent_problem_and_a_query_problem_share_the_correction_limit(db) -> None:
    loop, client = make(db, answer(sql=TYPO), answer(sql=TYPO_2), answer(sql=FIXED))
    out = run(loop, generation(intent=BAD_INTENT))
    assert out.status == "unfixed"
    assert out.corrections == 2
    assert [(a.stage, a.category) for a in out.attempts] == [
        ("intent", "intent_columns"),
        ("validation", "unknown_column"),
        ("validation", "unknown_column"),
    ]
    assert len(client.calls) == 2


def test_the_intent_is_not_checked_again_after_its_correction(db) -> None:
    still_wrong = answer(sql=FIXED, intent=BAD_INTENT)
    loop, client = make(db, still_wrong)
    out = run(loop, generation(intent=BAD_INTENT))
    assert out.status == "ok"
    assert out.corrections == 1
    assert len(client.calls) == 1


def test_an_intent_that_cannot_be_corrected_does_not_stop_a_good_query(db) -> None:
    loop, client = make(db, settings=settings(max_corrections=0))
    out = run(loop, generation(intent=BAD_INTENT))
    assert out.status == "ok"
    assert out.attempts == (  # the problem is on record, and the query ran as written
        Attempt(SQL, "intent", "intent_columns"),
        Attempt(SQL, "execution", "ok"),
    )
    assert client.calls == []


def test_an_intent_is_left_alone_when_no_call_can_be_spared(db) -> None:
    budget = CallBudget(2)
    budget.charge()
    loop, client = make(db, answer(sql=FIXED))
    out = run(loop, generation(intent=BAD_INTENT), budget=budget)
    assert out.status == "ok"
    assert out.sql == SQL
    assert out.corrections == 0
    assert client.calls == []


@pytest.mark.parametrize("category", ["timeout", "api_error", "invalid_output"])
def test_a_failed_intent_correction_does_not_stop_a_good_query(db, category: str) -> None:
    error = LLMError(category, "x", usage=Usage(llm_calls=1, input_tokens=9))  # type: ignore[arg-type]
    loop, client = make(db, error)
    out = run(loop, generation(intent=BAD_INTENT))
    assert out.status == "ok"
    assert out.sql == SQL
    assert out.corrections == 1
    assert out.usage == Usage(llm_calls=1, input_tokens=9)
    assert len(client.calls) == 1  # not asked again


def test_an_intent_correction_with_no_sql_ends_the_turn_with_the_models_reason(db) -> None:
    giving_up = answer(sql=None, explanation="There is no revnue column.", chart=None)
    loop, _ = make(db, giving_up)
    out = run(loop, generation(intent=BAD_INTENT))
    assert out.status == "gave_up"
    assert out.message == "There is no revnue column."


# -- logs -----------------------------------------------------------------------------------------


def test_logs_carry_categories_counts_and_hashes_never_text(
    db, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    loop, _ = make(db, answer(sql=FIXED))
    run(loop, generation(CONVERT))
    attempts = [r for r in caplog.records if r.getMessage() == "query_attempt"]
    assert [(r.stage, r.category) for r in attempts] == [
        ("execution", "conversion"),
        ("execution", "ok"),
    ]
    assert attempts[0].sql_hash != attempts[1].sql_hash
    assert len(attempts[0].sql_hash) == 12
    (final,) = [r for r in caplog.records if r.getMessage() == "query_run"]
    assert (final.status, final.attempts, final.corrections, final.last_category) == (
        "ok",
        2,
        1,
        "ok",
    )
    text = " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)
    for forbidden in ("CAST(", "SUM(", "North", "South", "Could not convert", "region"):
        assert forbidden not in text


def test_a_rejection_is_logged_with_its_category_and_without_the_query(db, caplog) -> None:
    caplog.set_level(logging.INFO)
    loop, _ = make(db, settings=settings(max_corrections=0))
    run(loop, generation(intent=BAD_INTENT, sql="DROP TABLE secret_table"))
    (final,) = [r for r in caplog.records if r.getMessage() == "query_run"]
    assert final.status == "rejected"
    assert final.last_category == "forbidden_statement"
    assert "secret_table" not in " ".join(f"{r.getMessage()} {vars(r)}" for r in caplog.records)
