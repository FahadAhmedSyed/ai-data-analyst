"""Query executor tests (T1.09): limits, error categories and the integrity check.

Expected values are written by hand. Queries that must pass the validator go through
the real one; queries the validator would refuse (slow, forged or failing ones) are run
through hand-built ``ValidationResult`` objects, which is also how a bug or a tampered
result would reach the executor.
"""

# ruff: noqa: S608  (SQL text is built from fixed test strings, never from input)
from __future__ import annotations

import io
import json
import threading
import time

import duckdb
import pandas as pd
import pytest

from analyst.config.settings import Settings
from analyst.core.errors import QueryExecutionError, SessionCorruptedError
from analyst.core.logging import configure_logging, sql_hash
from analyst.core.models import ColumnSchema, QueryResult, ValidationResult
from analyst.database.executor import (
    ENGINE_MESSAGE_CHARS,
    QueryExecutor,
    _Deadline,
    classify_error,
)
from analyst.database.session_db import SessionDatabase
from analyst.ingestion.normalize import TypedTable
from analyst.sql.validator import SqlValidator

ROWS = 600


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def sales_table() -> TypedTable:
    ids = list(range(1, ROWS + 1))
    frame = pd.DataFrame(
        {
            "order_id": pd.array(ids, dtype="Int64"),
            "region": pd.array(
                [["North", "South", "East", "West"][i % 4] for i in ids], dtype="string"
            ),
            "revenue": pd.array([float(i * 10) for i in ids], dtype="Float64"),
            "order_date": pd.Series(
                [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in ids],
                dtype="datetime64[us]",
            ),
        }
    )
    columns = (
        ColumnSchema(sql_name="order_id", display_name="Order ID", dtype="integer"),
        ColumnSchema(sql_name="region", display_name="Region", dtype="text"),
        ColumnSchema(sql_name="revenue", display_name="Revenue", dtype="decimal"),
        ColumnSchema(sql_name="order_date", display_name="Order Date", dtype="date"),
    )
    return TypedTable(frame, columns, "sales.csv", None)


@pytest.fixture
def db():
    with SessionDatabase(settings()) as database:
        database.load(sales_table())
        yield database


def accepted(sql: str, max_rows: int = 10_000) -> ValidationResult:
    """What the validator would hand over for ``sql``, built by hand for queries it refuses."""
    return ValidationResult(status="ok", category="ok", safe_sql=sql, max_rows=max_rows)


def run(db: SessionDatabase, sql: str, **kw) -> QueryResult:
    result = SqlValidator.for_database(db, **kw).validate(sql)
    assert result.ok, result.reason
    return QueryExecutor(db).execute(result)


# ---------- results ----------


def test_runs_a_validated_query_and_returns_its_rows(db):
    result = run(
        db, "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY region"
    )
    assert list(result.dataframe.columns) == ["region", "total"]
    # Worked out by hand: region k holds the ids i with i % 4 == k, and revenue is 10 * i.
    assert result.dataframe.to_dict("records") == [
        {"region": "East", "total": 450_000.0},
        {"region": "North", "total": 453_000.0},
        {"region": "South", "total": 448_500.0},
        {"region": "West", "total": 451_500.0},
    ]
    assert result.row_count == 4
    assert result.truncated is False
    assert result.duration_ms > 0


def test_totals_match_the_data(db):
    result = run(db, "SELECT SUM(revenue) AS total, COUNT(*) AS n FROM sales")
    assert result.dataframe.iloc[0].tolist() == [sum(i * 10 for i in range(1, ROWS + 1)), ROWS]


def test_an_empty_result_is_a_result_not_an_error(db):
    result = run(db, "SELECT * FROM sales WHERE order_id < 0")
    assert (result.row_count, result.truncated, result.is_empty) == (0, False, True)
    assert list(result.dataframe.columns) == ["order_id", "region", "revenue", "order_date"]


def test_the_order_the_query_asked_for_is_kept(db):
    result = run(db, "SELECT order_id FROM sales ORDER BY revenue DESC LIMIT 5")
    assert result.dataframe["order_id"].tolist() == [600, 599, 598, 597, 596]
    assert result.dataframe.index.tolist() == [0, 1, 2, 3, 4]


def test_dates_and_missing_values_come_back_usable(db):
    result = run(db, "SELECT MIN(order_date) AS first, NULL AS nothing FROM sales")
    assert pd.Timestamp(result.dataframe["first"].iloc[0]) == pd.Timestamp("2024-01-02")
    assert pd.isna(result.dataframe["nothing"].iloc[0])


def test_division_by_zero_gives_a_value_not_an_error(db):
    result = run(db, "SELECT 1 / 0 AS x")
    assert result.dataframe["x"].iloc[0] == float("inf")


def test_the_session_is_marked_active():
    clock = {"now": 0.0}
    with SessionDatabase(settings(), clock=lambda: clock["now"]) as database:
        database.load(sales_table())
        clock["now"] = 500.0
        assert database.idle_seconds() == 500.0
        QueryExecutor(database).execute(accepted("SELECT 1"))
        assert database.idle_seconds() == 0.0


# ---------- row cap ----------


def test_a_result_over_the_cap_is_cut_and_flagged(db):
    result = run(db, "SELECT order_id FROM sales ORDER BY order_id", max_rows=5)
    assert result.row_count == 5
    assert result.truncated is True
    assert result.dataframe["order_id"].tolist() == [1, 2, 3, 4, 5]


def test_a_result_exactly_at_the_cap_is_not_flagged(db):
    result = run(db, "SELECT order_id FROM sales WHERE order_id <= 5", max_rows=5)
    assert (result.row_count, result.truncated) == (5, False)


def test_one_row_over_the_cap_is_flagged(db):
    result = run(db, "SELECT order_id FROM sales WHERE order_id <= 6", max_rows=5)
    assert (result.row_count, result.truncated) == (5, True)


def test_the_cap_comes_from_the_validation_not_from_settings(db):
    executor = QueryExecutor(db, settings(max_result_rows=3))
    validation = SqlValidator.for_database(db, max_rows=8).validate("SELECT order_id FROM sales")
    result = executor.execute(validation)
    assert (result.row_count, result.truncated) == (8, True)


def test_extra_rows_are_dropped_even_if_the_sql_returns_more_than_the_cap_allows(db):
    """A forged result whose text has no LIMIT must still not return more than the cap."""
    result = QueryExecutor(db).execute(accepted("SELECT order_id FROM sales", max_rows=10))
    assert (result.row_count, result.truncated) == (10, True)


# ---------- only validated SQL runs ----------


@pytest.mark.parametrize("not_a_result", ["SELECT 1", None, {"safe_sql": "SELECT 1"}, b"SELECT 1"])
def test_anything_but_a_validation_result_is_refused(db, not_a_result):
    with pytest.raises(TypeError):
        QueryExecutor(db).execute(not_a_result)


@pytest.mark.parametrize(
    "validation",
    [
        ValidationResult(status="rejected", category="forbidden_statement", reason="DROP"),
        ValidationResult(status="correctable", category="unknown_column", reason="x"),
        ValidationResult(status="ok", category="ok"),  # no text to run
        ValidationResult(status="ok", category="ok", safe_sql="SELECT 1"),  # no row cap recorded
    ],
)
def test_a_result_that_is_not_an_accepted_query_is_refused(db, validation):
    with pytest.raises(ValueError, match="validated"):
        QueryExecutor(db).execute(validation)


def test_the_validator_records_the_cap_it_applied(db):
    result = SqlValidator.for_database(db, max_rows=7).validate("SELECT 1")
    assert result.max_rows == 7
    assert SqlValidator.for_database(db).validate("SELECT nope FROM sales").max_rows is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE sales",
        "DROP TABLE sales",
        "DELETE FROM sales",
        "UPDATE sales SET revenue = 0",
        "CREATE TABLE copy AS SELECT * FROM sales",
        "INSERT INTO sales SELECT * FROM sales",
        "PRAGMA database_list",
        "SET memory_limit = '10GB'",
        "ATTACH 'x.db'",
        "COPY sales TO 'x.csv'",
        "EXPLAIN SELECT 1",
        "SELEC 1",
        "",
    ],
)
def test_forged_text_that_is_not_exactly_one_select_never_runs(db, sql):
    with pytest.raises(QueryExecutionError) as caught:
        QueryExecutor(db).execute(accepted(sql))
    assert caught.value.category == "other"
    assert db.query_rows("SELECT COUNT(*) FROM sales") == [(ROWS,)]
    db.verify_catalog()


def test_forged_text_cannot_read_a_file_because_the_database_is_locked_down(db, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("a\nsecret-value\n", encoding="utf-8")
    with pytest.raises(QueryExecutionError) as caught:
        QueryExecutor(db).execute(accepted(f"SELECT * FROM read_csv('{secret}')"))
    assert caught.value.category == "other"
    assert "secret-value" not in str(caught.value)


# ---------- error categories ----------


@pytest.mark.parametrize(
    ("sql", "category", "engine_type"),
    [
        ("SELECT CAST('abc' AS INTEGER)", "conversion", "ConversionException"),
        ("SELECT 'a'::DATE", "conversion", "ConversionException"),
        ("SELECT 9223372036854775807 + 1", "conversion", "OutOfRangeException"),
        ("SELECT nope FROM sales", "binder", "BinderException"),
        ("SELECT * FROM nope", "binder", "CatalogException"),
        ("SELECT strptime('x', '%Y')", "other", "InvalidInputException"),
        ("SELECT unknown_function(1)", "binder", "CatalogException"),
    ],
)
def test_database_errors_are_categorised(db, sql, category, engine_type):
    with pytest.raises(QueryExecutionError) as caught:
        QueryExecutor(db).execute(accepted(f"SELECT * FROM ({sql}) LIMIT 10"))
    assert caught.value.category == category
    assert caught.value.detail == engine_type
    assert caught.value.engine_message
    assert caught.value.user_message  # shown to the user; never the engine message


def test_the_engine_message_goes_to_the_corrector_but_not_to_the_user_or_the_log(db):
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    sql = "SELECT * FROM (SELECT CAST('SECRETVALUE' AS INTEGER)) LIMIT 5"
    with pytest.raises(QueryExecutionError) as caught:
        QueryExecutor(db).execute(accepted(sql))
    error = caught.value
    assert "SECRETVALUE" in error.engine_message  # the corrector needs the real message
    assert "SECRETVALUE" not in error.user_message
    assert "SECRETVALUE" not in error.detail
    assert "SECRETVALUE" not in str(error)
    assert "SECRETVALUE" not in stream.getvalue()
    [event] = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert event["event"] == "query_failed"
    assert event["category"] == "conversion"
    assert event["error_type"] == "ConversionException"
    assert event["sql_hash"] == sql_hash(sql)
    assert event["level"] == "WARNING"


def test_the_engine_message_is_cut_and_kept_on_one_line(db):
    long_name = "x" * 2000
    with pytest.raises(QueryExecutionError) as caught:
        QueryExecutor(db).execute(accepted(f"SELECT {long_name} FROM sales"))
    message = caught.value.engine_message
    assert len(message) <= ENGINE_MESSAGE_CHARS
    assert "\n" not in message


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (duckdb.OutOfMemoryException("x"), "out_of_memory"),
        (MemoryError(), "out_of_memory"),
        (duckdb.BinderException("x"), "binder"),
        (duckdb.CatalogException("x"), "binder"),
        (duckdb.ConversionException("x"), "conversion"),
        (duckdb.TypeMismatchException("x"), "conversion"),
        (duckdb.OutOfRangeException("x"), "conversion"),
        (duckdb.InvalidInputException("x"), "other"),
        (duckdb.IOException("x"), "other"),
        (duckdb.ParserException("x"), "other"),
        (ValueError("x"), "other"),
    ],
)
def test_error_classes_map_to_categories(error, category):
    assert classify_error(error) == category


def test_a_failure_while_building_the_dataframe_is_an_execution_error(db, monkeypatch):
    class Broken:
        def df(self):
            raise OverflowError("out of bounds")

    connection = db.connection

    class Wrapper:
        def extract_statements(self, sql):
            return connection.extract_statements(sql)

        def execute(self, sql):
            return Broken()

    monkeypatch.setattr(type(db), "connection", property(lambda self: Wrapper()))
    with pytest.raises(QueryExecutionError) as caught:
        QueryExecutor(db).execute(accepted("SELECT 1"))
    assert (caught.value.category, caught.value.detail) == ("other", "OverflowError")


# ---------- memory cap ----------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT length(list(i)) FROM range(50000000) t(i)",
        "SELECT count(*) FROM (SELECT i, count(*) AS c FROM range(20000000) t(i) GROUP BY i)",
    ],
)
def test_a_query_over_the_memory_cap_is_an_out_of_memory_error_and_the_session_survives(
    tmp_path, sql
):
    with SessionDatabase(settings(duckdb_memory_limit="50MB")) as database:
        database.load(sales_table())
        executor = QueryExecutor(database)
        with pytest.raises(QueryExecutionError) as caught:
            executor.execute(accepted(f"SELECT * FROM ({sql}) LIMIT 5"))
        assert caught.value.category == "out_of_memory"
        assert caught.value.detail == "OutOfMemoryException"
        assert list(tmp_path.rglob("*")) == []  # nothing was spilled to disk
        assert executor.execute(accepted("SELECT 1 AS one")).dataframe["one"].tolist() == [1]


# ---------- timeout ----------

SLOW = "SELECT COUNT(*) FROM range(1000000000) a, range(100) b"


def test_a_slow_query_is_stopped_at_the_timeout_and_the_session_keeps_working():
    with SessionDatabase(settings()) as database:
        database.load(sales_table())
        executor = QueryExecutor(database, settings(query_timeout_s=0.3))
        assert executor.timeout_s == 0.3
        started = time.perf_counter()
        with pytest.raises(QueryExecutionError) as caught:
            executor.execute(accepted(f"SELECT * FROM ({SLOW}) LIMIT 5"))
        elapsed = time.perf_counter() - started
        assert caught.value.category == "timeout"
        assert caught.value.detail == "InterruptException"
        assert 0.25 < elapsed < executor.timeout_s + 1  # testing.md: limit + 1 s
        assert executor.execute(accepted("SELECT 1 AS one")).row_count == 1
        assert not database.corrupted
        database.verify_catalog()


def test_a_cancel_from_another_thread_is_not_reported_as_a_timeout(db):
    executor = QueryExecutor(db, settings(query_timeout_s=60))
    threading.Timer(0.3, db.interrupt).start()
    with pytest.raises(QueryExecutionError) as caught:
        executor.execute(accepted(f"SELECT * FROM ({SLOW}) LIMIT 5"))
    assert caught.value.category == "other"
    assert caught.value.detail == "InterruptException"


def test_a_query_that_finishes_in_time_leaves_no_timer_behind(db):
    before = {t for t in threading.enumerate() if isinstance(t, threading.Timer)}
    QueryExecutor(db, settings(query_timeout_s=30)).execute(accepted("SELECT 1"))
    after = {t for t in threading.enumerate() if isinstance(t, threading.Timer)}
    assert after <= before


def test_a_late_timer_cannot_hit_the_next_query(db):
    """Deadlines at the edge of query duration: each query is ok or a timeout, nothing else."""
    executor = QueryExecutor(db, settings(query_timeout_s=0.001))
    outcomes = set()
    for _ in range(150):
        try:
            executor.execute(accepted("SELECT COUNT(*) AS n FROM sales"))
            outcomes.add("ok")
        except QueryExecutionError as error:
            assert error.category in {"timeout"}
            outcomes.add("timeout")
    assert outcomes <= {"ok", "timeout"}
    assert QueryExecutor(db).execute(accepted("SELECT COUNT(*) AS n FROM sales")).row_count == 1
    assert not db.corrupted
    db.verify_catalog()


def test_a_timer_that_fires_after_the_query_was_released_does_nothing():
    calls = []
    deadline = _Deadline(60, lambda: calls.append(1))
    deadline.finish()
    deadline._fire()
    assert calls == []
    assert deadline.fired is False


def test_the_timeout_comes_from_settings():
    with SessionDatabase(settings()) as database:
        assert QueryExecutor(database, settings(query_timeout_s=2.5)).timeout_s == 2.5
        assert QueryExecutor(database).timeout_s == 10.0


# ---------- the catalog check ----------


def test_the_catalog_is_verified_after_every_query_whether_it_works_or_fails(db, monkeypatch):
    calls = []
    original = SessionDatabase.verify_catalog

    def counting(self):
        calls.append(1)
        original(self)

    monkeypatch.setattr(SessionDatabase, "verify_catalog", counting)
    executor = QueryExecutor(db)
    executor.execute(accepted("SELECT 1"))
    assert len(calls) == 1
    with pytest.raises(QueryExecutionError):
        executor.execute(accepted("SELECT nope"))
    assert len(calls) == 2
    with pytest.raises(QueryExecutionError):
        executor.execute(accepted("DROP TABLE sales"))
    assert len(calls) == 3


def test_a_changed_table_raises_session_corrupted_instead_of_returning_the_result(db):
    db.connection.execute("UPDATE sales SET revenue = revenue + 1")
    with pytest.raises(SessionCorruptedError):
        QueryExecutor(db).execute(accepted("SELECT 1"))
    assert db.corrupted


def test_the_check_runs_even_when_the_changing_query_is_the_one_that_ran(db, monkeypatch):
    """A change made while a query runs is caught right after it, in the same call."""
    original = QueryExecutor._run

    def tamper_then_run(self, connection, sql):
        frame = original(self, connection, sql)
        connection.execute("DELETE FROM sales WHERE order_id = 1")
        return frame

    monkeypatch.setattr(QueryExecutor, "_run", tamper_then_run)
    with pytest.raises(SessionCorruptedError) as caught:
        QueryExecutor(db).execute(accepted("SELECT COUNT(*) FROM sales"))
    assert "row_count" in str(caught.value) or "content" in str(caught.value)


def test_corruption_wins_over_the_queries_own_error(db, monkeypatch):
    original = QueryExecutor._run

    def tamper_then_fail(self, connection, sql):
        connection.execute("DELETE FROM sales")
        return original(self, connection, "SELECT nope")

    monkeypatch.setattr(QueryExecutor, "_run", tamper_then_fail)
    with pytest.raises(SessionCorruptedError):
        QueryExecutor(db).execute(accepted("SELECT 1"))


def test_a_corrupted_session_runs_nothing_more(db):
    db.connection.execute("DELETE FROM sales")
    executor = QueryExecutor(db)
    with pytest.raises(SessionCorruptedError):
        executor.execute(accepted("SELECT 1"))
    with pytest.raises(SessionCorruptedError):
        executor.execute(accepted("SELECT 2"))


def test_an_empty_or_closed_session_cannot_run_a_query():
    database = SessionDatabase(settings())
    with pytest.raises(RuntimeError, match="No dataset"):
        QueryExecutor(database).execute(accepted("SELECT 1"))
    database.close()


# ---------- concurrency ----------


def test_the_session_lock_is_held_while_a_query_runs(db):
    executor = QueryExecutor(db, settings(query_timeout_s=60))
    started = threading.Event()
    seen: list[bool] = []

    def probe():
        started.wait(5)
        time.sleep(0.2)
        got = db.lock.acquire(blocking=False)
        if got:
            db.lock.release()
        seen.append(got)
        db.interrupt()

    thread = threading.Thread(target=probe)
    thread.start()
    started.set()
    with pytest.raises(QueryExecutionError):
        executor.execute(accepted(f"SELECT * FROM ({SLOW}) LIMIT 5"))
    thread.join()
    assert seen == [False]


def test_queries_from_several_threads_are_serialised_and_all_succeed(db):
    executor = QueryExecutor(db)
    validator = SqlValidator.for_database(db)
    results: list[int] = []
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            validation = validator.validate(
                f"SELECT COUNT(*) AS n FROM sales WHERE order_id <= {n}"
            )
            results.append(int(executor.execute(validation).dataframe["n"].iloc[0]))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(1, 21)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(results) == list(range(1, 21))


# ---------- logging ----------


def test_a_successful_query_logs_counts_and_a_hash_but_no_sql_or_values(db):
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    validation = SqlValidator.for_database(db, max_rows=3).validate(
        "SELECT region, revenue FROM sales WHERE region = 'North'"
    )
    QueryExecutor(db).execute(validation)
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    executed = next(e for e in events if e["event"] == "query_executed")
    assert executed["row_count"] == 3
    assert executed["truncated"] is True
    assert executed["duration_ms"] >= 0
    assert executed["sql_hash"] == sql_hash(validation.safe_sql)
    assert "North" not in stream.getvalue()
    assert "SELECT" not in stream.getvalue()
