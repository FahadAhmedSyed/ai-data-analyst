"""Session database tests (T1.06): loading, lockdown and the catalog integrity check.

Expected values are written by hand. DuckDB itself is the independent oracle for
the settings it reports.
"""

from __future__ import annotations

import io
import json
import threading
import time

import duckdb
import pandas as pd
import pytest

from analyst.config.settings import Settings
from analyst.core.errors import IngestionError, SessionCorruptedError
from analyst.core.logging import configure_logging
from analyst.core.models import ColumnSchema
from analyst.database import session_db
from analyst.database.session_db import SessionDatabase, default_table_name
from analyst.ingestion.normalize import TypedTable

T = pd.Timestamp


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def typed_table(source: str = "retail_sales.csv") -> TypedTable:
    """Six columns, one of each type, with a missing value in most of them."""
    frame = pd.DataFrame(
        {
            "order_id": pd.array([1, 2, 3], dtype="Int64"),
            "order_date": pd.Series(
                [T("2024-01-13"), T("2024-02-02"), pd.NaT], dtype="datetime64[us]"
            ),
            "region": pd.array(["North", None, "South"], dtype="string"),
            "revenue": pd.array([1200.5, 3.0, None], dtype="Float64"),
            "paid": pd.array([True, False, None], dtype="boolean"),
            "sent_at": pd.Series(
                [T("2024-01-13 10:30"), T("2024-01-14 11:00"), pd.NaT], dtype="datetime64[us]"
            ),
        }
    )
    columns = (
        ColumnSchema(sql_name="order_id", display_name="Order ID", dtype="integer"),
        ColumnSchema(sql_name="order_date", display_name="Order Date", dtype="date"),
        ColumnSchema(sql_name="region", display_name="Region", dtype="text"),
        ColumnSchema(sql_name="revenue", display_name="Revenue ($)", dtype="decimal"),
        ColumnSchema(sql_name="paid", display_name="Paid", dtype="boolean"),
        ColumnSchema(sql_name="sent_at", display_name="Sent At", dtype="timestamp"),
    )
    return TypedTable(frame, columns, source, None)


def other_table() -> TypedTable:
    frame = pd.DataFrame({"city": pd.array(["Wellington", "Auckland"], dtype="string")})
    columns = (ColumnSchema(sql_name="city", display_name="City", dtype="text"),)
    return TypedTable(frame, columns, "cities.csv", None)


@pytest.fixture
def db():
    with SessionDatabase(settings()) as database:
        database.load(typed_table())
        yield database


# ---------- table names ----------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("retail_sales.csv", "retail_sales"),
        ("Retail Sales (2024).xlsx", "retail_sales_2024"),
        ("my.data.v2.csv", "my_data_v2"),
        ("order.csv", "order_col"),
        ("2024 rainfall.csv", "c_2024_rainfall"),
        ("温度.csv", "col"),
        ("/some/folder/Rainfall.TXT", "rainfall"),
    ],
)
def test_default_table_name(source, expected):
    assert default_table_name(source) == expected


# ---------- loading ----------


def test_load_creates_a_typed_table(db):
    assert db.is_loaded
    assert db.table_name == "retail_sales"
    assert db.row_count == 3
    assert [c.sql_name for c in db.columns] == [
        "order_id",
        "order_date",
        "region",
        "revenue",
        "paid",
        "sent_at",
    ]
    types = db.query_df("SELECT column_name, column_type FROM (DESCRIBE retail_sales)")
    assert list(zip(types["column_name"], types["column_type"], strict=True)) == [
        ("order_id", "BIGINT"),
        ("order_date", "DATE"),
        ("region", "VARCHAR"),
        ("revenue", "DOUBLE"),
        ("paid", "BOOLEAN"),
        ("sent_at", "TIMESTAMP"),
    ]


def test_load_keeps_values_and_missing_values(db):
    result = db.query_df(
        "SELECT order_id, CAST(order_date AS VARCHAR) AS d, region, revenue, paid, "
        "CAST(sent_at AS VARCHAR) AS s FROM retail_sales ORDER BY order_id"
    )
    rows = [[None if pd.isna(v) else v for v in row] for row in result.itertuples(index=False)]
    assert rows == [
        [1, "2024-01-13", "North", 1200.5, True, "2024-01-13 10:30:00"],
        [2, "2024-02-02", None, 3.0, False, "2024-01-14 11:00:00"],
        [3, None, "South", None, None, None],
    ]


def test_load_honours_an_explicit_table_name_and_returns_it():
    with SessionDatabase(settings()) as database:
        assert database.load(typed_table(), "sales") == "sales"
        assert database.query_df("SELECT count(*) AS n FROM sales")["n"].tolist() == [3]


def test_load_does_not_leave_the_input_view_behind(db):
    names = db.query_df("SELECT table_name FROM duckdb_tables()")["table_name"].tolist()
    views = db.query_df("SELECT view_name FROM duckdb_views() WHERE NOT internal")
    assert names == ["retail_sales"]
    assert views.empty


@pytest.mark.parametrize("name", ["Order Table", "order", "2024", "a;b", 'x"y', ""])
def test_unsafe_table_names_are_rejected(name):
    if name == "":  # an empty name falls back to the file name, which is safe
        with SessionDatabase(settings()) as database:
            assert database.load(typed_table(), name) == "retail_sales"
        return
    with SessionDatabase(settings()) as database, pytest.raises(ValueError, match="not a safe"):
        database.load(typed_table(), name)
    assert not database.is_loaded


def test_columns_must_match_the_schema():
    table = typed_table()
    wrong = TypedTable(
        table.dataframe.rename(columns={"region": "area"}), table.columns, "x.csv", None
    )
    with SessionDatabase(settings()) as database, pytest.raises(ValueError, match="do not match"):
        database.load(wrong)
    empty = TypedTable(pd.DataFrame(), (), "x.csv", None)
    with SessionDatabase(settings()) as database, pytest.raises(ValueError, match="do not match"):
        database.load(empty)


def test_an_empty_table_loads_and_passes_the_check():
    frame = pd.DataFrame({"x": pd.array([], dtype="Int64")})
    table = TypedTable(
        frame, (ColumnSchema(sql_name="x", display_name="X", dtype="integer"),), "e.csv", None
    )
    with SessionDatabase(settings()) as database:
        database.load(table)
        assert database.row_count == 0
        database.verify_catalog()


def test_failed_load_leaves_no_open_database(monkeypatch):
    def boom(*_args, **_kwargs):
        raise duckdb.OutOfMemoryException("Out of Memory Error: secret-detail")

    monkeypatch.setattr(SessionDatabase, "_create_table", staticmethod(boom))
    with SessionDatabase(settings()) as database:
        with pytest.raises(IngestionError) as caught:
            database.load(typed_table())
        assert caught.value.category == "too_large"
        assert "secret-detail" not in caught.value.user_message
        assert "more memory" in caught.value.user_message
        assert not database.is_loaded


def test_load_fails_closed_if_the_lockdown_did_not_take_effect(monkeypatch):
    monkeypatch.setattr(session_db, "_EXPECTED_SETTINGS", {"enable_external_access": "true"})
    with SessionDatabase(settings()) as database:
        with pytest.raises(
            RuntimeError, match=r"lockdown did not take effect.*enable_external_access"
        ):
            database.load(typed_table())
        assert not database.is_loaded


def test_loading_again_replaces_the_dataset():
    with SessionDatabase(settings()) as database:
        database.load(typed_table())
        old_connection = database.connection
        database.load(other_table())
        assert database.table_name == "cities"
        assert database.row_count == 2
        names = database.query_df("SELECT table_name FROM duckdb_tables()")["table_name"].tolist()
        assert names == ["cities"]
        with pytest.raises(duckdb.ConnectionException):
            old_connection.execute("SELECT 1")
        with pytest.raises(duckdb.CatalogException):
            database.query_df("SELECT * FROM retail_sales")


def test_load_logs_counts_and_timing_but_no_names_or_values():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    table = typed_table("confidential_payroll.csv")
    with SessionDatabase(settings()) as database:
        database.load(table)
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    loaded = next(e for e in events if e["event"] == "table_loaded")
    assert (loaded["rows"], loaded["columns"], loaded["content_checked"]) == (3, 6, True)
    assert loaded["duration_ms"] >= 0
    for private in ("confidential", "payroll", "North", "Revenue"):
        assert private not in stream.getvalue()


# ---------- lockdown ----------

LOCKDOWN_ATTEMPTS = [
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_text('/etc/passwd')",
    "SELECT * FROM glob('/*')",
    "SELECT * FROM read_parquet('/tmp/none.parquet')",
    "SELECT * FROM 'https://example.com/x.csv'",
    "ATTACH '{tmp}/x.db'",
    "COPY (SELECT 1) TO '{tmp}/out.csv'",
    "COPY retail_sales TO '{tmp}/out.parquet' (FORMAT parquet)",
    "EXPORT DATABASE '{tmp}/export'",
    "INSTALL httpfs",
    "INSTALL httpfs FROM community",
    "LOAD httpfs",
    "SET memory_limit = '10GB'",
    "PRAGMA memory_limit = '10GB'",
    "SET threads = 64",
    "SET enable_external_access = true",
    "SET lock_configuration = false",
    "SET autoinstall_known_extensions = true",
    "SET autoload_known_extensions = true",
    "SET allow_community_extensions = true",
    "SET python_enable_replacements = true",
    "SET temp_directory = '{tmp}/spill'",
]


@pytest.mark.parametrize("statement", LOCKDOWN_ATTEMPTS)
def test_lockdown_blocks_access_attempts(db, tmp_path, statement):
    with pytest.raises(duckdb.Error):
        db.connection.execute(statement.format(tmp=tmp_path))
    assert [p.name for p in tmp_path.iterdir()] == []  # nothing was written
    db.verify_catalog()  # and the database is unchanged


def test_lockdown_applies_the_configured_limits():
    with SessionDatabase(settings(duckdb_memory_limit="256MB", duckdb_threads=3)) as database:
        database.load(typed_table())
        reported = database.connection.execute(
            "SELECT current_setting('memory_limit'), current_setting('threads')"
        ).fetchone()
    reference = duckdb.connect(":memory:")
    reference.execute("SET memory_limit = '256MB'")
    expected = reference.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    assert reported == (expected, 3)


def test_lockdown_accepts_the_binary_size_units_the_settings_normalise_to():
    # Settings upper-cases sizes, so "1GiB" arrives as "1GIB".
    assert settings(duckdb_memory_limit="1GiB").duckdb_memory_limit == "1GIB"
    with SessionDatabase(settings(duckdb_memory_limit="1GiB")) as database:
        database.load(typed_table())
        limit = database.connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    assert limit == "1.0 GiB"


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        ("1GB", "1GB"),
        ("512MB", "512MB"),
        ("1GIB", "1GiB"),
        ("0.5GIB", "0.5GiB"),
        ("64KIB", "64KiB"),
    ],
)
def test_memory_limit_spelling_for_duckdb(limit, expected):
    assert session_db._duckdb_size(limit) == expected


@pytest.mark.parametrize("limit", ["1TB", "lots", "", "10"])
def test_unsupported_memory_limits_are_refused(limit):
    with pytest.raises(ValueError, match="Unsupported memory limit"):
        session_db._duckdb_size(limit)


def _python_variable_lookup(connection: duckdb.DuckDBPyConnection) -> list[tuple]:
    secret = pd.DataFrame({"leak": ["data from another part of the process"]})  # noqa: F841
    return connection.execute("SELECT * FROM secret").fetchall()


def test_queries_cannot_read_python_variables(db):
    # Control: without lockdown DuckDB finds the local variable, so the check below means something.
    assert _python_variable_lookup(duckdb.connect(":memory:")) == [
        ("data from another part of the process",)
    ]
    with pytest.raises(duckdb.CatalogException):
        _python_variable_lookup(db.connection)


SPILLING_QUERY = (
    "SELECT count(*) FROM (SELECT i, count(*) AS c FROM range(20000000) t(i) GROUP BY i)"
)


@pytest.mark.parametrize(
    "query",
    [
        # Must hold every value at once.
        "SELECT length(list(i)) FROM range(50000000) t(i)",
        # A hash aggregate: the kind of query DuckDB would otherwise spill to disk.
        SPILLING_QUERY,
    ],
)
def test_a_query_over_the_memory_cap_fails_cleanly_and_writes_nothing(tmp_path, query):
    with SessionDatabase(settings(duckdb_memory_limit="50MB")) as database:
        database.load(typed_table())
        # The executor (T1.09) maps this to ``out_of_memory``.
        with pytest.raises(duckdb.OutOfMemoryException):
            database.connection.execute(query).fetchall()
        assert list(tmp_path.rglob("*")) == []  # no spill file and no .tmp directory
        assert database.query_df("SELECT 1 AS one")["one"].tolist() == [1]  # still usable
        database.verify_catalog()


def test_spilling_would_write_to_disk_without_the_temp_directory_setting(tmp_path):
    """Control for the test above: with only external access off, DuckDB 2.0.1 does spill."""
    connection = duckdb.connect(":memory:")
    for statement in (
        "SET enable_external_access = false",
        "SET memory_limit = '50MB'",
        "SET threads = 2",
        "SET lock_configuration = true",
    ):
        connection.execute(statement)
    with pytest.raises(duckdb.OutOfMemoryException):
        connection.execute(SPILLING_QUERY).fetchall()
    assert (tmp_path / ".tmp").is_dir()  # the leak the lockdown setting prevents
    connection.close()


def test_a_running_query_can_be_interrupted_from_another_thread(db):
    timer = threading.Timer(0.3, db.interrupt)
    timer.start()
    started = time.perf_counter()
    with pytest.raises(duckdb.InterruptException):
        db.connection.execute("SELECT count(*) FROM range(100000000000)").fetchall()
    elapsed = time.perf_counter() - started
    timer.cancel()
    assert elapsed < 5
    assert db.query_df("SELECT 1 AS one")["one"].tolist() == [1]
    db.verify_catalog()


# ---------- catalog integrity ----------


def test_an_untouched_database_passes_the_check_after_queries(db):
    db.query_df("SELECT region, sum(revenue) FROM retail_sales GROUP BY region")
    db.verify_catalog()
    db.verify_catalog()
    assert not db.corrupted


@pytest.mark.parametrize(
    ("statement", "reported"),
    [
        ("CREATE TABLE extra AS SELECT 1 AS x", "table"),
        ("CREATE TEMP TABLE scratch AS SELECT 1 AS x", "table"),
        ("CREATE VIEW v AS SELECT 1 AS x", "view"),
        ("CREATE MACRO m(x) AS x + 1", "macro"),
        ("CREATE SCHEMA s", "schema"),
        ("CREATE SEQUENCE q", "sequence"),
        ("CREATE TYPE mood AS ENUM ('a', 'b')", "type"),
        ("CREATE INDEX ix ON retail_sales (order_id)", "index"),
        ("ALTER TABLE retail_sales ADD COLUMN z INTEGER", "column"),
        ("ALTER TABLE retail_sales RENAME COLUMN region TO area", "column"),
        ("ALTER TABLE retail_sales ALTER order_id TYPE VARCHAR", "column"),
        ("ATTACH ':memory:' AS other", "database"),
        ("DELETE FROM retail_sales WHERE order_id = 1", "row_count"),
        ("INSERT INTO retail_sales VALUES (4, NULL, 'East', 1.0, TRUE, NULL)", "row_count"),
        ("UPDATE retail_sales SET region = 'West' WHERE order_id = 1", "content"),
        ("DROP TABLE retail_sales", "table"),
    ],
)
def test_the_check_detects_every_kind_of_change(db, statement, reported):
    db.connection.execute(statement)
    with pytest.raises(SessionCorruptedError) as caught:
        db.verify_catalog()
    assert caught.value.category == "catalog_changed"
    assert reported in caught.value.detail
    assert "changed" in caught.value.user_message
    assert db.corrupted


def test_a_value_swap_that_keeps_every_count_is_still_detected(db):
    db.connection.execute("UPDATE retail_sales SET revenue = 3.0 WHERE order_id = 1")
    with pytest.raises(SessionCorruptedError, match="content"):
        db.verify_catalog()


def test_deleting_one_of_two_identical_rows_is_detected():
    frame = pd.DataFrame({"x": pd.array([7, 7, 8], dtype="Int64")})
    table = TypedTable(
        frame, (ColumnSchema(sql_name="x", display_name="X", dtype="integer"),), "d.csv", None
    )
    with SessionDatabase(settings()) as database:
        database.load(table)
        database.connection.execute("DELETE FROM d WHERE x = 7")
        database.connection.execute("INSERT INTO d VALUES (9), (9)")  # same row count as before
        with pytest.raises(SessionCorruptedError, match="content"):
            database.verify_catalog()


def test_a_corrupted_session_refuses_further_use_until_reloaded(db):
    db.connection.execute("DROP TABLE retail_sales")
    with pytest.raises(SessionCorruptedError):
        db.verify_catalog()
    for use in (lambda: db.connection, lambda: db.query_df("SELECT 1"), db.verify_catalog):
        with pytest.raises(SessionCorruptedError):
            use()
    db.load(typed_table())  # uploading the file again recovers
    assert not db.corrupted
    assert db.query_df("SELECT count(*) AS n FROM retail_sales")["n"].tolist() == [3]
    db.verify_catalog()


def test_the_check_logs_what_changed_without_dataset_values():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    with SessionDatabase(settings()) as database:
        database.load(typed_table())
        database.connection.execute("UPDATE retail_sales SET region = 'West'")
        with pytest.raises(SessionCorruptedError):
            database.verify_catalog()
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    changed = next(e for e in events if e["event"] == "catalog_changed")
    assert changed["level"] == "ERROR"
    assert changed["changed"] == ["content"]
    assert "North" not in stream.getvalue()
    assert "West" not in stream.getvalue()


def test_an_unreadable_catalog_counts_as_corrupted(db, monkeypatch):
    def broken(*_args, **_kwargs):
        raise duckdb.InternalException("catalog unavailable")

    monkeypatch.setattr(SessionDatabase, "_take_snapshot", broken)
    with pytest.raises(SessionCorruptedError, match="unreadable"):
        db.verify_catalog()
    assert db.corrupted


def test_very_large_tables_skip_the_value_hash_but_still_check_structure_and_counts(monkeypatch):
    monkeypatch.setattr(session_db, "FINGERPRINT_MAX_CELLS", 5)  # the fixture has 18 cells
    with SessionDatabase(settings()) as database:
        database.load(typed_table())
        assert not database.content_checked
        database.connection.execute("UPDATE retail_sales SET region = 'West'")
        database.verify_catalog()  # documented limit: a changed value is not seen
        database.connection.execute("DELETE FROM retail_sales WHERE order_id = 1")
        with pytest.raises(SessionCorruptedError, match="row_count"):
            database.verify_catalog()


# ---------- isolation, lifetime ----------


def test_two_sessions_are_isolated_from_each_other():
    with SessionDatabase(settings()) as first, SessionDatabase(settings()) as second:
        first.load(typed_table())
        second.load(other_table())
        assert first.query_df("SELECT count(*) AS n FROM retail_sales")["n"].tolist() == [3]
        assert second.query_df("SELECT city FROM cities ORDER BY city")["city"].tolist() == [
            "Auckland",
            "Wellington",
        ]
        with pytest.raises(duckdb.CatalogException):
            first.query_df("SELECT * FROM cities")
        with pytest.raises(duckdb.CatalogException):
            second.query_df("SELECT * FROM retail_sales")
        first.connection.execute("DROP TABLE retail_sales")
        with pytest.raises(SessionCorruptedError):
            first.verify_catalog()
        second.verify_catalog()  # the other session is untouched
        assert not second.corrupted


def test_idle_expiry_follows_the_clock_and_use_resets_it():
    now = [1000.0]
    with SessionDatabase(settings(session_idle_minutes=1), clock=lambda: now[0]) as database:
        database.load(typed_table())
        assert database.idle_seconds() == 0
        now[0] += 60
        assert database.idle_seconds() == 60
        assert not database.is_expired()  # exactly the limit is still active
        now[0] += 0.5
        assert database.is_expired()
        database.query_df("SELECT 1")  # using the session resets the timer
        assert database.idle_seconds() == 0
        assert not database.is_expired()
        now[0] += 30
        database.touch()
        assert database.idle_seconds() == 0


def test_close_is_idempotent_and_forgets_the_table():
    database = SessionDatabase(settings())
    database.load(typed_table())
    database.close()
    database.close()
    assert not database.is_loaded
    for attribute in ("connection", "table_name", "columns", "row_count"):
        with pytest.raises(RuntimeError, match="No dataset"):
            getattr(database, attribute)


def test_a_new_session_has_nothing_loaded_and_interrupt_does_nothing():
    database = SessionDatabase(settings())
    assert not database.is_loaded
    assert not database.corrupted
    database.interrupt()
    with pytest.raises(RuntimeError, match="No dataset"):
        database.verify_catalog()


def test_the_context_manager_closes_the_database():
    with SessionDatabase(settings()) as database:
        database.load(typed_table())
    assert not database.is_loaded


def test_query_df_passes_parameters(db):
    result = db.query_df("SELECT region FROM retail_sales WHERE order_id = ?", [3])
    assert result["region"].tolist() == ["South"]


# ---------- end to end and size ----------


def test_a_csv_goes_through_loading_typing_and_the_session_database():
    from analyst.ingestion.csv_loader import load_csv
    from analyst.ingestion.normalize import normalize

    csv = (
        b"Order ID,Order Date,Region,Revenue ($)\r\n"
        b'1,13/01/2024,North,"$1,200.50"\r\n'
        b"2,02/02/2024,,$3.00\r\n"
        b"3,28/02/2024,South,$4.25\r\n"
    )
    typed = normalize(load_csv(csv, "Retail Sales.csv", settings=settings()), settings=settings())
    with SessionDatabase(settings()) as database:
        name = database.load(typed)
        assert name == "retail_sales"
        result = database.query_df(
            "SELECT sum(revenue) AS total, min(order_date) AS first, count(region) AS regions "
            "FROM retail_sales"
        )
        assert result["total"].tolist() == [1207.75]
        assert result["first"].tolist() == [T("2024-01-13")]
        assert result["regions"].tolist() == [2]
        database.verify_catalog()


@pytest.mark.slow
def test_a_million_row_table_loads_and_checks_quickly():
    rows = 1_000_000
    base = pd.Series(range(rows))
    frame = pd.DataFrame(
        {
            "id": base.astype("Int64"),
            "value": (base * 0.5).astype("Float64"),
            "category": ("cat " + (base % 50).astype(str)).astype("string"),
            "day": (pd.Timestamp("2020-01-01") + pd.to_timedelta(base % 1000, unit="D")).astype(
                "datetime64[us]"
            ),
        }
    )
    columns = (
        ColumnSchema(sql_name="id", display_name="id", dtype="integer"),
        ColumnSchema(sql_name="value", display_name="value", dtype="decimal"),
        ColumnSchema(sql_name="category", display_name="category", dtype="text"),
        ColumnSchema(sql_name="day", display_name="day", dtype="date"),
    )
    with SessionDatabase(settings()) as database:
        started = time.perf_counter()
        database.load(TypedTable(frame, columns, "big.csv", None))
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        database.verify_catalog()
        check_seconds = time.perf_counter() - started
        assert database.row_count == rows
        assert database.content_checked
        assert database.query_df("SELECT sum(id) AS s FROM big")["s"].tolist() == [
            rows * (rows - 1) // 2
        ]
    assert load_seconds < 10, f"load took {load_seconds:.1f}s"
    assert check_seconds < 1, f"check took {check_seconds:.2f}s"
