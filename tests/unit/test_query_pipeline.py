"""Validator and executor together on the locked-down session database (M2).

The validator tests compare rewritten SQL with the original on a plain DuckDB
connection. These tests run the same corpora through the real path: validator,
then executor, on a session whose lockdown (no extension autoload, no external
access, no temp files) could in principle break a function the validator allows.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from analyst.config.settings import Settings
from analyst.core.models import ColumnSchema
from analyst.database.executor import QueryExecutor
from analyst.database.session_db import SessionDatabase
from analyst.ingestion.normalize import TypedTable
from analyst.sql.validator import SqlValidator
from tests.unit.test_validator import CASES, FUNCTION_CASES, SYNTHETIC_DDL

_TYPES = {
    "order_id": ("integer", "Int64"),
    "order_date": ("date", "datetime64[us]"),
    "region": ("text", "string"),
    "product": ("text", "string"),
    "category": ("text", "string"),
    "quantity": ("integer", "Int64"),
    "unit_price": ("decimal", "Float64"),
    "revenue": ("decimal", "Float64"),
    "customer_id": ("text", "string"),
}


def sales_table() -> tuple[TypedTable, duckdb.DuckDBPyConnection]:
    """The synthetic sales table as a loadable table, plus a plain connection to compare with."""
    plain = duckdb.connect()
    plain.execute("SET threads = 1")
    plain.execute(SYNTHETIC_DDL)
    frame = plain.execute("SELECT * FROM sales").df()
    frame = frame.astype({name: dtype for name, (_, dtype) in _TYPES.items()})
    columns = tuple(
        ColumnSchema(sql_name=name, display_name=name, dtype=kind)  # type: ignore[arg-type]
        for name, (kind, _) in _TYPES.items()
    )
    return TypedTable(frame, columns, "sales.csv", None), plain


@pytest.fixture(scope="module")
def session():
    table, plain = sales_table()
    with SessionDatabase(Settings(_env_file=None)) as database:
        database.load(table)
        yield database, SqlValidator.for_database(database), QueryExecutor(database), plain
    plain.close()


def _expected_row_count(plain: duckdb.DuckDBPyConnection, sql: str) -> int:
    return min(len(plain.execute(sql).fetchall()), 10_000)


@pytest.mark.parametrize("case", CASES["valid"], ids=[c["id"] for c in CASES["valid"]])
def test_valid_corpus_runs_end_to_end_on_the_locked_down_session(session, case):
    database, validator, executor, plain = session
    validation = validator.validate(case["sql"])
    assert validation.ok, validation.reason
    result = executor.execute(validation)
    assert result.row_count == _expected_row_count(plain, case["sql"])
    database.verify_catalog()


@pytest.mark.parametrize("name", sorted(FUNCTION_CASES))
def test_every_allowlisted_function_runs_on_the_locked_down_session(session, name):
    """No allowlisted function may need an extension or file that the lockdown removes."""
    _, validator, executor, plain = session
    sql = FUNCTION_CASES[name]
    result = executor.execute(validator.validate(sql))
    assert result.row_count == _expected_row_count(plain, sql)
    assert result.row_count > 0


def test_nothing_in_the_adversarial_corpus_reaches_the_executor_or_changes_the_table(session):
    database, validator, executor, _ = session
    executed = 0
    for case in CASES["adversarial"]:
        validation = validator.validate(case["sql"])
        assert not validation.ok, case["id"]
        with pytest.raises(ValueError, match="validated"):
            executor.execute(validation)
        executed += 1
    assert executed == len(CASES["adversarial"])
    assert database.query_rows("SELECT COUNT(*) FROM sales") == [(600,)]
    database.verify_catalog()
    frame = pd.DataFrame(database.query_df("SELECT COUNT(*) AS n FROM sales"))
    assert frame["n"].iloc[0] == 600
