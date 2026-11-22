"""SQL validator tests (T1.08).

Expected verdicts come from docs/security.md section 4 and are written by hand in
``sql_cases.yaml``. DuckDB is the independent oracle for what a query means: every
accepted query is run twice on synthetic data, once as written and once as the
validator rewrote it, and the rows must match.
"""

# ruff: noqa: S105, S608  (SQL text is built from fixed test strings, never from input)
from __future__ import annotations

import io
import json
import logging
import re
import threading
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from analyst.config.settings import Settings
from analyst.core.logging import configure_logging, sql_hash
from analyst.core.models import ColumnSchema
from analyst.database.session_db import SessionDatabase
from analyst.ingestion.normalize import TypedTable
from analyst.sql import validator as validator_module
from analyst.sql.validator import ALLOWED_FUNCTION_NAMES, SqlValidator

ROOT = Path(__file__).resolve().parents[2]
CASES = yaml.safe_load((Path(__file__).parent / "sql_cases.yaml").read_text(encoding="utf-8"))
SEED_CASES = yaml.safe_load((ROOT / "eval" / "cases" / "seed_cases.yaml").read_text("utf-8"))

# Column names from eval/README.md, section 1. The synthetic tables below must match.
SCHEMAS: dict[str, list[str]] = {
    "sales": [
        "order_id",
        "order_date",
        "region",
        "product",
        "category",
        "quantity",
        "unit_price",
        "revenue",
        "customer_id",
    ],
    "survey": [
        "respondent_id",
        "submitted_at",
        "age_group",
        "channel",
        "satisfaction",
        "likelihood_to_recommend",
        "comment",
    ],
    "rainfall": ["date", "site", "rainfall_mm", "max_temp_c", "min_temp_c"],
    "water_quality": ["sample_id", "site", "sampled_at", "parameter", "result", "unit", "lab"],
    "finance": ["month", "department", "budget", "actual"],
    "incidents": ["incident_id", "reported_on", "site", "severity", "resolved_on", "cost"],
}

SYNTHETIC_DDL = """
CREATE TABLE sales AS SELECT
    i AS order_id,
    DATE '2023-01-01' + CAST(i % 700 AS INTEGER) AS order_date,
    CASE WHEN i % 17 = 0 THEN NULL
         ELSE (['North', 'South', 'East', 'West'])[1 + i % 4] END AS region,
    'Product ' || CAST(i % 20 AS VARCHAR) AS product,
    (['Electronics', 'Furniture', 'Office'])[1 + i % 3] AS category,
    CAST(1 + i % 7 AS INTEGER) AS quantity,
    CAST(10 + (i * 37) % 1500 AS DOUBLE) AS unit_price,
    CAST((1 + i % 7) * (10 + (i * 37) % 1500) AS DOUBLE) AS revenue,
    'C' || CAST(i % 300 AS VARCHAR) AS customer_id
FROM range(1, 601) t(i);

CREATE TABLE survey AS SELECT
    i AS respondent_id,
    TIMESTAMP '2024-01-01 08:00:00' + to_minutes(CAST(i * 977 AS INTEGER)) AS submitted_at,
    (['18-24', '25-34', '35-44', '45-54', '55+'])[1 + i % 5] AS age_group,
    (['Online', 'Phone', 'In store'])[1 + i % 3] AS channel,
    CASE WHEN i % 13 = 0 THEN NULL ELSE CAST(1 + i % 5 AS INTEGER) END AS satisfaction,
    CAST(i % 11 AS INTEGER) AS likelihood_to_recommend,
    CASE WHEN i % 4 = 0 THEN 'ok' ELSE NULL END AS comment
FROM range(1, 301) t(i);

CREATE TABLE rainfall AS SELECT
    DATE '2020-01-01' + CAST(d AS INTEGER) AS date,
    'Site ' || chr(CAST(65 + s AS INTEGER)) AS site,
    CASE WHEN (d * 7 + s) % 40 = 0 THEN NULL
         ELSE ROUND(((d * 13 + s * 7) % 60) / 2.0, 1) END AS rainfall_mm,
    ROUND(15 + (d % 20) + s, 1) AS max_temp_c,
    ROUND(5 + (d % 10) + s, 1) AS min_temp_c
FROM range(0, 1827) a(d), range(0, 6) b(s);

CREATE TABLE water_quality AS SELECT
    'S' || CAST(s AS VARCHAR) AS sample_id,
    'Site ' || chr(CAST(65 + s % 6 AS INTEGER)) AS site,
    TIMESTAMP '2024-01-01 09:00:00' + to_hours(CAST(s * 11 AS INTEGER)) AS sampled_at,
    (['E. coli', 'Nitrate-N', 'Dissolved oxygen', 'Turbidity'])[1 + p] AS parameter,
    CASE WHEN (s + p) % 9 = 0 THEN '<0.5'
         ELSE CAST(ROUND(((s * 3 + p) % 50) / 4.0, 2) AS VARCHAR) END AS result,
    (['cfu/100mL', 'mg/L', 'mg/L', 'NTU'])[1 + p] AS unit,
    'Lab ' || CAST(s % 3 AS VARCHAR) AS lab
FROM range(0, 200) a(s), range(0, 4) b(p);

CREATE TABLE finance AS SELECT
    CAST(DATE '2023-01-01' + to_months(CAST(m AS INTEGER)) AS DATE) AS month,
    (['Ops', 'HR', 'IT', 'Sales'])[1 + d] AS department,
    CAST(1000 + m * 10 + d * 100 AS DOUBLE) AS budget,
    (1000 + m * 10 + d * 100) * (0.9 + ((m + d) % 5) / 20.0) AS actual
FROM range(0, 24) a(m), range(0, 4) b(d);

CREATE TABLE incidents AS SELECT
    i AS incident_id,
    DATE '2023-01-01' + CAST((i * 3) % 700 AS INTEGER) AS reported_on,
    'Site ' || chr(CAST(65 + i % 6 AS INTEGER)) AS site,
    (['Low', 'Medium', 'High'])[1 + i % 3] AS severity,
    CASE WHEN i % 5 = 0 THEN NULL
         ELSE DATE '2023-01-01' + CAST((i * 3) % 700 + 1 + i % 9 AS INTEGER) END AS resolved_on,
    ROUND(100 + (i * 17) % 900, 2) AS cost
FROM range(1, 301) t(i);
"""


# ---------- helpers ----------


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    """Synthetic copies of the benchmark tables. One thread keeps float sums repeatable."""
    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute(SYNTHETIC_DDL)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def sales_validator() -> SqlValidator:
    return SqlValidator({"sales": SCHEMAS["sales"]})


def _norm(value: object) -> object:
    if isinstance(value, (float, Decimal)):
        return round(float(value), 9)
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    return value


def rows(connection: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    """Result rows in a stable order, floats rounded, so two runs can be compared."""
    result = [tuple(_norm(v) for v in row) for row in connection.execute(sql).fetchall()]
    return sorted(result, key=repr)


def assert_same_rows(
    connection: duckdb.DuckDBPyConnection, original: str, validated: str, *, label: str
) -> None:
    expected = rows(connection, original)
    assert rows(connection, validated) == expected, f"{label}: rewritten SQL returns other rows"
    assert expected, f"{label}: the synthetic data gives an empty result, which proves nothing"


def split_verdict(expect: str) -> tuple[str, str]:
    status, category = expect.split("/")
    return status, category


def ids(cases: list[dict]) -> list[str]:
    return [case["id"] for case in cases]


# ---------- adversarial corpus ----------


def test_the_adversarial_corpus_is_large_enough_and_covers_every_category():
    cases = CASES["adversarial"]
    assert len(cases) >= 60
    assert len({case["id"] for case in cases}) == len(cases)
    seen = {case["expect"] for case in cases}
    assert {
        "rejected/multiple_statements",
        "rejected/forbidden_statement",
        "rejected/forbidden_table",
        "rejected/table_function",
        "rejected/disallowed_function",
        "rejected/parse_error",
        "correctable/disallowed_function",
        "correctable/unknown_table",
        "correctable/unknown_column",
        "correctable/unsupported_syntax",
        "correctable/too_complex",
    } <= seen


@pytest.mark.parametrize("case", CASES["adversarial"], ids=ids(CASES["adversarial"]))
def test_adversarial_query_gets_the_documented_verdict(sales_validator, case):
    result = sales_validator.validate(case["sql"])
    status, category = split_verdict(case["expect"])
    assert (result.status, result.category) == (status, category), result.reason
    assert not result.ok
    assert result.safe_sql is None
    assert result.reason


def test_the_reason_never_repeats_the_model_s_text_beyond_short_identifiers(sales_validator):
    secret = "TOPSECRET-VALUE-1234567890"
    for sql in (
        f"SELECT nope FROM sales WHERE region = '{secret}'",
        f"SELECT * FROM '{secret}/x.csv'",
        f"SELECT {secret.replace('-', '_').lower()}(1)",
        f"SELECT * FROM sales WHERE region = '{secret}'; DROP TABLE sales",
    ):
        reason = sales_validator.validate(sql).reason
        assert secret not in reason
        assert secret.lower() not in reason.lower()


def test_long_or_odd_names_are_cut_and_made_printable_in_reasons(sales_validator):
    reason = sales_validator.validate(f"SELECT {'a' * 200}(1) FROM sales").reason
    assert "a" * 41 not in reason
    assert "..." in reason
    assert validator_module._safe_name("x\ny\t;--'\"") == "x?y?;--'\"".replace(";", "?").replace(
        "-", "?"
    ).replace("'", "?").replace('"', "?")


def test_unknown_column_reason_suggests_the_nearest_real_column(sales_validator):
    result = sales_validator.validate("SELECT revenu FROM sales")
    assert "revenue" in result.reason
    assert "'revenu'" in result.reason


def test_an_ambiguous_column_is_reported_as_ambiguous_not_unknown(sales_validator):
    result = sales_validator.validate(
        "SELECT region FROM sales a JOIN sales b ON a.order_id = b.order_id"
    )
    assert result.category == "unknown_column"
    assert "ambiguous" in result.reason
    assert "Did you mean" not in result.reason


def test_unknown_table_reason_lists_the_real_table(sales_validator):
    reason = sales_validator.validate("SELECT * FROM weather").reason
    assert "'weather'" in reason
    assert "sales" in reason


def test_unknown_function_reason_suggests_an_allowed_name_when_the_model_typed_a_near_miss(
    sales_validator,
):
    reason = sales_validator.validate("SELECT stddev_sample(revenue) FROM sales").reason
    assert "'stddev_sample'" in reason
    assert "Did you mean 'stddev_samp'" in reason
    assert "Did you mean" not in sales_validator.validate("SELECT strlen(region) FROM sales").reason


# ---------- valid corpus ----------


def test_the_valid_corpus_is_large_enough():
    cases = CASES["valid"]
    assert len(cases) >= 40
    assert len({case["id"] for case in cases}) == len(cases)


@pytest.mark.parametrize("case", CASES["valid"], ids=ids(CASES["valid"]))
def test_valid_query_is_accepted_and_the_rewritten_text_returns_the_same_rows(
    sales_validator, con, case
):
    result = sales_validator.validate(case["sql"])
    assert result.ok, f"{result.status}/{result.category}: {result.reason}"
    assert (result.status, result.category, result.reason) == ("ok", "ok", "")
    assert result.safe_sql.startswith("SELECT * FROM (")
    assert result.safe_sql.endswith(") LIMIT 10001")
    assert_same_rows(con, case["sql"], result.safe_sql, label=case["id"])


def test_comments_do_not_reach_the_text_that_runs(sales_validator):
    result = sales_validator.validate(
        "SELECT COUNT(*) /* ; DROP TABLE sales */ FROM sales -- ; DROP TABLE sales"
    )
    assert result.safe_sql == "SELECT * FROM (SELECT COUNT(*) FROM sales) LIMIT 10001"


def test_the_text_that_runs_is_regenerated_from_the_checked_tree(sales_validator):
    """Odd spelling and layout in, canonical SQL out: what runs is what was checked."""
    result = sales_validator.validate(
        "  select   Region ,\n SUM( Revenue )as total\nfrom   SALES\ngroup by 1 ;  "
    )
    assert result.safe_sql == (
        "SELECT * FROM (SELECT Region, SUM(Revenue) AS total FROM SALES GROUP BY 1) LIMIT 10001"
    )


# ---------- gold queries ----------


def gold_queries() -> list[tuple[str, str, str]]:
    found = []
    for case in SEED_CASES["cases"]:
        for key in ("gold_sql", "alt_gold_sql"):
            if case.get(key):
                found.append((f"{case['id']}-{key[:3]}", case["dataset"], case[key]))
    return found


def test_there_are_at_least_37_gold_queries():
    assert len(gold_queries()) >= 37


def test_the_synthetic_tables_have_the_columns_the_benchmark_documents(con):
    for table, columns in SCHEMAS.items():
        actual = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
        assert actual == columns, table


@pytest.mark.parametrize(
    ("label", "dataset", "sql"), gold_queries(), ids=[g[0] for g in gold_queries()]
)
def test_every_gold_query_is_accepted_and_means_the_same_after_rewriting(con, label, dataset, sql):
    result = SqlValidator({dataset: SCHEMAS[dataset]}).validate(sql)
    assert result.ok, f"{label}: {result.status}/{result.category}: {result.reason}"
    assert_same_rows(con, sql, result.safe_sql, label=label)


def test_a_gold_query_is_refused_against_the_wrong_dataset():
    sql = next(s for label, dataset, s in gold_queries() if label.startswith("S01"))
    result = SqlValidator({"rainfall": SCHEMAS["rainfall"]}).validate(sql)
    assert (result.status, result.category) == ("correctable", "unknown_table")


# ---------- one test per allowlisted function ----------

_W = "FROM sales WHERE order_id <= 20"
FUNCTION_CASES: dict[str, str] = {
    # aggregates
    "count": f"SELECT COUNT(revenue) {_W}",
    "sum": f"SELECT SUM(revenue) {_W}",
    "avg": f"SELECT AVG(revenue) {_W}",
    "mean": f"SELECT MEAN(revenue) {_W}",
    "min": f"SELECT MIN(revenue) {_W}",
    "max": f"SELECT MAX(revenue) {_W}",
    "median": f"SELECT MEDIAN(revenue) {_W}",
    "mode": f"SELECT MODE(category) {_W}",
    "stddev": f"SELECT STDDEV(revenue) {_W}",
    "stddev_samp": f"SELECT STDDEV_SAMP(revenue) {_W}",
    "stddev_pop": f"SELECT STDDEV_POP(revenue) {_W}",
    "variance": f"SELECT VARIANCE(revenue) {_W}",
    "var_samp": f"SELECT VAR_SAMP(revenue) {_W}",
    "var_pop": f"SELECT VAR_POP(revenue) {_W}",
    "quantile_cont": f"SELECT QUANTILE_CONT(revenue, 0.25) {_W}",
    "quantile_disc": f"SELECT QUANTILE_DISC(revenue, 0.75) {_W}",
    "approx_count_distinct": f"SELECT APPROX_COUNT_DISTINCT(customer_id) {_W}",
    "count_if": f"SELECT COUNT_IF(revenue > 100) {_W}",
    "bool_and": f"SELECT BOOL_AND(revenue > 0) {_W}",
    "bool_or": f"SELECT BOOL_OR(revenue > 1000) {_W}",
    "first": f"SELECT FIRST(order_id ORDER BY order_id) {_W}",
    "last": f"SELECT LAST(order_id ORDER BY order_id) {_W}",
    "arg_min": f"SELECT ARG_MIN(order_id, revenue) {_W}",
    "arg_max": f"SELECT ARG_MAX(order_id, revenue) {_W}",
    "corr": f"SELECT CORR(unit_price, quantity) {_W}",
    "covar_pop": f"SELECT COVAR_POP(unit_price, quantity) {_W}",
    "covar_samp": f"SELECT COVAR_SAMP(unit_price, quantity) {_W}",
    "string_agg": f"SELECT STRING_AGG(category, ',' ORDER BY order_id) {_W}",
    "list": f"SELECT LIST(order_id ORDER BY order_id) {_W}",
    # window
    "row_number": f"SELECT order_id, ROW_NUMBER() OVER (ORDER BY order_id) {_W}",
    "rank": f"SELECT order_id, RANK() OVER (ORDER BY quantity) {_W}",
    "dense_rank": f"SELECT order_id, DENSE_RANK() OVER (ORDER BY quantity) {_W}",
    "percent_rank": f"SELECT order_id, PERCENT_RANK() OVER (ORDER BY quantity) {_W}",
    "cume_dist": f"SELECT order_id, CUME_DIST() OVER (ORDER BY quantity) {_W}",
    "ntile": f"SELECT order_id, NTILE(4) OVER (ORDER BY order_id) {_W}",
    "lag": f"SELECT order_id, LAG(revenue) OVER (ORDER BY order_id) {_W}",
    "lead": f"SELECT order_id, LEAD(revenue) OVER (ORDER BY order_id) {_W}",
    "first_value": f"SELECT order_id, FIRST_VALUE(revenue) OVER (ORDER BY order_id) {_W}",
    "last_value": f"SELECT order_id, LAST_VALUE(revenue) OVER (ORDER BY order_id) {_W}",
    "nth_value": f"SELECT order_id, NTH_VALUE(revenue, 2) OVER (ORDER BY order_id) {_W}",
    # math
    "abs": f"SELECT order_id, ABS(quantity - 4) {_W}",
    "round": f"SELECT order_id, ROUND(unit_price / 7, 2) {_W}",
    "floor": f"SELECT order_id, FLOOR(unit_price / 7) {_W}",
    "ceil": f"SELECT order_id, CEIL(unit_price / 7) {_W}",
    "sqrt": f"SELECT order_id, SQRT(unit_price) {_W}",
    "power": f"SELECT order_id, POWER(quantity, 3) {_W}",
    "ln": f"SELECT order_id, LN(unit_price) {_W}",
    "log": f"SELECT order_id, LOG(unit_price) {_W}",
    "log10": f"SELECT order_id, LOG10(unit_price) {_W}",
    "exp": f"SELECT order_id, EXP(quantity) {_W}",
    "sign": f"SELECT order_id, SIGN(quantity - 4) {_W}",
    "greatest": f"SELECT order_id, GREATEST(quantity, 3) {_W}",
    "least": f"SELECT order_id, LEAST(quantity, 3) {_W}",
    # conditional and null
    "coalesce": f"SELECT order_id, COALESCE(region, 'none') {_W}",
    "nullif": f"SELECT order_id, NULLIF(quantity, 3) {_W}",
    "ifnull": f"SELECT order_id, IFNULL(region, 'none') {_W}",
    "if": f"SELECT order_id, IF(quantity > 3, 'big', 'small') {_W}",
    "case": f"SELECT order_id, CASE WHEN quantity > 3 THEN 'big' ELSE 'small' END {_W}",
    # casting
    "cast": f"SELECT order_id, CAST(unit_price AS INTEGER), unit_price::VARCHAR {_W}",
    "try_cast": f"SELECT order_id, TRY_CAST(product AS INTEGER) {_W}",
    # text
    "lower": f"SELECT order_id, LOWER(product) {_W}",
    "upper": f"SELECT order_id, UPPER(product) {_W}",
    "trim": f"SELECT order_id, TRIM('  ' || product || '  ') {_W}",
    "ltrim": f"SELECT order_id, LTRIM('  ' || product) {_W}",
    "rtrim": f"SELECT order_id, RTRIM(product || '  ') {_W}",
    "length": f"SELECT order_id, LENGTH(product) {_W}",
    "substring": f"SELECT order_id, SUBSTRING(product, 2, 4) {_W}",
    "replace": f"SELECT order_id, REPLACE(product, 'Product', 'P') {_W}",
    "concat": f"SELECT order_id, CONCAT(product, '-', category) {_W}",
    "starts_with": f"SELECT order_id, STARTS_WITH(product, 'Prod') {_W}",
    "ends_with": f"SELECT order_id, ENDS_WITH(product, '1') {_W}",
    "contains": f"SELECT order_id, CONTAINS(product, 'uct') {_W}",
    "regexp_matches": f"SELECT order_id, REGEXP_MATCHES(product, '[0-9]$') {_W}",
    "split_part": f"SELECT order_id, SPLIT_PART(product, ' ', 2) {_W}",
    "left": f"SELECT order_id, LEFT(product, 4) {_W}",
    "right": f"SELECT order_id, RIGHT(product, 2) {_W}",
    # date and time
    "date_trunc": f"SELECT order_id, DATE_TRUNC('month', order_date) {_W}",
    "date_part": f"SELECT order_id, DATE_PART('year', order_date) {_W}",
    "extract": f"SELECT order_id, EXTRACT(month FROM order_date) {_W}",
    "year": f"SELECT order_id, YEAR(order_date) {_W}",
    "quarter": f"SELECT order_id, QUARTER(order_date) {_W}",
    "month": f"SELECT order_id, MONTH(order_date) {_W}",
    "week": f"SELECT order_id, WEEK(order_date) {_W}",
    "day": f"SELECT order_id, DAY(order_date) {_W}",
    "dayofweek": f"SELECT order_id, DAYOFWEEK(order_date) {_W}",
    "dayofyear": f"SELECT order_id, DAYOFYEAR(order_date) {_W}",
    "hour": f"SELECT order_id, HOUR(CAST(order_date AS TIMESTAMP) + INTERVAL 5 HOUR) {_W}",
    "strftime": f"SELECT order_id, STRFTIME(order_date, '%d/%m/%Y') {_W}",
    "strptime": f"SELECT order_id, STRPTIME(CAST(order_date AS VARCHAR), '%Y-%m-%d') {_W}",
    "make_date": f"SELECT order_id, MAKE_DATE(2024, quantity, 1) {_W}",
    "date_diff": f"SELECT order_id, DATE_DIFF('day', order_date, DATE '2024-06-01') {_W}",
    "date_add": f"SELECT order_id, DATE_ADD(order_date, INTERVAL 3 DAY) {_W}",
    "date_sub": f"SELECT order_id, DATE_SUB('day', order_date, DATE '2024-06-01') {_W}",
    "age": f"SELECT order_id, AGE(DATE '2024-06-01', order_date) {_W}",
    "current_date": f"SELECT order_id, CURRENT_DATE > DATE '2020-01-01' {_W}",
    "epoch": f"SELECT order_id, EPOCH(CAST(order_date AS TIMESTAMP)) {_W}",
    "last_day": f"SELECT order_id, LAST_DAY(order_date) {_W}",
}


def test_there_is_one_test_case_for_every_allowlisted_function():
    assert set(FUNCTION_CASES) == set(ALLOWED_FUNCTION_NAMES)


def test_the_validator_allowlist_is_the_one_in_the_security_document():
    text = (ROOT / "docs" / "security.md").read_text(encoding="utf-8")
    section = text.split("### Function allowlist", 1)[1].split("## 5.", 1)[0]
    documented = {
        name
        for line in section.splitlines()
        if line.startswith("- ")
        for name in re.findall(r"`([a-z_0-9]+)`", line)
    }
    assert documented == set(ALLOWED_FUNCTION_NAMES)


@pytest.mark.parametrize("name", sorted(FUNCTION_CASES))
def test_allowlisted_function_is_accepted_and_keeps_its_meaning(sales_validator, con, name):
    sql = FUNCTION_CASES[name]
    result = sales_validator.validate(sql)
    assert result.ok, f"{name}: {result.status}/{result.category}: {result.reason}"
    assert_same_rows(con, sql, result.safe_sql, label=name)


@pytest.mark.parametrize(
    "expression",
    [
        "CEILING(revenue)",
        "SUBSTR(product, 1, 2)",
        "LEN(product)",
        "UCASE(product)",
        "LCASE(product)",
        "ARRAY_AGG(order_id ORDER BY order_id)",
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revenue)",
        "DATETRUNC('month', order_date)",
        "GROUP_CONCAT(product)",
        "LOG2(quantity + 1)",
    ],
)
def test_other_spellings_of_allowlisted_functions_are_accepted_and_keep_their_meaning(
    sales_validator, con, expression
):
    sql = f"SELECT {expression} FROM sales WHERE order_id <= 20"
    result = sales_validator.validate(sql)
    assert result.ok, result.reason
    assert_same_rows(con, sql, result.safe_sql, label=expression)


def test_a_spelling_duckdb_lacks_is_translated_so_the_validated_query_runs(sales_validator, con):
    sql = "SELECT NVL(region, 'none') FROM sales WHERE order_id <= 20"
    with pytest.raises(duckdb.CatalogException):
        con.execute(sql)
    result = sales_validator.validate(sql)
    assert result.ok
    assert len(con.execute(result.safe_sql).fetchall()) == 20


@pytest.mark.parametrize(
    "name",
    [
        "getenv('X')",
        "current_setting('threads')",
        "read_text('/etc/passwd')",
        "pg_sleep(1)",
        "duckdb_extensions()",
        "pragma_database_list()",
        "write_csv('x')",
        "sqlite_version()",
        "query_table('sales')",
    ],
)
def test_functions_that_touch_the_environment_are_rejected_not_corrected(sales_validator, name):
    result = sales_validator.validate(f"SELECT {name}")
    assert (result.status, result.category) == ("rejected", "disallowed_function")


# ---------- V11: the row cap ----------


def test_the_query_is_wrapped_with_one_more_row_than_the_cap(sales_validator):
    result = sales_validator.validate("SELECT COUNT(*) FROM sales")
    assert result.safe_sql == "SELECT * FROM (SELECT COUNT(*) FROM sales) LIMIT 10001"


def test_the_cap_is_configurable_and_must_be_positive():
    small = SqlValidator({"sales": SCHEMAS["sales"]}, max_rows=5)
    assert small.max_rows == 5
    assert small.validate("SELECT 1").safe_sql == "SELECT * FROM (SELECT 1) LIMIT 6"
    with pytest.raises(ValueError, match="at least 1"):
        SqlValidator({}, max_rows=0)


def test_the_cap_returns_one_extra_row_so_truncation_can_be_detected(con):
    validator = SqlValidator({"sales": SCHEMAS["sales"]}, max_rows=5)
    result = validator.validate("SELECT order_id FROM sales")
    assert len(con.execute(result.safe_sql).fetchall()) == 6
    result = validator.validate("SELECT order_id FROM sales WHERE order_id <= 3")
    assert len(con.execute(result.safe_sql).fetchall()) == 3


def test_a_smaller_limit_written_by_the_model_still_applies(con):
    validator = SqlValidator({"sales": SCHEMAS["sales"]}, max_rows=100)
    result = validator.validate("SELECT order_id FROM sales LIMIT 2")
    assert len(con.execute(result.safe_sql).fetchall()) == 2


def test_wrapping_keeps_the_order_the_model_asked_for():
    """The cap must not scramble ORDER BY, even with several threads and many rows."""
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(
        "CREATE TABLE big AS SELECT i AS id, (i * 7919) % 100003 AS v FROM range(300000) t(i)"
    )
    validator = SqlValidator({"big": ["id", "v"]}, max_rows=50)
    result = validator.validate("SELECT id, v FROM big ORDER BY v DESC, id")
    got = connection.execute(result.safe_sql).fetchall()
    want = connection.execute("SELECT id, v FROM big ORDER BY v DESC, id LIMIT 51").fetchall()
    assert got == want
    connection.close()


def test_duplicate_output_names_survive_the_wrapper(con):
    validator = SqlValidator({"sales": SCHEMAS["sales"]})
    result = validator.validate(
        "SELECT a.order_id, b.order_id FROM sales a JOIN sales b ON a.order_id = b.order_id"
    )
    assert result.ok
    frame = con.execute(result.safe_sql).fetchdf()
    assert frame.shape == (600, 2)


# ---------- V10: joins and size limits ----------


@pytest.mark.parametrize(
    "sql",
    [
        # one side is a single row
        "SELECT * FROM sales a CROSS JOIN (SELECT MAX(revenue) AS m FROM sales) b",
        "SELECT * FROM (SELECT MAX(revenue) AS m FROM sales) b CROSS JOIN sales a",
        "SELECT * FROM sales a, (SELECT 1 AS one) b",
        "WITH one AS (SELECT MIN(order_id) AS lo FROM sales) SELECT * FROM sales, one",
        "WITH one AS (SELECT MIN(order_id) AS lo FROM sales), two AS (SELECT * FROM one) "
        "SELECT * FROM sales, two",
        # both sides are known to be small
        "SELECT * FROM (SELECT region FROM sales GROUP BY region LIMIT 4) a "
        "CROSS JOIN (SELECT product FROM sales GROUP BY product LIMIT 20) b",
        "WITH a AS (SELECT region FROM sales LIMIT 100), "
        "b AS (SELECT product FROM sales LIMIT 100) SELECT * FROM a, b",
        "SELECT * FROM (SELECT 1 AS x UNION ALL SELECT 2) a "
        "CROSS JOIN (SELECT 3 AS y UNION ALL SELECT 4 LIMIT 10) b",
        "SELECT * FROM (SELECT 1 AS x EXCEPT SELECT 2 AS x) a CROSS JOIN sales b",
        "SELECT * FROM (SELECT 1 AS x INTERSECT SELECT 1 AS x) a, sales b",
    ],
)
def test_a_join_without_a_condition_is_allowed_only_between_small_inputs(sales_validator, sql):
    assert sales_validator.validate(sql).ok


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM sales a CROSS JOIN sales b",
        "SELECT * FROM sales a, (SELECT * FROM sales LIMIT 5) b",
        "SELECT * FROM (SELECT * FROM sales LIMIT 1001) a "
        "CROSS JOIN (SELECT * FROM sales LIMIT 2) b",
        "SELECT * FROM sales a CROSS JOIN (SELECT region FROM sales GROUP BY region) b",
        "SELECT * FROM sales a JOIN sales b ON a.order_id = b.order_id CROSS JOIN sales c",
        "WITH big AS (SELECT * FROM sales), two AS (SELECT * FROM big) SELECT * FROM sales, two",
        "SELECT * FROM (SELECT region FROM sales EXCEPT SELECT product FROM sales) a, sales b",
        "SELECT * FROM (SELECT 1 AS x UNION ALL SELECT region FROM sales) a CROSS JOIN sales b",
        "WITH t AS (SELECT region FROM sales LIMIT 5000) SELECT * FROM sales, (SELECT * FROM t) u",
    ],
)
def test_a_join_without_a_condition_over_a_large_input_is_refused(sales_validator, sql):
    result = sales_validator.validate(sql)
    assert (result.status, result.category) == ("correctable", "too_complex")


def test_four_joins_are_allowed_and_five_are_not(sales_validator):
    def joins(count: int) -> str:
        parts = " ".join(
            f"JOIN sales t{i} ON t0.order_id = t{i}.order_id" for i in range(1, count + 1)
        )
        return f"SELECT COUNT(*) FROM sales t0 {parts}"

    assert sales_validator.validate(joins(4)).ok
    five = sales_validator.validate(joins(5))
    assert (five.status, five.category) == ("correctable", "too_complex")
    assert "At most 4" in five.reason


def test_joins_inside_subqueries_count_toward_the_limit(sales_validator):
    inner = (
        "SELECT t0.order_id FROM sales t0 JOIN sales t1 ON t0.order_id = t1.order_id "
        "JOIN sales t2 ON t0.order_id = t2.order_id JOIN sales t3 ON t0.order_id = t3.order_id"
    )
    sql = (
        f"SELECT * FROM ({inner}) x JOIN sales y ON x.order_id = y.order_id "
        "JOIN sales z ON x.order_id = z.order_id"
    )
    assert sales_validator.validate(sql).category == "too_complex"


def test_a_very_long_query_goes_back_to_the_corrector(sales_validator):
    """One long string is a handful of parts, so only the length rule can catch it."""
    sql = "SELECT 1 FROM sales WHERE region = '" + "x" * validator_module.MAX_SQL_CHARS + "'"
    result = sales_validator.validate(sql)
    assert (result.status, result.category) == ("correctable", "too_complex")
    assert "characters" in result.reason
    assert sales_validator.validate("SELECT 1 FROM sales WHERE region = 'x'").ok


def test_deep_nesting_goes_back_to_the_corrector(sales_validator):
    sql = "SELECT 1 FROM sales"
    for _ in range(20):
        sql = f"SELECT * FROM ({sql})"
    result = sales_validator.validate(sql)
    assert (result.status, result.category) == ("correctable", "too_complex")


def test_pathological_nesting_is_refused_without_a_crash(sales_validator):
    sql = "SELECT " + "(" * 3000 + "1" + ")" * 3000
    result = sales_validator.validate(sql)
    assert not result.ok
    assert result.category == "too_complex"


def test_a_query_with_too_many_parts_goes_back_to_the_corrector(sales_validator, monkeypatch):
    monkeypatch.setattr(validator_module, "MAX_NODES", 30)
    result = sales_validator.validate(
        "SELECT " + ", ".join(f"order_id + {i}" for i in range(20)) + " FROM sales"
    )
    assert (result.status, result.category) == ("correctable", "too_complex")
    assert sales_validator.validate("SELECT order_id FROM sales").ok


# ---------- case, schema and construction ----------


def test_table_and_column_names_are_compared_without_regard_to_case():
    validator = SqlValidator({"Retail_Sales": ["Order_ID", "Region"]})
    assert validator.validate('SELECT order_id, REGION FROM "RETAIL_SALES"').ok
    assert validator.validate("SELECT nope FROM retail_sales").category == "unknown_column"


def test_a_cte_may_share_a_name_with_nothing_else_in_the_query(sales_validator):
    assert sales_validator.validate("WITH weather AS (SELECT 1 AS x) SELECT x FROM weather").ok
    assert sales_validator.validate("SELECT * FROM weather").category == "unknown_table"


def test_a_session_table_whose_name_looks_like_a_system_table_is_still_usable():
    validator = SqlValidator({"pg_data": ["a"]})
    assert validator.validate("SELECT a FROM pg_data").ok
    assert validator.validate("SELECT * FROM pg_tables").category == "forbidden_table"


def test_the_validator_can_be_built_from_a_loaded_session_database():
    frame = pd.DataFrame(
        {"city": pd.array(["Auckland"], dtype="string"), "pop": pd.array([1], dtype="Int64")}
    )
    columns = (
        ColumnSchema(sql_name="city", display_name="City", dtype="text"),
        ColumnSchema(sql_name="pop", display_name="Pop", dtype="integer"),
    )
    with SessionDatabase(Settings(_env_file=None)) as database:
        database.load(TypedTable(frame, columns, "Cities.csv", None))
        validator = SqlValidator.for_database(database, max_rows=7)
        result = validator.validate("SELECT city FROM cities WHERE pop > 0")
        assert result.safe_sql == "SELECT * FROM (SELECT city FROM cities WHERE pop > 0) LIMIT 8"
        assert database.query_rows(result.safe_sql) == [("Auckland",)]
        assert validator.validate("SELECT nope FROM cities").category == "unknown_column"
        assert validator.validate("SELECT * FROM retail").category == "unknown_table"


def test_non_text_input_is_refused_not_raised(sales_validator):
    for value in (None, 42, b"SELECT 1", ["SELECT 1"]):
        result = sales_validator.validate(value)
        assert (result.status, result.category) == ("rejected", "parse_error")


def test_validation_is_repeatable_and_safe_across_threads(sales_validator):
    queries = [case["sql"] for case in CASES["valid"][:20]] + [
        case["sql"] for case in CASES["adversarial"][:20]
    ]
    expected = [sales_validator.validate(q).model_dump() for q in queries]
    failures: list[str] = []

    def work() -> None:
        for _ in range(5):
            for query, want in zip(queries, expected, strict=True):
                if sales_validator.validate(query).model_dump() != want:
                    failures.append(query)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


# ---------- fuses: the checks that back up the checks ----------


def test_the_generated_text_is_checked_again_and_a_bad_rewrite_is_refused(
    sales_validator, monkeypatch
):
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    monkeypatch.setattr(
        SqlValidator, "_generate", lambda self, tree: "SELECT * FROM sales; DROP TABLE sales"
    )
    result = sales_validator.validate("SELECT 1")
    assert (result.status, result.category) == ("rejected", "multiple_statements")
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert any(e["event"] == "validator_fuse_failed" and e["level"] == "ERROR" for e in events)


def test_a_rewrite_that_introduces_a_forbidden_function_is_refused(sales_validator, monkeypatch):
    monkeypatch.setattr(
        SqlValidator, "_generate", lambda self, tree: "SELECT * FROM (SELECT getenv('X')) LIMIT 5"
    )
    result = sales_validator.validate("SELECT 1")
    assert (result.status, result.category) == ("rejected", "disallowed_function")


def test_a_rewrite_that_is_not_a_select_is_refused(sales_validator, monkeypatch):
    monkeypatch.setattr(SqlValidator, "_generate", lambda self, tree: "DROP TABLE sales")
    result = sales_validator.validate("SELECT 1")
    assert result.status == "rejected"


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (None, ("rejected", "parse_error")),
        (["SELECT", "DROP"], ("rejected", "multiple_statements")),
        ([], ("rejected", "multiple_statements")),
        (["PRAGMA"], ("rejected", "forbidden_statement")),
        (["COPY"], ("rejected", "forbidden_statement")),
    ],
)
def test_duckdbs_own_parser_has_the_last_word(sales_validator, monkeypatch, types, expected):
    monkeypatch.setattr(validator_module._duckdb_parser, "statement_types", lambda sql: types)
    result = sales_validator.validate("SELECT 1")
    assert (result.status, result.category) == expected


def test_duckdb_parser_reports_statement_types_and_survives_garbage():
    parser = validator_module._DuckDBParser()
    assert parser.statement_types("SELECT 1") == ["SELECT"]
    assert parser.statement_types("WITH t AS (SELECT 1) SELECT * FROM t") == ["SELECT"]
    assert parser.statement_types("SELECT 1; DROP TABLE x") == ["SELECT", "DROP"]
    assert parser.statement_types("SELEC 1") is None
    assert parser.statement_types("") == []


def test_the_parse_only_connection_cannot_read_files(tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n", encoding="utf-8")
    parser = validator_module._DuckDBParser()
    # Statements are only parsed, never run; check the settings anyway.
    assert parser.statement_types(f"SELECT * FROM '{secret}'") == ["SELECT"]
    connection = parser._connection
    setting = connection.execute(
        "SELECT value FROM duckdb_settings() WHERE name = 'enable_external_access'"
    ).fetchone()[0]
    assert str(setting).lower() == "false"


def test_a_failure_inside_the_validator_refuses_the_query(sales_validator, monkeypatch):
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)

    def boom(*args, **kwargs):
        raise KeyError("internal")

    monkeypatch.setattr(validator_module, "qualify", boom)
    result = sales_validator.validate("SELECT region FROM sales")
    assert (result.status, result.category) == ("rejected", "parse_error")
    assert any(
        json.loads(line)["event"] == "column_check_failed"
        for line in stream.getvalue().splitlines()
    )


def test_an_unrecognised_column_error_still_goes_back_to_the_corrector(
    sales_validator, monkeypatch
):
    def odd(*args, **kwargs):
        raise validator_module.OptimizeError("something new")

    monkeypatch.setattr(validator_module, "qualify", odd)
    result = sales_validator.validate("SELECT region FROM sales")
    assert (result.status, result.category) == ("correctable", "unknown_column")
    assert "schema" in result.reason


def test_text_the_generator_produced_but_sqlglot_cannot_read_is_refused(
    sales_validator, monkeypatch
):
    monkeypatch.setattr(SqlValidator, "_generate", lambda self, tree: "SELECT (")
    result = sales_validator.validate("SELECT 1")
    assert (result.status, result.category) == ("rejected", "parse_error")


def test_the_size_bound_of_something_that_is_not_a_query_is_unknown():
    from sqlglot import exp

    check = validator_module._TreeCheck({})
    assert check._query_bound(exp.Values(), frozenset()) is None


def test_an_unexpected_error_anywhere_refuses_the_query(sales_validator, monkeypatch):
    monkeypatch.setattr(SqlValidator, "_check_tree", lambda self, tree: 1 / 0)
    result = sales_validator.validate("SELECT 1")
    assert (result.status, result.category) == ("rejected", "parse_error")


def test_a_recursion_error_while_parsing_is_a_refusal(sales_validator, monkeypatch):
    def recurse(*args, **kwargs):
        raise RecursionError

    monkeypatch.setattr(validator_module.sqlglot, "parse", recurse)
    result = sales_validator.validate("SELECT 1")
    assert (result.status, result.category) == ("rejected", "too_complex")


# ---------- allowlist hygiene ----------


def test_every_class_name_on_the_allowlists_exists_in_sqlglot():
    # Import resolves them; this documents that a SQLGlot upgrade that removes one fails loudly.
    assert len(validator_module._SYNTAX) == len(validator_module._SYNTAX_NAMES)
    assert len(validator_module._FUNCTIONS) == len(validator_module._FUNCTION_NAMES)


def test_dangerous_classes_are_never_on_the_allowlist():
    from sqlglot import exp

    allowed = validator_module._SYNTAX | validator_module._FUNCTIONS
    for name in (
        "ReadCSV",
        "Anonymous",
        "Command",
        "Pragma",
        "Set",
        "Copy",
        "Attach",
        "Drop",
        "Insert",
        "Update",
        "Delete",
        "Create",
        "Install",
        "Load",
        "Placeholder",
        "Parameter",
        "Lateral",
        "Unnest",
        "Values",
        "Pivot",
        "TableSample",
        "Into",
        "Struct",
        "Array",
        "Bracket",
        "Dot",
        "Lambda",
        "Rand",
        "Uuid",
    ):
        assert getattr(exp, name, None) not in allowed, name


# ---------- logging ----------


def test_a_refusal_is_logged_with_its_category_and_a_hash_but_not_the_sql():
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    sql = "SELECT * FROM 'confidential_payroll_2024.csv'"
    SqlValidator({"sales": SCHEMAS["sales"]}).validate(sql)
    [event] = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert event["event"] == "sql_rejected"
    assert event["level"] == "WARNING"
    assert event["category"] == "forbidden_table"
    assert event["sql_hash"] == sql_hash(sql)
    assert "payroll" not in stream.getvalue()


def test_a_correctable_verdict_and_an_accepted_query_log_nothing_at_warning_level():
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    validator = SqlValidator({"sales": SCHEMAS["sales"]})
    validator.validate("SELECT nope FROM sales")
    validator.validate("SELECT 1")
    assert stream.getvalue() == ""


def test_sqlglot_cannot_write_the_models_sql_to_the_logs(capsys):
    """SQLGlot logs the statement when it falls back to an unparsed Command."""
    assert not logging.getLogger("sqlglot").isEnabledFor(logging.WARNING)
    SqlValidator({}).validate("LOAD httpfs_secret_extension_name")
    captured = capsys.readouterr()
    assert "httpfs_secret_extension_name" not in captured.out + captured.err
