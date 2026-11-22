"""Static validation of model-written SQL (T1.08, security.md section 4).

The language model's SQL is untrusted. This module decides whether it may run
and, if so, produces the exact text that will run. Nothing is executed here.

How it works
------------
1. **Parse** with SQLGlot's DuckDB dialect. Anything that does not parse to
   exactly one complete statement is refused (V1, V12).
2. **Walk the tree with a default-deny allowlist.** Every node class must be on
   an explicit list of syntax (``_SYNTAX``) or functions (``_FUNCTIONS``).
   A node on neither list is refused, so a construct added to SQLGlot or DuckDB
   later stays blocked until someone allows it on purpose (spike note B2).
   Table references, joins, recursion and size limits are checked in the same
   walk (V2 to V7, V9, V10).
3. **Qualify columns** against the session schema with SQLGlot's optimizer (V8).
4. **Generate the SQL that will run from the validated tree**, not from the
   model's text, with comments removed and the row cap applied (V11). What runs
   is therefore what was checked, even if SQLGlot and DuckDB would tokenize the
   model's text differently.
5. **Re-check the generated text** as a fuse: it must parse again, pass the
   same checks, and DuckDB's own parser must report exactly one SELECT. If any
   of that fails the query is refused.

Verdicts
--------
``rejected`` ends the turn with a safety message; ``correctable`` goes back to
the corrector with the ``reason``. Deviations from the table in security.md:

* A call to a function that reads files, the environment, settings or nested SQL
  (``getenv``, ``read_csv``, ``query`` and similar) is *rejected*, not
  correctable, because it is an attack signature, not a slip. Other functions
  that are merely not on the allowlist stay correctable (V7).
* Syntax that is not on the allowlist (``PIVOT``, ``VALUES``, ``LATERAL``, list
  and struct literals and so on) is correctable under the category
  ``unsupported_syntax``.
* An unknown table name is rejected, not correctable, unless it is a plain
  identifier. Names with spaces or hyphens count as path-like (V4).

``reason`` is written for the corrector and the logs. It contains identifiers
the model wrote, cut to 40 characters, and never dataset values.
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, TokenError
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import MappingSchema

from analyst.core.logging import get_logger, sql_hash
from analyst.core.models import ValidationCategory, ValidationResult, ValidationStatus

if TYPE_CHECKING:
    from analyst.database.session_db import SessionDatabase

log = get_logger(__name__)

# SQLGlot logs the offending SQL when it falls back to a ``Command`` node. The
# model's SQL must not reach the logs (threat T10), so its logger is silenced.
_sqlglot_logger = logging.getLogger("sqlglot")
_sqlglot_logger.addHandler(logging.NullHandler())
_sqlglot_logger.setLevel(logging.CRITICAL)
_sqlglot_logger.propagate = False

DIALECT = "duckdb"

DEFAULT_MAX_ROWS = 10_000
MAX_SQL_CHARS = 10_000
MAX_NODES = 2_000
MAX_DEPTH = 40
MAX_JOINS = 4
# A join without a condition is allowed only when both inputs are small.
SMALL_INPUT_ROWS = 1_000
_NAME_CHARS = 40

# --------------------------------------------------------------------------
# Allowlists
# --------------------------------------------------------------------------

# Function names (as DuckDB spells them) grouped as in security.md section 4,
# keyed by the SQLGlot class they parse to. Several names can share a class:
# ``log10`` and ``log`` both become ``Log``, ``ltrim``/``rtrim``/``trim`` become
# ``Trim``. The accepted synonyms that come with a class (``ceiling``, ``substr``,
# ``len``, ``nvl`` and so on) are harmless spellings of an allowlisted function.
_FUNCTION_NAMES: dict[str, tuple[str, ...]] = {
    "Count": ("count",),
    "Sum": ("sum",),
    "Avg": ("avg",),
    "Min": ("min",),
    "Max": ("max",),
    "Median": ("median",),
    "Mode": ("mode",),
    "Stddev": ("stddev",),
    "StddevSamp": ("stddev_samp",),
    "StddevPop": ("stddev_pop",),
    "Variance": ("variance", "var_samp"),
    "VariancePop": ("var_pop",),
    "PercentileCont": ("quantile_cont",),
    "PercentileDisc": ("quantile_disc",),
    "ApproxDistinct": ("approx_count_distinct",),
    "CountIf": ("count_if",),
    "LogicalAnd": ("bool_and",),
    "LogicalOr": ("bool_or",),
    "First": ("first",),
    "Last": ("last",),
    "ArgMin": ("arg_min",),
    "ArgMax": ("arg_max",),
    "Corr": ("corr",),
    "CovarPop": ("covar_pop",),
    "CovarSamp": ("covar_samp",),
    "GroupConcat": ("string_agg",),
    "ArrayAgg": ("list",),
    "RowNumber": ("row_number",),
    "Rank": ("rank",),
    "DenseRank": ("dense_rank",),
    "PercentRank": ("percent_rank",),
    "CumeDist": ("cume_dist",),
    "Ntile": ("ntile",),
    "Lag": ("lag",),
    "Lead": ("lead",),
    "FirstValue": ("first_value",),
    "LastValue": ("last_value",),
    "NthValue": ("nth_value",),
    "Abs": ("abs",),
    "Round": ("round",),
    "Floor": ("floor",),
    "Ceil": ("ceil",),
    "Sqrt": ("sqrt",),
    "Pow": ("power",),
    "Ln": ("ln",),
    "Log": ("log", "log10"),
    "Exp": ("exp",),
    "Sign": ("sign",),
    "Greatest": ("greatest",),
    "Least": ("least",),
    "Coalesce": ("coalesce", "ifnull"),
    "Nullif": ("nullif",),
    "If": ("if",),
    "Case": ("case",),
    "Cast": ("cast",),
    "TryCast": ("try_cast",),
    "Lower": ("lower",),
    "Upper": ("upper",),
    "Trim": ("trim", "ltrim", "rtrim"),
    "Length": ("length",),
    "Substring": ("substring",),
    "Replace": ("replace",),
    "Concat": ("concat",),
    "StartsWith": ("starts_with",),
    "EndsWith": ("ends_with",),
    "Contains": ("contains",),
    "RegexpLike": ("regexp_matches",),
    "SplitPart": ("split_part",),
    "Left": ("left",),
    "Right": ("right",),
    "TimestampTrunc": ("date_trunc",),
    "Extract": ("extract",),
    "Year": ("year",),
    "Quarter": ("quarter",),
    "Month": ("month",),
    "Week": ("week",),
    "Day": ("day",),
    "DayOfWeek": ("dayofweek",),
    "DayOfYear": ("dayofyear",),
    "Hour": ("hour",),
    "TimeToStr": ("strftime",),
    "StrToTime": ("strptime",),
    "DateFromParts": ("make_date",),
    "DateDiff": ("date_diff",),
    "DateAdd": ("date_add",),
    "CurrentDate": ("current_date",),
    "TimeToUnix": ("epoch",),
    "LastDay": ("last_day",),
}

# Allowlisted functions that SQLGlot does not know and returns as ``Anonymous``.
_ANONYMOUS_FUNCTIONS: tuple[str, ...] = ("mean", "date_part", "date_sub", "age")

# Syntax that is not a function: query structure, operators, literals.
_SYNTAX_NAMES: tuple[str, ...] = tuple(
    (
        # Query structure
        "Select Union Intersect Except Subquery With CTE From Join Where Group Having "
        "Order Ordered Limit Offset Qualify Distinct Alias TableAlias Table Column Star "
        "Identifier "
        # Window and aggregate modifiers
        "Window WindowSpec Filter WithinGroup IgnoreNulls RespectNulls "
        # Values and types
        "Literal Null Boolean Paren Tuple Interval Var DataType DataTypeParam "
        # Operators
        "And Or Not EQ NEQ GT GTE LT LTE NullSafeEQ NullSafeNEQ Add Sub Mul Div IntDiv "
        "Mod Neg Between In Is Like ILike Escape DPipe Exists "
    )
    .strip()
    .split()
)

# Names of statement classes (V3). Compared by name so that a SQLGlot release
# that renames or adds one does not break import.
_STATEMENT_NAMES = frozenset(
    (
        "Insert Update Delete Merge Drop Create Alter AlterTable AlterColumn TruncateTable "
        "Copy Attach Detach Pragma Set SetItem Install Load Export Import Call Command Use Into "
        "Describe Show Summarize Analyze Transaction Commit Rollback Execute Kill Grant "
        "Revoke Cache Uncache "
    )
    .strip()
    .split()
)

# Functions that read files, the environment, settings or nested SQL, or that
# stall the engine. A call is an attack signature, so it is rejected (V7).
_DANGEROUS_FUNCTIONS = frozenset(
    {
        "getenv",
        "current_setting",
        "getvariable",
        "query",
        "query_table",
        "glob",
        "sleep",
        "pg_sleep",
        "error",
        "nextval",
        "currval",
        "setseed",
        "load_extension",
        "install_extension",
        "unnest",
        "generate_series",
        "range",
        "sniff_csv",
        "which_secret",
    }
)
_DANGEROUS_PREFIXES = (
    "read_",
    "write_",
    "duckdb_",
    "pragma_",
    "pg_",
    "sqlite_",
    "parquet_",
    "iceberg_",
    "delta_",
    "glob",
    "json_each",
)

# System catalogs (V5). A table that is not a session table and starts with
# one of these is rejected rather than offered for correction.
_SYSTEM_TABLE_PREFIXES = ("duckdb_", "pg_", "sqlite_", "pragma_", "information_schema")

_COLUMN_ERROR = re.compile(r"[Uu]nknown column: (\S+)|[Cc]olumn '([^']+)' could not be resolved")

_JOIN_METHODS = (None, "", "NATURAL")
_JOIN_KINDS = (None, "", "INNER", "OUTER", "CROSS")
_JOIN_SIDES = (None, "", "LEFT", "RIGHT", "FULL")

# Nodes that are allowed to carry these arguments on a Table (everything else,
# such as sampling or pivots, is unsupported syntax).
_TABLE_ARGS = frozenset({"this", "alias", "db", "catalog"})

# Which finding is reported when a query has several problems: refusals before
# corrections, and within each the more fundamental problem first (a wrong table
# makes every column question moot).
_REJECT_ORDER: tuple[ValidationCategory, ...] = (
    "multiple_statements",
    "forbidden_statement",
    "forbidden_table",
    "table_function",
    "disallowed_function",
    "too_complex",
    "parse_error",
)
_CORRECT_ORDER: tuple[ValidationCategory, ...] = (
    "unknown_table",
    "disallowed_function",
    "unsupported_syntax",
    "too_complex",
    "unknown_column",
)


def _rank(finding: _Finding) -> tuple[int, int]:
    if finding.status == "rejected":
        return (0, _REJECT_ORDER.index(finding.category))
    return (1, _CORRECT_ORDER.index(finding.category))


def _classes(names: Iterable[str]) -> frozenset[type[exp.Expression]]:
    """Resolve SQLGlot class names; a missing name fails at import, not in production."""
    return frozenset(getattr(exp, name) for name in names)


_SYNTAX = _classes(_SYNTAX_NAMES)
_FUNCTIONS = _classes(_FUNCTION_NAMES)
_QUERY_ROOTS = _classes(("Select", "Union", "Intersect", "Except"))
_SET_OPERATIONS = _classes(("Union", "Intersect", "Except"))

ALLOWED_FUNCTION_NAMES: tuple[str, ...] = tuple(
    sorted(
        {name for names in _FUNCTION_NAMES.values() for name in names} | set(_ANONYMOUS_FUNCTIONS)
    )
)
"""Every function name on the allowlist, for prompts and error messages."""


@dataclass(frozen=True)
class _Finding:
    status: ValidationStatus
    category: ValidationCategory
    reason: str


def _reject(category: ValidationCategory, reason: str) -> _Finding:
    return _Finding("rejected", category, reason)


def _correct(category: ValidationCategory, reason: str) -> _Finding:
    return _Finding("correctable", category, reason)


def _safe_name(value: object) -> str:
    """An identifier the model wrote, made safe to put in a reason or a log line."""
    text = re.sub(r"[^\w$]", "?", str(value))
    return text[:_NAME_CHARS] + ("..." if len(text) > _NAME_CHARS else "")


def _kind(node: exp.Expression) -> str:
    """Readable name of a syntax node class, for example ``TableSample``."""
    return _safe_name(type(node).__name__)


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return str(node.sql_name()).lower()


def _is_dangerous_function(name: str) -> bool:
    return name in _DANGEROUS_FUNCTIONS or name.startswith(_DANGEROUS_PREFIXES)


# --------------------------------------------------------------------------
# DuckDB's own parser, used as an independent second opinion
# --------------------------------------------------------------------------


class _DuckDBParser:
    """Asks DuckDB how many statements a text holds and what kind they are.

    SQLGlot and DuckDB tokenize SQL separately. If they ever disagreed about
    where a statement ends, the text SQLGlot checked could run as something else.
    This holds one empty, locked-down, parse-only connection; nothing is executed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection: duckdb.DuckDBPyConnection | None = None

    def statement_types(self, sql: str) -> list[str] | None:
        """Statement types DuckDB finds, or ``None`` if DuckDB cannot parse the text."""
        with self._lock:
            try:
                if self._connection is None:
                    self._connection = duckdb.connect(
                        ":memory:",
                        config={
                            "enable_external_access": "false",
                            "autoload_known_extensions": "false",
                            "autoinstall_known_extensions": "false",
                        },
                    )
                statements = self._connection.extract_statements(sql)
            except duckdb.Error:
                return None
        return [str(statement.type).rsplit(".", maxsplit=1)[-1] for statement in statements]


_duckdb_parser = _DuckDBParser()


# --------------------------------------------------------------------------
# The validator
# --------------------------------------------------------------------------


class SqlValidator:
    """Validates SQL against the session schema and prepares it for execution.

    Create one per loaded dataset. It holds no connection and is safe to share
    between threads.
    """

    def __init__(
        self,
        tables: Mapping[str, Sequence[str]],
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        """Create a validator.

        Args:
            tables: Session tables and their column names. Names are compared
                case-insensitively, as DuckDB does.
            max_rows: Row cap applied to every query (V11). The wrapped query asks
                for one more row so the executor can tell that the result was cut.
        """
        if max_rows < 1:
            msg = "max_rows must be at least 1"
            raise ValueError(msg)
        self._tables: dict[str, tuple[str, ...]] = {
            name.lower(): tuple(columns) for name, columns in tables.items()
        }
        self._max_rows = max_rows

    @classmethod
    def for_database(cls, db: SessionDatabase, *, max_rows: int = DEFAULT_MAX_ROWS) -> SqlValidator:
        """Build a validator for the table loaded into a session database."""
        columns = [column.sql_name for column in db.columns]
        return cls({db.table_name: columns}, max_rows=max_rows)

    @property
    def max_rows(self) -> int:
        """Row cap applied to every query."""
        return self._max_rows

    def validate(self, sql: str) -> ValidationResult:
        """Check ``sql`` and, if it is acceptable, return the text to execute.

        Never raises for bad input; every failure is a ``ValidationResult``.
        """
        try:
            finding, safe_sql = self._validate(sql)
        except RecursionError:
            finding, safe_sql = _reject("too_complex", "The query is nested too deeply."), None
        except Exception:
            # A bug here must refuse the query, never let it through.
            log.exception("validator_internal_error", extra={"sql_hash": _digest(sql)})
            finding = _reject("parse_error", "The query could not be checked.")
            safe_sql = None
        if finding is None:
            return ValidationResult(
                status="ok", category="ok", safe_sql=safe_sql, max_rows=self._max_rows
            )
        if finding.status == "rejected":
            log.warning(
                "sql_rejected",
                extra={
                    "category": finding.category,
                    "reason": finding.reason,
                    "sql_hash": _digest(sql),
                },
            )
        return ValidationResult(
            status=finding.status, category=finding.category, reason=finding.reason
        )

    # -- pipeline ------------------------------------------------------------

    def _validate(self, sql: str) -> tuple[_Finding | None, str | None]:
        if not isinstance(sql, str) or not sql.strip():
            return _reject("parse_error", "The query is empty."), None
        if "\x00" in sql:
            return _reject("parse_error", "The query contains a NUL character."), None
        if len(sql) > MAX_SQL_CHARS:
            return _correct(
                "too_complex", f"The query is longer than {MAX_SQL_CHARS} characters."
            ), None

        tree, finding = self._parse(sql)
        if finding is not None or tree is None:
            return finding, None

        finding = self._check_tree(tree)
        if finding is not None:
            return finding, None

        finding = self._check_columns(tree)
        if finding is not None:
            return finding, None

        safe_sql = self._generate(tree)
        finding = self._check_generated(safe_sql)
        if finding is not None:
            log.error(
                "validator_fuse_failed",
                extra={"category": finding.category, "sql_hash": _digest(sql)},
            )
            return finding, None
        finding = self._check_duckdb_agrees(safe_sql)
        if finding is not None:
            return finding, None
        return None, safe_sql

    @staticmethod
    def _parse(sql: str) -> tuple[exp.Expression | None, _Finding | None]:
        """Parse to exactly one statement (V1, V12)."""
        try:
            parsed = sqlglot.parse(sql, read=DIALECT)
        except ParseError as error:
            return None, _reject("parse_error", _parse_reason(error))
        except TokenError:
            return None, _reject("parse_error", "The query could not be split into tokens.")
        statements = [statement for statement in parsed if statement is not None]
        if not statements:
            return None, _reject("parse_error", "The query contains no statement.")
        if len(statements) > 1:
            return None, _reject(
                "multiple_statements",
                f"Exactly one statement is allowed, found {len(statements)}.",
            )
        return statements[0], None

    def _check_tree(self, tree: exp.Expression) -> _Finding | None:
        """Structure, tables, functions and cost rules (V2 to V7, V9, V10)."""
        root = tree.unnest() if isinstance(tree, exp.Subquery) else tree
        if type(root) not in _QUERY_ROOTS:
            kind = _kind(root).upper()
            return _reject("forbidden_statement", f"Only SELECT queries are allowed, found {kind}.")
        return _TreeCheck(self._tables).run(tree)

    def _check_columns(self, tree: exp.Expression) -> _Finding | None:
        """Every column exists in the session schema (V8)."""
        schema = MappingSchema(
            {table: dict.fromkeys(columns, "TEXT") for table, columns in self._tables.items()},
            dialect=DIALECT,
        )
        try:
            qualify(
                tree.copy(),
                schema=schema,
                dialect=DIALECT,
                validate_qualify_columns=True,
                identify=False,
            )
        except OptimizeError as error:
            return _correct("unknown_column", self._column_reason(str(error)))
        except Exception:
            log.exception("column_check_failed")
            return _reject("parse_error", "The column names could not be checked.")
        return None

    def _column_reason(self, message: str) -> str:
        """Tell the corrector which column failed and the nearest real names."""
        match = _COLUMN_ERROR.search(message)
        name = next((group for group in (match.groups() if match else ()) if group), None)
        if name is None:
            return "A column name could not be resolved. Use only columns from the schema."
        bare = name.lower().split(".")[-1]
        known = sorted({column for columns in self._tables.values() for column in columns})
        if bare in known:
            return (
                f"Column '{_safe_name(bare)}' is ambiguous or belongs to a different table. "
                "Qualify it with a table alias."
            )
        reason = f"Unknown column '{_safe_name(name)}'."
        close = difflib.get_close_matches(bare, known, n=3, cutoff=0.6)
        if close:
            reason += f" Did you mean: {', '.join(close)}?"
        return reason

    def _generate(self, tree: exp.Expression) -> str:
        """The text that will run: the validated tree under a row cap (V11)."""
        query = tree.unnest() if isinstance(tree, exp.Subquery) else tree
        wrapped = exp.select("*").from_(query.subquery()).limit(self._max_rows + 1)
        return wrapped.sql(dialect=DIALECT, comments=False)

    def _check_generated(self, safe_sql: str) -> _Finding | None:
        """Fuse: the generated text must parse again and pass the same checks.

        This cannot fail for a query that passed ``_check_tree`` unless the SQL
        generator changed its meaning, so a failure here is logged as an error.
        """
        try:
            parsed = [s for s in sqlglot.parse(safe_sql, read=DIALECT) if s is not None]
        except (ParseError, TokenError):
            return _reject("parse_error", "The query could not be prepared for execution.")
        if len(parsed) != 1:
            return _reject("multiple_statements", "The prepared query is not a single statement.")
        finding = self._check_tree(parsed[0])
        if finding is not None:
            return _reject(finding.category, "The prepared query failed validation.")
        return None

    @staticmethod
    def _check_duckdb_agrees(safe_sql: str) -> _Finding | None:
        """DuckDB's own parser must read the text as exactly one SELECT.

        SQLGlot accepts some text DuckDB does not (a trailing ``GROUP BY`` with no
        columns, for example); this catches it before execution.
        """
        types = _duckdb_parser.statement_types(safe_sql)
        if types is None:
            return _reject("parse_error", "DuckDB could not parse the query.")
        if len(types) != 1:
            return _reject("multiple_statements", "DuckDB reads the query as several statements.")
        if types[0] != "SELECT":
            return _reject(
                "forbidden_statement", f"DuckDB reads the query as {_safe_name(types[0])}."
            )
        return None


def _digest(sql: object) -> str:
    return sql_hash(sql) if isinstance(sql, str) else "n/a"


def _parse_reason(error: ParseError) -> str:
    """A syntax-error message without the SQL excerpt SQLGlot attaches."""
    first = error.errors[0] if error.errors else {}
    description = _safe_name(str(first.get("description", "syntax error")).replace("\n", " "))
    line, column = first.get("line"), first.get("col")
    where = f" at line {line}, column {column}" if line and column else ""
    return f"The query could not be parsed: {description}{where}."


# --------------------------------------------------------------------------
# Tree walk
# --------------------------------------------------------------------------


class _TreeCheck:
    """One pass over a statement. Collects findings and returns the most serious."""

    def __init__(self, tables: Mapping[str, Sequence[str]]) -> None:
        self._tables = tables
        self._findings: list[_Finding] = []
        self._ctes: dict[str, exp.Expression | None] = {}
        self._join_count = 0

    def run(self, tree: exp.Expression) -> _Finding | None:
        self._collect_ctes(tree)
        count = 0
        stack: list[tuple[exp.Expression, int]] = [(tree, 1)]
        while stack:
            node, depth = stack.pop()
            count += 1
            if count > MAX_NODES:
                self._add(_correct("too_complex", f"The query has more than {MAX_NODES} parts."))
                break
            if depth > MAX_DEPTH:
                self._add(
                    _correct("too_complex", f"The query is nested more than {MAX_DEPTH} levels.")
                )
                break
            self._inspect(node)
            stack.extend((child, depth + 1) for child in node.iter_expressions())
        if self._join_count > MAX_JOINS:
            self._add(
                _correct(
                    "too_complex",
                    f"At most {MAX_JOINS} joins are allowed, found {self._join_count}.",
                )
            )
        if not self._findings:
            return None
        return min(self._findings, key=_rank)

    def _add(self, finding: _Finding) -> None:
        self._findings.append(finding)

    def _collect_ctes(self, tree: exp.Expression) -> None:
        """Names defined by WITH anywhere in the query, with their definitions.

        A name defined twice, or that shadows a session table, maps to ``None``:
        its size is not known, so the cost rule treats it as unbounded.
        """
        for cte in tree.find_all(exp.CTE):
            name = cte.alias.lower()
            clash = name in self._ctes or name in self._tables
            self._ctes[name] = None if clash else cte.this

    # -- per node --------------------------------------------------------------

    def _inspect(self, node: exp.Expression) -> None:
        cls = type(node)
        if cls is exp.Table:
            self._check_table(node)
        elif cls is exp.Join:
            self._join_count += 1
            self._check_join_kind(node)
        elif cls is exp.With:
            if node.args.get("recursive"):
                self._add(_correct("too_complex", "WITH RECURSIVE is not allowed."))
        elif cls is exp.Select:
            self._check_sources(node)
        elif cls is exp.Anonymous:
            self._check_anonymous(node)
        elif cls not in _SYNTAX and cls not in _FUNCTIONS:
            self._deny(node)

    def _deny(self, node: exp.Expression) -> None:
        """A node that is on no allowlist."""
        name = type(node).__name__
        if name in _STATEMENT_NAMES:
            self._add(
                _reject("forbidden_statement", f"{_kind(node).upper()} is not allowed in a query.")
            )
        elif isinstance(node, exp.Func):
            self._deny_function(_function_name(node))
        else:
            self._add(_correct("unsupported_syntax", f"{_kind(node)} syntax is not supported."))

    def _check_anonymous(self, node: exp.Anonymous) -> None:
        name = _function_name(node)
        if name in _ANONYMOUS_FUNCTIONS:
            return
        self._deny_function(name, typed_by_model=True)

    def _deny_function(self, name: str, *, typed_by_model: bool = False) -> None:
        if _is_dangerous_function(name):
            self._add(
                _reject("disallowed_function", f"Function '{_safe_name(name)}' is not allowed.")
            )
            return
        hint = difflib.get_close_matches(name, ALLOWED_FUNCTION_NAMES, n=1, cutoff=0.75)
        suggestion = f" Did you mean '{hint[0]}'?" if hint and typed_by_model else ""
        self._add(
            _correct(
                "disallowed_function",
                f"Function '{_safe_name(name)}' is not supported.{suggestion} "
                "Use only the approved functions.",
            )
        )

    # -- tables (V4 to V6) ------------------------------------------------------

    def _check_table(self, node: exp.Table) -> None:
        extra = [key for key, value in node.args.items() if value and key not in _TABLE_ARGS]
        if extra:
            self._add(
                _correct(
                    "unsupported_syntax", f"{_safe_name(extra[0])} on a table is not supported."
                )
            )
        if not isinstance(node.this, exp.Identifier):
            self._add(_reject("table_function", "Functions used as tables are not allowed."))
            return
        if node.args.get("db") or node.args.get("catalog"):
            self._add(_reject("forbidden_table", "Schema-qualified table names are not allowed."))
            return
        name = str(node.name)
        lowered = name.lower()
        if lowered in self._tables or lowered in self._ctes:
            return
        if not name.isidentifier() or lowered.startswith(_SYSTEM_TABLE_PREFIXES):
            self._add(_reject("forbidden_table", "That table name is not allowed."))
            return
        known = ", ".join(sorted(self._tables))
        self._add(
            _correct(
                "unknown_table", f"Table '{_safe_name(name)}' does not exist. Available: {known}."
            )
        )

    # -- sources and joins (V6, V10) --------------------------------------------

    def _check_join_kind(self, join: exp.Join) -> None:
        """Plain INNER, OUTER and CROSS joins only; ASOF, SEMI, ANTI and the like are not."""
        allowed = (
            join.args.get("method") in _JOIN_METHODS
            and join.args.get("kind") in _JOIN_KINDS
            and join.args.get("side") in _JOIN_SIDES
            and not join.args.get("match_condition")
        )
        if not allowed:
            self._add(
                _correct(
                    "unsupported_syntax",
                    "Only INNER, LEFT, RIGHT, FULL and CROSS joins are supported.",
                )
            )

    def _check_sources(self, select: exp.Select) -> None:
        sources: list[exp.Expression] = []
        from_ = select.args.get("from_")
        if from_ is not None:
            sources.append(from_.this)
        joins: list[exp.Join] = list(select.args.get("joins") or [])
        sources.extend(join.this for join in joins)
        for source in sources:
            self._check_source(source)
        preceding: list[exp.Expression] = sources[:1]
        for join in joins:
            if not join.args.get("on") and not join.args.get("using"):
                self._check_unconditioned(preceding, join.this)
            preceding.append(join.this)

    def _check_source(self, source: exp.Expression) -> None:
        cls = type(source)
        if cls in (exp.Table, exp.Subquery):
            return
        if isinstance(source, exp.Func):
            self._add(_reject("table_function", "Functions used as tables are not allowed."))
        else:
            self._add(
                _correct("unsupported_syntax", f"{_kind(source)} is not supported as a table.")
            )

    def _check_unconditioned(self, preceding: list[exp.Expression], right: exp.Expression) -> None:
        """A join with no condition multiplies row counts: allow it only when small."""
        left_bound = self._product_of_bounds(preceding)
        right_bound = self._source_bound(right)
        both_small = left_bound is not None and right_bound is not None
        if both_small or left_bound == 1 or right_bound == 1:
            return
        self._add(
            _correct(
                "too_complex",
                "A join without a condition is only allowed when both sides are subqueries "
                f"limited to {SMALL_INPUT_ROWS} rows, or one side is a single row. "
                "Add a join condition.",
            )
        )

    def _product_of_bounds(self, sources: list[exp.Expression]) -> int | None:
        total = 1
        for source in sources:
            bound = self._source_bound(source)
            if bound is None:
                return None
            total *= bound
        return total

    def _source_bound(
        self, source: exp.Expression, seen: frozenset[str] = frozenset()
    ) -> int | None:
        """Most rows a FROM or JOIN input can have, if provably at most 1,000."""
        if isinstance(source, exp.Subquery):
            return self._query_bound(source.this, seen)
        if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier):
            name = str(source.name).lower()
            definition = self._ctes.get(name)
            if definition is None or name in seen:
                return None
            return self._query_bound(definition, seen | {name})
        return None

    def _query_bound(self, query: exp.Expression, seen: frozenset[str]) -> int | None:
        query = query.unnest() if isinstance(query, exp.Subquery) else query
        limit = _literal_limit(query)
        if isinstance(query, exp.Select):
            bounds = [limit] if limit is not None else []
            if _is_single_row(query):
                bounds.append(1)
            from_ = query.args.get("from_")
            if from_ is not None and not query.args.get("joins"):
                # Filtering, grouping and sorting never add rows to a single input.
                source_bound = self._source_bound(from_.this, seen)
                if source_bound is not None:
                    bounds.append(source_bound)
            result = min(bounds) if bounds else None
        elif type(query) in _SET_OPERATIONS:
            result = self._set_operation_bound(query, seen)
            if limit is not None:
                result = limit if result is None else min(result, limit)
        else:
            result = None
        return result if result is not None and result <= SMALL_INPUT_ROWS else None

    def _set_operation_bound(self, query: exp.Expression, seen: frozenset[str]) -> int | None:
        left = self._query_bound(query.this, seen)
        right = self._query_bound(query.expression, seen)
        if isinstance(query, exp.Union):
            return None if left is None or right is None else left + right
        return left


def _literal_limit(query: exp.Expression) -> int | None:
    limit = query.args.get("limit")
    value = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(value, exp.Literal) and not value.is_string and value.name.isdigit():
        return int(value.name)
    return None


def _is_single_row(select: exp.Select) -> bool:
    """True for ``SELECT <aggregates>`` with no GROUP BY, and for SELECT without FROM."""
    if select.args.get("group"):
        return False
    if select.args.get("from_") is None and not select.args.get("joins"):
        return True
    for projection in select.expressions:
        if projection.find(exp.AggFunc) is not None and projection.find(exp.Window) is None:
            return True
    return False
