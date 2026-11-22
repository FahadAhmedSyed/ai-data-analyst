"""Per-session DuckDB database: table load, lockdown and catalog integrity (T1.06).

Each browser session owns one ``SessionDatabase`` (ADR 0002). It is an in-memory
DuckDB connection holding the one uploaded table. Three layers protect it:

1. **Lockdown** (security.md, section 5). After the table is loaded, external
   access, extension loading, Python variable lookups and spilling to a
   temporary directory are switched off, and the configuration is locked.
   Turning external access off does *not* stop DuckDB 2.0.1 from spilling large
   intermediate results to a ``.tmp`` directory, so ``temp_directory`` is set to
   empty: query data is never written to disk (NFR-5), and a query that needs
   more memory than the cap fails with an out-of-memory error (the executor maps
   it to ``out_of_memory``).
2. **Validator and executor** (T1.08, T1.09) stop anything but a single SELECT
   over the session table from being run.
3. **Catalog integrity check** (spike note A1). DuckDB has no read-only mode for
   in-memory databases, so DDL and DML are not blocked by the database itself.
   ``verify_catalog`` compares the database with a snapshot taken at load:
   every database object, the column names and types, the row count and, for
   tables up to ``FINGERPRINT_MAX_CELLS`` cells, a hash of all values. Any
   difference marks the session corrupted and raises ``SessionCorruptedError``.
   Beyond the cell cap the value hash is skipped (it would add seconds per
   question), so a changed value in a very large table is caught only by the
   validator.

``query_df`` is for SQL written by this application (the profiler). SQL written
by the language model never runs here: it goes through the validator and executor.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

import duckdb
import pandas as pd

from analyst.config.settings import Settings, get_settings
from analyst.core.errors import IngestionError, SessionCorruptedError
from analyst.core.logging import get_logger
from analyst.core.models import ColumnSchema, ColumnType
from analyst.ingestion.normalize import TypedTable, to_identifier

log = get_logger(__name__)

# Above this many cells (rows times columns) the per-question value hash is
# skipped. About 50 million cells hash in roughly 0.3 s on two threads.
FINGERPRINT_MAX_CELLS = 50_000_000

_INPUT_VIEW = "typed_input"
_SQL_TYPES: dict[ColumnType, str] = {
    "integer": "BIGINT",
    "decimal": "DOUBLE",
    "text": "VARCHAR",
    "boolean": "BOOLEAN",
    "date": "DATE",
    "timestamp": "TIMESTAMP",
}
_SIZE_UNITS = {"KB": "KB", "MB": "MB", "GB": "GB", "KIB": "KiB", "MIB": "MiB", "GIB": "GiB"}
_SIZE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([A-Za-z]+)$")

# Every named object a session could create or change, across all attached
# databases. Rows are (kind, qualified name, detail).
_CATALOG_SQL = """
SELECT 'database' AS kind, database_name AS name, '' AS detail
  FROM duckdb_databases() WHERE NOT internal
UNION ALL SELECT 'schema', database_name || '.' || schema_name, ''
  FROM duckdb_schemas() WHERE NOT internal
UNION ALL SELECT 'table', database_name || '.' || schema_name || '.' || table_name,
  CAST(temporary AS VARCHAR) FROM duckdb_tables()
UNION ALL SELECT 'view', database_name || '.' || schema_name || '.' || view_name, ''
  FROM duckdb_views() WHERE NOT internal
UNION ALL SELECT 'sequence', database_name || '.' || schema_name || '.' || sequence_name, ''
  FROM duckdb_sequences()
UNION ALL SELECT 'type', database_name || '.' || schema_name || '.' || type_name, ''
  FROM duckdb_types() WHERE NOT internal
UNION ALL SELECT 'macro', database_name || '.' || schema_name || '.' || function_name,
  function_type FROM duckdb_functions() WHERE NOT internal
UNION ALL SELECT 'index', database_name || '.' || schema_name || '.' || index_name, ''
  FROM duckdb_indexes()
UNION ALL SELECT 'column',
  database_name || '.' || schema_name || '.' || table_name || '.' || column_name, data_type
  FROM duckdb_columns() WHERE NOT internal
"""

# Settings that must hold after lockdown. Checked, not assumed.
_EXPECTED_SETTINGS = {
    "enable_external_access": "false",
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
    "allow_community_extensions": "false",
    "python_enable_replacements": "false",
    "temp_directory": "",
    "lock_configuration": "true",
}


def default_table_name(source_name: str) -> str:
    """Table name from a file name: ``Retail Sales.csv`` gives ``retail_sales``."""
    return to_identifier(PurePath(source_name).stem)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _duckdb_size(limit: str) -> str:
    """Normalise a validated size such as ``1GIB`` to the spelling DuckDB documents."""
    match = _SIZE_PATTERN.match(limit.strip())
    unit = _SIZE_UNITS.get(match.group(2).upper()) if match else None
    if match is None or unit is None:
        msg = f"Unsupported memory limit: {limit!r}"
        raise ValueError(msg)
    return f"{match.group(1)}{unit}"


@dataclass(frozen=True)
class CatalogSnapshot:
    """What the session database held when it was last known to be good."""

    objects: frozenset[tuple[str, str, str]]
    row_count: int
    fingerprint: int | None

    def differences(self, other: CatalogSnapshot) -> list[str]:
        """Names of what changed from this snapshot to ``other``; empty when identical."""
        changed = sorted({kind for kind, _, _ in self.objects ^ other.objects})
        if self.row_count != other.row_count:
            changed.append("row_count")
        if (
            self.fingerprint is not None
            and other.fingerprint is not None
            and self.fingerprint != other.fingerprint
        ):
            changed.append("content")
        return changed


class SessionDatabase:
    """One session's locked-down in-memory DuckDB database and its single table."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty session. Nothing is opened until ``load`` is called.

        Args:
            settings: Limits to apply; defaults to the application settings.
            clock: Monotonic time source in seconds; replaced in tests.
        """
        self._settings = settings or get_settings()
        self._clock = clock
        self._con: duckdb.DuckDBPyConnection | None = None
        self._table_name: str | None = None
        self._columns: tuple[ColumnSchema, ...] = ()
        self._snapshot: CatalogSnapshot | None = None
        self._content_checked = False
        self._corrupted = False
        self._last_active = clock()
        #: Serialises use of the connection. ``interrupt`` deliberately ignores it.
        self.lock = threading.RLock()

    # ----- state -----

    @property
    def is_loaded(self) -> bool:
        """True when a table has been loaded and the session is open."""
        return self._con is not None

    @property
    def corrupted(self) -> bool:
        """True once the catalog check has found an unexpected change."""
        return self._corrupted

    @property
    def table_name(self) -> str:
        """SQL name of the session table."""
        self._require_open()
        assert self._table_name is not None  # noqa: S101 - set together with the connection
        return self._table_name

    @property
    def columns(self) -> tuple[ColumnSchema, ...]:
        """Schema of the session table."""
        self._require_open()
        return self._columns

    @property
    def row_count(self) -> int:
        """Rows in the table as loaded."""
        self._require_open()
        assert self._snapshot is not None  # noqa: S101 - set together with the connection
        return self._snapshot.row_count

    @property
    def content_checked(self) -> bool:
        """True when the catalog check also hashes the table's values."""
        return self._content_checked

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The locked-down connection, for the executor. Refused once the session is corrupted."""
        self._require_open()
        assert self._con is not None  # noqa: S101 - guaranteed by _require_open
        return self._con

    def _require_open(self) -> None:
        if self._corrupted:
            raise SessionCorruptedError("session is corrupted")
        if self._con is None:
            msg = "No dataset is loaded in this session"
            raise RuntimeError(msg)

    # ----- loading -----

    def load(self, table: TypedTable, table_name: str | None = None) -> str:
        """Load a typed table into a fresh database, lock it down and snapshot it.

        Calling ``load`` again replaces the dataset: the old database is closed and a
        new one is built, so nothing from the previous file can remain.

        Args:
            table: The typed table from ``normalize``.
            table_name: SQL name for the table; defaults to the file name.

        Returns:
            The table name.

        Raises:
            ValueError: If the table name is unsafe or the columns do not match.
            IngestionError: If the table does not fit in memory.
        """
        name = table_name or default_table_name(table.source_name)
        if name != to_identifier(name):
            msg = f"{name!r} is not a safe table name (try {to_identifier(name)!r})"
            raise ValueError(msg)
        frame_columns = [str(c) for c in table.dataframe.columns]
        if not table.columns or frame_columns != [c.sql_name for c in table.columns]:
            msg = "The table's columns do not match its schema"
            raise ValueError(msg)

        started = time.perf_counter()
        with self.lock:
            self.close()
            con = duckdb.connect(":memory:")
            try:
                self._create_table(con, table, name)
                self._apply_lockdown(con)
                cells = self._count_rows(con, name) * len(table.columns)
                snapshot = self._take_snapshot(con, name, content=cells <= FINGERPRINT_MAX_CELLS)
            except duckdb.OutOfMemoryException as exc:
                con.close()
                raise IngestionError(
                    "too_large",
                    str(exc),
                    user_message="This file needs more memory than is available to analyse it.",
                ) from exc
            except BaseException:
                con.close()
                raise
            self._con, self._table_name, self._columns = con, name, table.columns
            self._snapshot, self._corrupted = snapshot, False
            self._content_checked = snapshot.fingerprint is not None
            self.touch()

        log.info(
            "table_loaded",
            extra={
                "rows": snapshot.row_count,
                "columns": len(table.columns),
                "content_checked": self._content_checked,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return name

    @staticmethod
    def _create_table(con: duckdb.DuckDBPyConnection, table: TypedTable, name: str) -> None:
        select_list = ", ".join(
            f"CAST({_quote(c.sql_name)} AS {_SQL_TYPES[c.dtype]}) AS {_quote(c.sql_name)}"
            for c in table.columns
        )
        con.register(_INPUT_VIEW, table.dataframe)
        try:
            con.execute(
                f"CREATE TABLE {_quote(name)} AS SELECT {select_list} FROM {_INPUT_VIEW}"  # noqa: S608
            )
        finally:
            con.unregister(_INPUT_VIEW)

    def _apply_lockdown(self, con: duckdb.DuckDBPyConnection) -> None:
        """Apply the lockdown settings, then confirm they took effect (fail closed)."""
        memory = _duckdb_size(self._settings.duckdb_memory_limit)
        for statement in (
            # Must come before external access is switched off, which freezes it.
            "SET temp_directory = ''",
            "SET enable_external_access = false",
            "SET autoinstall_known_extensions = false",
            "SET autoload_known_extensions = false",
            "SET allow_community_extensions = false",
            "SET python_enable_replacements = false",
            f"SET memory_limit = '{memory}'",
            f"SET threads = {int(self._settings.duckdb_threads)}",
            "SET lock_configuration = true",
        ):
            con.execute(statement)
        placeholders = ", ".join("?" for _ in _EXPECTED_SETTINGS)
        rows = con.execute(
            f"SELECT name, value FROM duckdb_settings() WHERE name IN ({placeholders})",  # noqa: S608
            list(_EXPECTED_SETTINGS),
        ).fetchall()
        actual = {name: str(value).lower() for name, value in rows}
        wrong = sorted(k for k, v in _EXPECTED_SETTINGS.items() if actual.get(k) != v)
        if wrong:
            msg = f"DuckDB lockdown did not take effect for: {', '.join(wrong)}"
            raise RuntimeError(msg)

    # ----- catalog integrity -----

    @staticmethod
    def _count_rows(con: duckdb.DuckDBPyConnection, name: str) -> int:
        """Rows in the table, or -1 if it no longer exists."""
        try:
            row = con.execute(f"SELECT count(*) FROM {_quote(name)}").fetchone()  # noqa: S608
        except duckdb.Error:
            return -1
        return int(row[0]) if row else -1

    def _take_snapshot(
        self, con: duckdb.DuckDBPyConnection, name: str, *, content: bool
    ) -> CatalogSnapshot:
        """Snapshot the catalog, the row count and, if ``content``, a hash of all values."""
        objects = frozenset(
            (str(kind), str(obj), str(detail))
            for kind, obj, detail in con.execute(_CATALOG_SQL).fetchall()
        )
        rows = self._count_rows(con, name)
        fingerprint = None
        if content and rows >= 0:
            # The sum is order-independent and, unlike XOR, does not cancel duplicate rows.
            # ``name`` is validated in ``load`` and quoted here.
            query = (
                "SELECT COALESCE(SUM(CAST(hash(__row) AS HUGEINT)), 0) "  # noqa: S608
                f"FROM {_quote(name)} AS __row"
            )
            hashed = con.execute(query).fetchone()
            fingerprint = int(hashed[0]) if hashed else None
        return CatalogSnapshot(objects, rows, fingerprint)

    def verify_catalog(self) -> None:
        """Raise ``SessionCorruptedError`` if the database no longer matches its snapshot.

        The executor calls this after every query. On a mismatch the session is
        marked corrupted: it refuses further use and the user is asked to upload
        the file again.
        """
        with self.lock:
            self._require_open()
            assert self._con is not None and self._snapshot is not None  # noqa: S101
            try:
                current = self._take_snapshot(
                    self._con, self._table_name or "", content=self._content_checked
                )
                changed = self._snapshot.differences(current)
            except duckdb.Error as exc:
                changed = ["unreadable"]
                detail = f"changed: unreadable ({type(exc).__name__})"
            else:
                detail = f"changed: {', '.join(changed)}"
            if changed:
                self._corrupted = True
                log.error("catalog_changed", extra={"changed": changed})
                raise SessionCorruptedError(detail)

    # ----- use -----

    def query_df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Run SQL written by this application (not the language model) and return a dataframe."""
        with self.lock:
            con = self.connection
            self.touch()
            return con.execute(sql, list(params) if params is not None else None).df()

    def interrupt(self) -> None:
        """Stop a running query. Safe to call from another thread, and does not take the lock."""
        con = self._con
        if con is not None:
            con.interrupt()

    # ----- lifetime -----

    def touch(self) -> None:
        """Record activity, which resets the idle timer."""
        self._last_active = self._clock()

    def idle_seconds(self) -> float:
        """Seconds since the session was last used."""
        return self._clock() - self._last_active

    def is_expired(self) -> bool:
        """True when the session has been idle longer than ``session_idle_minutes``."""
        return self.idle_seconds() > self._settings.session_idle_minutes * 60

    def close(self) -> None:
        """Release the database and forget the table. Safe to call more than once."""
        with self.lock:
            if self._con is not None:
                self._con.close()
            self._con = None
            self._table_name = None
            self._columns = ()
            self._snapshot = None
            self._content_checked = False

    def __enter__(self) -> SessionDatabase:
        """Use as a context manager so the database is always closed."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the database."""
        self.close()
