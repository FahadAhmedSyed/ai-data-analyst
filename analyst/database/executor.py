"""Query executor (T1.09): runs validated SQL with a timeout, a row cap and error categories.

Only text that came out of the validator is run. The validator already wrapped it as
``SELECT * FROM (<query>) LIMIT cap + 1``, so the database never builds more than one
row beyond the cap and the extra row tells the executor that the result was cut.

For each query, while holding the session lock:

1. Confirm on the session connection that the text is exactly one ``SELECT``. The
   validator makes the same check on a separate connection; repeating it at the point
   of execution means a forged or corrupted ``ValidationResult`` still cannot run a
   second statement or a statement of another kind.
2. Start a timer that calls ``interrupt()`` at the deadline (``query_timeout_s``).
3. Run the query and fetch the rows.
4. Verify the catalog (spike note A1). DuckDB has no read-only mode for an in-memory
   database, so this check is what notices any change to the table. It runs after
   every query, failed or not; a changed catalog raises ``SessionCorruptedError``
   and takes precedence over the query's own result or error.

Failures raise ``QueryExecutionError`` with one of the categories the corrector and the
UI understand. The error's ``detail`` (type name only) is safe for logs. The database's
own message goes in ``engine_message`` for the corrector, because it can quote values
from the data; it is never logged.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import duckdb

from analyst.config.settings import Settings, get_settings
from analyst.core.errors import ExecutionCategory, QueryExecutionError
from analyst.core.logging import get_logger, sql_hash
from analyst.core.models import QueryResult, ValidationResult

if TYPE_CHECKING:
    import pandas as pd

    from analyst.database.session_db import SessionDatabase

log = get_logger(__name__)

ENGINE_MESSAGE_CHARS = 600

_CATEGORIES: tuple[tuple[type[BaseException], ExecutionCategory], ...] = (
    (duckdb.OutOfMemoryException, "out_of_memory"),
    (MemoryError, "out_of_memory"),
    (duckdb.BinderException, "binder"),
    (duckdb.CatalogException, "binder"),
    (duckdb.ConversionException, "conversion"),
    (duckdb.TypeMismatchException, "conversion"),
    (duckdb.OutOfRangeException, "conversion"),
)


def classify_error(error: BaseException) -> ExecutionCategory:
    """Map a database exception to an execution category (timeouts are decided by the caller)."""
    for exception_type, category in _CATEGORIES:
        if isinstance(error, exception_type):
            return category
    return "other"


class _Deadline:
    """Calls ``interrupt`` once after ``seconds`` unless the query finishes first.

    The flag and the call share a lock with ``finish``, so the interrupt can never be
    issued after the query has been released: a late timer cannot hit the next query.
    """

    def __init__(self, seconds: float, interrupt: Callable[[], None]) -> None:
        self.fired = False
        self._finished = False
        self._interrupt = interrupt
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    def start(self) -> None:
        self._timer.start()

    def finish(self) -> None:
        with self._lock:
            self._finished = True
        self._timer.cancel()

    def _fire(self) -> None:
        with self._lock:
            if self._finished:
                return
            self.fired = True
            self._interrupt()


class QueryExecutor:
    """Runs validated queries on one session database."""

    def __init__(self, db: SessionDatabase, settings: Settings | None = None) -> None:
        """Create an executor for ``db``; limits default to the application settings."""
        self._db = db
        self._timeout_s = (settings or get_settings()).query_timeout_s

    @property
    def timeout_s(self) -> float:
        """Seconds a query may run before it is interrupted."""
        return self._timeout_s

    def execute(self, validation: ValidationResult) -> QueryResult:
        """Run a validated query.

        Args:
            validation: An accepted result from ``SqlValidator.validate``. Anything else
                is refused, because only validated SQL may run.

        Returns:
            The rows, capped at ``validation.max_rows``, with ``truncated`` set if the
            query had more.

        Raises:
            TypeError: ``validation`` is not a ``ValidationResult``.
            ValueError: ``validation`` is not an accepted result.
            QueryExecutionError: the query was stopped or failed.
            SessionCorruptedError: the session table changed, or the session was
                already marked corrupted.
        """
        if not isinstance(validation, ValidationResult):
            msg = "Only a ValidationResult from the validator can be executed"
            raise TypeError(msg)
        if not validation.ok or validation.safe_sql is None or validation.max_rows is None:
            msg = "Only validated SQL can be executed"
            raise ValueError(msg)
        sql, max_rows = validation.safe_sql, validation.max_rows
        digest = sql_hash(sql)

        with self._db.lock:
            connection = self._db.connection  # refuses a corrupted or empty session
            self._db.touch()
            started = time.perf_counter()
            frame: pd.DataFrame | None = None
            failure: QueryExecutionError | None = None
            try:
                frame = self._run(connection, sql)
            except QueryExecutionError as error:
                failure = error
            duration_ms = (time.perf_counter() - started) * 1000
            self._db.verify_catalog()  # raises SessionCorruptedError; wins over the result

        if failure is not None:
            log.warning(
                "query_failed",
                extra={
                    "category": failure.category,
                    "error_type": failure.detail,
                    "duration_ms": round(duration_ms, 1),
                    "sql_hash": digest,
                },
            )
            raise failure
        assert frame is not None  # noqa: S101 - set whenever there is no failure
        truncated = len(frame) > max_rows
        if truncated:
            frame = frame.iloc[:max_rows]
        log.info(
            "query_executed",
            extra={
                "row_count": len(frame),
                "truncated": truncated,
                "duration_ms": round(duration_ms, 1),
                "sql_hash": digest,
            },
        )
        return QueryResult(
            dataframe=frame,
            row_count=len(frame),
            truncated=truncated,
            duration_ms=duration_ms,
        )

    def _run(self, connection: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
        """Run one query under the deadline and turn any failure into a categorised error."""
        deadline = _Deadline(self._timeout_s, self._db.interrupt)
        try:
            self._require_single_select(connection, sql)
            deadline.start()
            return connection.execute(sql).df()
        except QueryExecutionError:
            raise
        except duckdb.InterruptException as error:
            category: ExecutionCategory = "timeout" if deadline.fired else "other"
            raise self._error(category, error) from None
        except Exception as error:
            raise self._error(classify_error(error), error) from None
        finally:
            deadline.finish()

    @staticmethod
    def _require_single_select(connection: duckdb.DuckDBPyConnection, sql: str) -> None:
        try:
            types = [
                str(s.type).rsplit(".", maxsplit=1)[-1] for s in connection.extract_statements(sql)
            ]
        except duckdb.Error as error:
            raise QueryExecutionError("other", type(error).__name__) from None
        if types != ["SELECT"]:
            raise QueryExecutionError("other", "not_a_single_select")

    @staticmethod
    def _error(category: ExecutionCategory, error: BaseException) -> QueryExecutionError:
        message = " ".join(str(error).split())[:ENGINE_MESSAGE_CHARS]
        return QueryExecutionError(category, type(error).__name__, engine_message=message)
