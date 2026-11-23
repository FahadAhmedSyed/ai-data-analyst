"""Gold queries, run the way the application runs a model's query: validator, then executor.

That is deliberate. Running the gold SQL through the same checks measures the ADR 0013
threshold "gold SQL wrongly blocked by the validator" (0), and a gold query the validator
refuses is a problem with the case, not with the model.
"""

from __future__ import annotations

import pandas as pd

from analyst.core.errors import AnalystError
from analyst.database.executor import QueryExecutor
from analyst.sql.validator import SqlValidator
from eval.scoring import Expected, limit_free_sql, sort_columns


class GoldError(Exception):
    """A gold query that did not run. ``blocked`` is true when the validator refused it."""

    def __init__(self, sql: str, message: str, *, blocked: bool) -> None:
        super().__init__(message)
        self.sql = sql
        self.blocked = blocked


class GoldRunner:
    """Runs gold queries on one loaded dataset and remembers the answers."""

    def __init__(self, validator: SqlValidator, executor: QueryExecutor) -> None:
        self._validator = validator
        self._executor = executor
        self._cache: dict[str, Expected | GoldError] = {}

    def run(self, sql: str) -> pd.DataFrame:
        """The result of one query.

        Raises:
            GoldError: The validator refused it or it failed to run.
        """
        verdict = self._validator.validate(sql)
        if not verdict.ok:
            msg = f"the validator refused it ({verdict.category}): {verdict.reason}"
            raise GoldError(sql, msg, blocked=True)
        try:
            return self._executor.execute(verdict).dataframe
        except AnalystError as error:
            msg = f"it did not run ({getattr(error, 'category', type(error).__name__)})"
            raise GoldError(sql, msg, blocked=False) from None

    def expected(self, sql: str) -> Expected:
        """The gold result for ``sql``, with what ties at a LIMIT need.

        Raises:
            GoldError: The query or its LIMIT-free form did not run.
        """
        cached = self._cache.get(sql)
        if cached is None:
            try:
                cached = self._build(sql)
            except GoldError as error:
                cached = error
            self._cache[sql] = cached
        if isinstance(cached, GoldError):
            raise cached
        return cached

    def _build(self, sql: str) -> Expected:
        frame = self.run(sql)
        free = limit_free_sql(sql)
        unlimited = self.run(free) if free else None
        return Expected(
            sql=sql,
            frame=frame,
            unlimited=unlimited,
            sort_columns=sort_columns(sql, [str(c) for c in frame.columns]),
        )
