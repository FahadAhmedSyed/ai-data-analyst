"""Static checks on model-written SQL, and later the query executor."""

from analyst.sql.validator import SqlValidator

__all__ = ["SqlValidator"]
