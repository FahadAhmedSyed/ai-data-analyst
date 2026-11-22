"""Structured JSON logging.

Log records carry metadata only: timings, counts, categories and hashes.
Callers must never pass dataset values, sample rows, result rows, answer text
or secrets as log fields (see docs/security.md, threat T10).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Attributes present on every LogRecord; anything else came from ``extra``.
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

ROOT_LOGGER_NAME = "analyst"


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record, its ``extra`` fields and any exception as JSON."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", stream: Any = None) -> logging.Logger:
    """Configure the ``analyst`` logger to write JSON lines.

    Safe to call more than once (Streamlit reruns the script on every
    interaction): existing handlers are replaced, not duplicated.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level.upper())
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``analyst`` logger, for example ``get_logger(__name__)``."""
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def sql_hash(sql: str) -> str:
    """Short, stable fingerprint of a SQL string, for logs that must not contain the SQL."""
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:12]
