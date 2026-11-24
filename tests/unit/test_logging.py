from __future__ import annotations

import io
import json
import logging

from analyst.core.logging import configure_logging, get_logger, sql_hash


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_emits_json_with_extra_fields():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger("pipeline").info("query_executed", extra={"row_count": 12, "duration_ms": 8})
    [record] = _lines(stream)
    assert record["event"] == "query_executed"
    assert record["level"] == "INFO"
    assert record["logger"] == "analyst.pipeline"
    assert record["row_count"] == 12
    assert record["duration_ms"] == 8
    assert record["ts"].endswith("+00:00")


def test_respects_level():
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    log = get_logger("x")
    log.info("hidden")
    log.warning("shown")
    assert [r["event"] for r in _lines(stream)] == ["shown"]


def test_reconfiguring_does_not_duplicate_handlers():
    stream = io.StringIO()
    for _ in range(3):
        configure_logging("INFO", stream=stream)
    get_logger("x").info("once")
    assert len(_lines(stream)) == 1
    assert len(logging.getLogger("analyst").handlers) == 1


def test_exception_is_serialised():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("x").exception("failed")
    [record] = _lines(stream)
    assert record["exc_type"] == "ValueError"
    assert "ValueError" in record["exc"]  # the type and the place are logged ...
    assert "boom" not in record["exc"]  # ... the message is not (security T10)


def test_get_logger_names():
    assert get_logger("analyst").name == "analyst"
    assert get_logger("analyst.sql").name == "analyst.sql"
    assert get_logger("sql").name == "analyst.sql"


def test_sql_hash_is_stable_and_short():
    assert sql_hash("SELECT 1") == sql_hash("SELECT 1")
    assert sql_hash("SELECT 1") != sql_hash("SELECT 2")
    assert len(sql_hash("SELECT 1")) == 12


def test_an_exception_message_is_never_logged():
    """A message can hold a value from the data (a parse error quotes the SQL around it)."""
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        raise ValueError("Invalid expression near WHERE name = 'Alice Smith' (secret-value-123)")
    except ValueError:
        get_logger("x").exception("failed")
    text = stream.getvalue()
    assert "secret-value-123" not in text
    assert "Alice Smith" not in text
    [record] = _lines(stream)
    assert record["exc_type"] == "ValueError"
    assert "test_an_exception_message_is_never_logged" in record["exc"]  # where, not what


def test_a_chained_exception_message_is_not_logged_either():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        try:
            raise KeyError("cell-value-456")
        except KeyError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError:
        get_logger("x").exception("failed")
    text = stream.getvalue()
    assert "cell-value-456" not in text
    [record] = _lines(stream)
    assert "KeyError" in record["exc"]
    assert "RuntimeError" in record["exc"]
