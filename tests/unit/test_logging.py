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
    assert "boom" in record["exc"]


def test_get_logger_names():
    assert get_logger("analyst").name == "analyst"
    assert get_logger("analyst.sql").name == "analyst.sql"
    assert get_logger("sql").name == "analyst.sql"


def test_sql_hash_is_stable_and_short():
    assert sql_hash("SELECT 1") == sql_hash("SELECT 1")
    assert sql_hash("SELECT 1") != sql_hash("SELECT 2")
    assert len(sql_hash("SELECT 1")) == 12
