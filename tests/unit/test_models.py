from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from analyst.core.models import (
    AnalystResponse,
    ChartSpec,
    ColumnProfile,
    ColumnSchema,
    DatasetProfile,
    FilterCondition,
    GenerationResult,
    Intent,
    KeyNumber,
    Measure,
    ModelExplanation,
    QueryResult,
    SortKey,
    TimeRange,
    Turn,
    Usage,
    ValidationResult,
    ValueCount,
)


def make_intent(**overrides) -> Intent:
    data = {
        "operation": "aggregate",
        "measures": [{"column": "rainfall_mm", "aggregation": "avg"}],
        "dimensions": ["site"],
        "filters": [{"column": "date", "operator": "month_in", "values": [12, 1, 2]}],
        "time_range": None,
        "sort": [{"column": "avg_rainfall_mm", "direction": "desc"}],
        "limit": None,
        "output": "table",
    }
    data.update(overrides)
    return Intent.model_validate(data)


def make_generation(**overrides) -> GenerationResult:
    data = {
        "resolved_question": "Average rainfall per site, Dec to Feb, all years",
        "intent": make_intent().model_dump(),
        "assumptions": ["Summer = Dec to Feb (NZ default)"],
        "ambiguities": [],
        "needs_clarification": False,
        "clarification_question": None,
        "sql": "SELECT site, AVG(rainfall_mm) FROM rainfall GROUP BY site",
        "explanation": "Averages daily rainfall per site over summer months",
        "chart": {"type": "bar", "x": "site", "y": ["avg_rainfall_mm"], "color": None},
    }
    data.update(overrides)
    return GenerationResult.model_validate(data)


# ---------- OpenAI strict structured-output compatibility ----------


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


@pytest.mark.parametrize("model", [GenerationResult, ModelExplanation, Intent])
def test_llm_models_are_strict_schema_compatible(model):
    schema = to_strict_json_schema(model)
    for node in _walk(schema):
        if node.get("type") == "object" and "properties" in node:
            assert node.get("additionalProperties") is False
            assert set(node["required"]) == set(node["properties"]), node.get("title")
    # Free-form maps would break strict mode.
    assert 'additionalProperties": true' not in json.dumps(schema)


def test_generation_result_field_order_puts_intent_before_sql():
    fields = list(GenerationResult.model_fields)
    assert fields.index("intent") < fields.index("sql")
    assert fields.index("needs_clarification") < fields.index("sql")


def test_generation_result_field_order_asks_before_it_commits_to_a_reading():
    """Listing the ambiguities and deciding to ask come before the intent, which holds
    concrete filters and ranges: a model that has written a reading down stops asking."""
    fields = list(GenerationResult.model_fields)
    assert fields.index("ambiguities") < fields.index("needs_clarification")
    assert fields.index("needs_clarification") < fields.index("intent")


# ---------- LLM output models ----------


def test_intent_source_columns_excludes_sort_aliases():
    intent = make_intent()
    assert intent.source_columns() == {"rainfall_mm", "site", "date"}


def test_intent_count_star_is_not_a_column():
    intent = make_intent(measures=[{"column": "*", "aggregation": "count"}], filters=[])
    assert intent.source_columns() == {"site"}


def test_intent_rejects_unknown_operator_and_extra_fields():
    with pytest.raises(ValidationError):
        make_intent(filters=[{"column": "x", "operator": "LIKE", "values": []}])
    with pytest.raises(ValidationError):
        Intent.model_validate({**make_intent().model_dump(), "extra": 1})


def test_filter_values_accept_text_and_numbers():
    f = FilterCondition(column="site", operator="in", values=["Site A", 3.5])
    assert f.values == ["Site A", 3.5]


def test_generation_without_problems():
    assert make_generation().problems() == []


def test_generation_clarification_without_question_is_a_problem():
    g = make_generation(needs_clarification=True, clarification_question=" ", sql=None)
    assert g.problems() == ["needs_clarification is true but no clarification_question was given"]


def test_generation_without_sql_or_clarification_is_a_problem():
    g = make_generation(sql=None)
    assert g.problems() == ["no SQL was given and no clarification was requested"]


def test_generation_valid_clarification():
    g = make_generation(
        needs_clarification=True,
        clarification_question="What counts as high rainfall?",
        sql=None,
        chart=None,
    )
    assert g.problems() == []


def test_generation_round_trips_through_json():
    g = make_generation()
    assert GenerationResult.model_validate_json(g.model_dump_json()) == g


def test_model_explanation_parses():
    e = ModelExplanation(
        answer_markdown="Site B averaged 9.4 mm.",
        key_numbers=[KeyNumber(value=9.4, column="avg_rainfall_mm", label="Site B average")],
        caveats=[],
    )
    assert e.key_numbers[0].value == 9.4


def test_chart_and_parts_construct():
    assert ChartSpec(type="line", x="month", y=["total"], color=None).type == "line"
    assert Measure(column="x", aggregation="sum").aggregation == "sum"
    assert SortKey(column="x", direction="asc").direction == "asc"
    assert TimeRange(start="2024-01-01", end=None, description="from 2024").start == "2024-01-01"


# ---------- dataset profile ----------


def make_profile() -> DatasetProfile:
    site = ColumnProfile(
        sql_name="site",
        display_name="Site",
        dtype="text",
        null_count=0,
        distinct_count=2,
        top_values=(ValueCount(value="Site A", count=6), ValueCount(value="Site B", count=4)),
    )
    rain = ColumnProfile(
        sql_name="rainfall_mm",
        display_name="Rainfall (mm)",
        dtype="decimal",
        null_count=2,
        distinct_count=8,
        min_value=0.0,
        max_value=31.5,
        mean=7.2,
    )
    when = ColumnProfile(
        sql_name="date",
        display_name="Date",
        dtype="date",
        flags=("ambiguous_date",),
        null_count=0,
        distinct_count=10,
        date_min=date(2024, 1, 1),
        date_max=date(2024, 1, 10),
    )
    return DatasetProfile(
        table_name="rainfall",
        source_name="rainfall.xlsx",
        sheet_name="Daily",
        row_count=10,
        column_count=3,
        columns=(site, rain, when),
        primary_date_column="date",
        latest_date=date(2024, 1, 10),
    )


def test_profile_lookups():
    p = make_profile()
    assert p.column("rainfall_mm").display_name == "Rainfall (mm)"
    assert p.column("missing") is None
    assert p.column_names == {"site", "rainfall_mm", "date"}
    assert p.display_names["rainfall_mm"] == "Rainfall (mm)"
    assert p.column("rainfall_mm").null_fraction(p.row_count) == pytest.approx(0.2)
    assert p.column("site").null_fraction(0) == 0.0


def test_profile_round_trips_and_is_frozen():
    p = make_profile()
    assert DatasetProfile.model_validate_json(p.model_dump_json()) == p
    with pytest.raises(ValidationError):
        p.row_count = 5


@pytest.mark.parametrize("bad", ["", "x" * 64])
def test_column_sql_name_length_is_checked(bad):
    with pytest.raises(ValidationError):
        ColumnSchema(sql_name=bad, display_name="X", dtype="text")


def test_column_rejects_unknown_type_and_flag():
    with pytest.raises(ValidationError):
        ColumnSchema(sql_name="x", display_name="X", dtype="string")
    with pytest.raises(ValidationError):
        ColumnSchema(sql_name="x", display_name="X", dtype="text", flags=("weird",))


def test_profile_rejects_negative_counts():
    with pytest.raises(ValidationError):
        ColumnProfile(sql_name="x", display_name="X", dtype="text", null_count=-1, distinct_count=0)


# ---------- validation and query results ----------


def test_validation_result_ok_flag():
    ok = ValidationResult(status="ok", category="ok", safe_sql="SELECT 1")
    bad = ValidationResult(status="rejected", category="forbidden_statement", reason="DROP")
    assert ok.ok is True
    assert bad.ok is False


def test_query_result_summary_converts_values():
    df = pd.DataFrame(
        {
            "site": ["A", "B", None],
            "total": [np.float64(1.5), np.float64(np.nan), np.float64(3.0)],
            "n": np.array([1, 2, 3], dtype=np.int64),
            "day": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
        }
    )
    result = QueryResult(dataframe=df, row_count=3, truncated=True, duration_ms=4.2)
    summary = result.summary(max_rows=2)
    assert summary.columns == ("site", "total", "n", "day")
    assert summary.row_count == 3
    assert summary.truncated is True
    assert summary.preview_rows == (
        {"site": "A", "total": 1.5, "n": 1, "day": "2024-01-01T00:00:00"},
        {"site": "B", "total": None, "n": 2, "day": "2024-01-02T00:00:00"},
    )
    json.dumps(summary.model_dump(mode="json"))  # serialisable for memory


def test_query_result_summary_handles_dates_and_lists():
    df = pd.DataFrame({"d": [date(2024, 3, 1)], "ts": [datetime(2024, 3, 1, 9, 30)], "l": [[1, 2]]})
    summary = QueryResult(dataframe=df, row_count=1).summary()
    row = summary.preview_rows[0]
    assert row["d"] == "2024-03-01"
    assert row["ts"] == "2024-03-01T09:30:00"
    assert row["l"] == [1, 2]


def test_query_result_empty():
    result = QueryResult(dataframe=pd.DataFrame({"x": []}), row_count=0)
    assert result.is_empty
    assert result.summary().preview_rows == ()


# ---------- usage, turn, response ----------


def test_usage_adds_up():
    total = Usage(llm_calls=1, input_tokens=100, output_tokens=20, cost_usd=0.001, latency_ms=900)
    total = total + Usage(llm_calls=1, input_tokens=50, output_tokens=10, cost_usd=0.002)
    assert total.llm_calls == 2
    assert total.input_tokens == 150
    assert total.output_tokens == 30
    assert total.cost_usd == pytest.approx(0.003)
    assert total.latency_ms == 900


def test_turn_and_response():
    turn = Turn(question="Average rainfall?", status="answered", sql="SELECT 1")
    assert turn.created_at.tzinfo is not None
    assert Turn.model_validate_json(turn.model_dump_json()) == turn
    response = AnalystResponse(
        status="answered",
        question="Average rainfall?",
        message="It was 8.2 mm.",
        result=QueryResult(dataframe=pd.DataFrame({"x": [8.2]}), row_count=1),
        grounded=True,
        turn=turn,
    )
    assert response.usage.llm_calls == 0
    assert response.result.row_count == 1
    with pytest.raises(ValidationError):
        AnalystResponse(status="done", question="q", message="m", turn=turn)


def test_jsonable_edge_cases():
    from analyst.core.models import _jsonable

    assert _jsonable(None) is None
    assert _jsonable(pd.NaT) is None
    assert _jsonable("text") == "text"
    assert _jsonable(np.int32(7)) == 7
    assert _jsonable(pd.Timestamp("2024-05-01")) == "2024-05-01T00:00:00"
    assert _jsonable({"k": 1}) == {"k": 1}
