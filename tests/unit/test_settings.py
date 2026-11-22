from __future__ import annotations

import pytest
from pydantic import ValidationError

from analyst.config.settings import ModelPrice, Settings, get_settings


def test_defaults_match_adr_0014():
    s = Settings(_env_file=None)
    assert s.max_upload_mb == 50
    assert s.max_upload_bytes == 50 * 1024 * 1024
    assert s.max_rows == 2_000_000
    assert s.max_columns == 300
    assert s.query_timeout_s == 10
    assert s.max_result_rows == 10_000
    assert s.duckdb_memory_limit == "1GB"
    assert s.duckdb_threads == 2
    assert s.max_corrections == 2
    assert s.max_llm_calls == 5
    assert s.rate_limit_questions == 20
    assert s.rate_limit_window_s == 600
    assert s.session_idle_minutes == 60
    assert s.default_region == "NZ"
    assert s.log_sql_text is False


def test_reads_environment(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "20")
    monkeypatch.setenv("QUERY_TIMEOUT_S", "5.5")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("DUCKDB_MEMORY_LIMIT", "512 mb")
    s = Settings(_env_file=None)
    assert s.max_upload_mb == 20
    assert s.query_timeout_s == 5.5
    assert s.log_level == "DEBUG"
    assert s.duckdb_memory_limit == "512MB"


def test_reads_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MAX_ROWS=1000\nAPP_PASSWORD=pw\n", encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.max_rows == 1000
    assert s.password_required is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_UPLOAD_MB", "0"),
        ("MAX_UPLOAD_MB", "-5"),
        ("QUERY_TIMEOUT_S", "0"),
        ("DUCKDB_THREADS", "0"),
        ("MAX_LLM_CALLS", "0"),
        ("DUCKDB_MEMORY_LIMIT", "lots"),
        ("DUCKDB_MEMORY_LIMIT", "1TB"),
        ("DEFAULT_REGION", "US"),
        ("LOG_LEVEL", "VERBOSE"),
    ],
)
def test_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_secrets_are_masked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("APP_PASSWORD", "hunter2")
    s = Settings(_env_file=None)
    text = repr(s) + str(s.model_dump())
    assert "sk-test-secret" not in text
    assert "hunter2" not in text
    assert s.openai_api_key.get_secret_value() == "sk-test-secret"


def test_password_required_only_when_non_empty(monkeypatch):
    assert Settings(_env_file=None).password_required is False
    monkeypatch.setenv("APP_PASSWORD", "")
    assert Settings(_env_file=None).password_required is False
    monkeypatch.setenv("APP_PASSWORD", "x")
    assert Settings(_env_file=None).password_required is True


def test_llm_configured_needs_key_and_both_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert Settings(_env_file=None).llm_configured is False
    monkeypatch.setenv("LLM_MODEL_GENERATE", "model-a")
    monkeypatch.setenv("LLM_MODEL_EXPLAIN", "model-b")
    assert Settings(_env_file=None).llm_configured is True


def test_settings_are_immutable():
    s = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        s.max_rows = 5


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_llm_prices_default_to_none_configured():
    assert Settings(_env_file=None).llm_prices == {}


def test_llm_prices_are_read_from_json(monkeypatch):
    monkeypatch.setenv(
        "LLM_PRICES",
        '{"model-a": {"input_per_mtok": 1.5, "output_per_mtok": 6, "cached_input_per_mtok": 0.2},'
        ' "model-b": {"input_per_mtok": 0.1, "output_per_mtok": 0.4}}',
    )
    prices = Settings(_env_file=None).llm_prices
    assert prices["model-a"] == ModelPrice(
        input_per_mtok=1.5, output_per_mtok=6.0, cached_input_per_mtok=0.2
    )
    assert prices["model-b"].cached_input_per_mtok is None


@pytest.mark.parametrize(
    "bad",
    [
        '{"m": {"input_per_mtok": -1, "output_per_mtok": 1}}',
        '{"m": {"input_per_mtok": 1}}',
        '{"m": {"input_per_mtok": 1, "output_per_mtok": 1, "typo_per_mtok": 3}}',
        '{"m": {"input_per_mtok": 1, "output_per_mtok": 1, "cached_input_per_mtok": -0.1}}',
        '{"m": 5}',
    ],
)
def test_llm_prices_reject_bad_entries(monkeypatch, bad):
    monkeypatch.setenv("LLM_PRICES", bad)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_model_price_is_immutable():
    price = ModelPrice(input_per_mtok=1, output_per_mtok=2)
    with pytest.raises(ValidationError):
        price.input_per_mtok = 5
