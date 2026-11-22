"""Application settings loaded from environment variables (ADR 0014).

Every limit in the application is defined here, so a deployment can change
behaviour without code changes. Values come from the environment or a local
``.env`` file; see ``.env.example`` for the full list.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MEMORY_PATTERN = re.compile(r"^\d+(\.\d+)?\s?(KB|MB|GB|KiB|MiB|GiB)$", re.IGNORECASE)

Region = Literal["NZ"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class ModelPrice(BaseModel):
    """What one model costs, in US dollars per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    cached_input_per_mtok: float | None = Field(
        default=None, ge=0, description="Price of cached input tokens; the input price if unset"
    )


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Credentials. Optional so the app and tests run without them; the LLM
    # client and the password gate check for them when they are needed.
    openai_api_key: SecretStr | None = None
    app_password: SecretStr | None = None

    # Models per pipeline stage, chosen after the model comparison spike (ADR 0004).
    llm_model_generate: str | None = None
    llm_model_explain: str | None = None
    llm_timeout_s: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    # Prices by model name, as JSON, for the cost estimate in the logs. A model with no
    # entry is logged with cost 0 and ``cost_known`` false. Filled in from the model
    # comparison spike, because prices change faster than code.
    llm_prices: dict[str, ModelPrice] = Field(default_factory=dict)

    # Upload limits.
    max_upload_mb: int = Field(default=50, gt=0, le=500)
    max_rows: int = Field(default=2_000_000, gt=0)
    max_columns: int = Field(default=300, gt=0, le=2_000)

    # Query execution limits.
    query_timeout_s: float = Field(default=10.0, gt=0, le=120)
    max_result_rows: int = Field(default=10_000, gt=0, le=1_000_000)
    duckdb_memory_limit: str = "1GB"
    duckdb_threads: int = Field(default=2, ge=1, le=64)

    # Pipeline budgets per question.
    max_corrections: int = Field(default=2, ge=0, le=5)
    max_llm_calls: int = Field(default=5, ge=1, le=20)

    # Abuse protection and session lifetime.
    rate_limit_questions: int = Field(default=20, gt=0)
    rate_limit_window_s: int = Field(default=600, gt=0)
    session_idle_minutes: int = Field(default=60, gt=0)

    # Region rules (ADR 0009).
    default_region: Region = "NZ"

    # Logging.
    log_level: LogLevel = "INFO"
    log_sql_text: bool = False

    @field_validator("duckdb_memory_limit")
    @classmethod
    def _check_memory_limit(cls, value: str) -> str:
        """Accept only sizes DuckDB understands, such as ``512MB`` or ``1GB``."""
        if not _MEMORY_PATTERN.match(value.strip()):
            msg = "duckdb_memory_limit must look like '512MB' or '1GB'"
            raise ValueError(msg)
        return value.strip().upper().replace(" ", "")

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        """Allow lower-case log levels in the environment."""
        return value.upper() if isinstance(value, str) else value

    @property
    def max_upload_bytes(self) -> int:
        """Upload limit in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def password_required(self) -> bool:
        """True when a non-empty shared password is configured."""
        return bool(self.app_password and self.app_password.get_secret_value())

    @property
    def llm_configured(self) -> bool:
        """True when an OpenAI key and both stage models are configured."""
        return bool(
            self.openai_api_key
            and self.openai_api_key.get_secret_value()
            and self.llm_model_generate
            and self.llm_model_explain
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once.

    Settings hold configuration only, never user data, so sharing one
    instance across sessions is safe.
    """
    return Settings()
