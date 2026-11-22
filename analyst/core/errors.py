"""Error types. Each carries a category for logs and a message that is safe to show users.

The technical ``detail`` goes to logs only; ``user_message`` is what the UI shows.
User messages never include stack traces, file paths, SQL engine internals or
dataset values.
"""

from __future__ import annotations

from typing import ClassVar, Literal

IngestionCategory = Literal[
    "unsupported_type",
    "too_large",
    "empty",
    "corrupt",
    "encoding",
    "too_many_rows",
    "too_many_columns",
    "no_header",
]

ExecutionCategory = Literal["timeout", "out_of_memory", "binder", "conversion", "other"]

LLMCategory = Literal["not_configured", "timeout", "rate_limited", "api_error", "invalid_output"]


class AnalystError(Exception):
    """Base class for all expected application errors."""

    default_messages: ClassVar[dict[str, str]] = {}
    fallback_message: ClassVar[str] = "Something went wrong. Please try again."

    def __init__(self, category: str, detail: str = "", *, user_message: str | None = None) -> None:
        """Create an error.

        Args:
            category: Machine-readable reason, used in logs and tests.
            detail: Technical detail for logs. Never shown to users.
            user_message: Overrides the default message for this category.
        """
        self.category = category
        self.detail = detail
        self.user_message = (
            user_message or self.default_messages.get(category) or self.fallback_message
        )
        super().__init__(f"{category}: {detail}" if detail else category)


class IngestionError(AnalystError):
    """A file could not be read or loaded."""

    default_messages: ClassVar[dict[str, str]] = {
        "unsupported_type": "This file type isn't supported. Please upload a CSV or Excel file "
        "(.csv, .txt, .xlsx or .xls).",
        "too_large": "This file is larger than the upload limit.",
        "empty": "This file has no data rows.",
        "corrupt": "This file couldn't be read. It may be damaged or not really the type its "
        "name suggests.",
        "encoding": "The text encoding of this file couldn't be detected. Try saving it as UTF-8.",
        "too_many_rows": "This file has more rows than the current limit.",
        "too_many_columns": "This file has more columns than the current limit.",
        "no_header": "No header row was found. Choose the header row manually.",
    }
    fallback_message: ClassVar[str] = "This file couldn't be loaded."

    def __init__(
        self, category: IngestionCategory, detail: str = "", *, user_message: str | None = None
    ) -> None:
        """Create an ingestion error for one of the known categories."""
        super().__init__(category, detail, user_message=user_message)


class QueryExecutionError(AnalystError):
    """A validated query failed while running in DuckDB."""

    default_messages: ClassVar[dict[str, str]] = {
        "timeout": "The query took too long and was stopped.",
        "out_of_memory": "The query needed more memory than is available.",
        "binder": "The query referred to something that doesn't exist in the data.",
        "conversion": "The query tried to treat a value as the wrong type.",
        "other": "The query failed to run.",
    }

    def __init__(
        self,
        category: ExecutionCategory,
        detail: str = "",
        *,
        user_message: str | None = None,
        engine_message: str = "",
    ) -> None:
        """Create an execution error for one of the known categories.

        ``engine_message`` is the database's own message, for the corrector that repairs
        the query. It can quote values from the data, so it is never logged or shown.
        """
        super().__init__(category, detail, user_message=user_message)
        self.engine_message = engine_message


class LLMError(AnalystError):
    """The language model could not be reached or returned unusable output."""

    default_messages: ClassVar[dict[str, str]] = {
        "not_configured": "Question answering isn't set up yet: the OpenAI key or model "
        "names are missing.",
        "timeout": "The language model took too long to respond. Please try again.",
        "rate_limited": "The language model is busy. Please wait a moment and try again.",
        "api_error": "The language model service returned an error. Please try again.",
        "invalid_output": "The language model returned an answer in an unexpected form.",
    }

    def __init__(
        self, category: LLMCategory, detail: str = "", *, user_message: str | None = None
    ) -> None:
        """Create an LLM error for one of the known categories."""
        super().__init__(category, detail, user_message=user_message)


class BudgetExceededError(AnalystError):
    """A per-question limit on LLM calls or corrections was reached."""

    default_messages: ClassVar[dict[str, str]] = {
        "llm_calls": "This question needed more steps than allowed. Try asking it more simply.",
        "corrections": "The query couldn't be fixed within the allowed number of attempts.",
    }

    def __init__(
        self,
        category: Literal["llm_calls", "corrections"],
        detail: str = "",
        *,
        user_message: str | None = None,
    ) -> None:
        """Create a budget error for LLM calls or correction attempts."""
        super().__init__(category, detail, user_message=user_message)


class RateLimitedError(AnalystError):
    """The session asked too many questions in the rate-limit window."""

    default_messages: ClassVar[dict[str, str]] = {
        "questions": "You've reached the question limit for now. Please wait a few minutes.",
    }

    def __init__(self, detail: str = "", *, user_message: str | None = None) -> None:
        """Create a rate-limit error."""
        super().__init__("questions", detail, user_message=user_message)


class SessionCorruptedError(AnalystError):
    """The session's table changed unexpectedly (spike note A1)."""

    default_messages: ClassVar[dict[str, str]] = {
        "catalog_changed": "Your data changed unexpectedly, so answers can't be trusted. "
        "Please upload the file again.",
    }

    def __init__(self, detail: str = "", *, user_message: str | None = None) -> None:
        """Create a session-corrupted error."""
        super().__init__("catalog_changed", detail, user_message=user_message)
