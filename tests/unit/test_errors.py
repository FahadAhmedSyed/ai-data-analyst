from __future__ import annotations

import pytest

from analyst.core.errors import (
    AnalystError,
    BudgetExceededError,
    IngestionError,
    LLMError,
    QueryExecutionError,
    RateLimitedError,
    SessionCorruptedError,
)

CASES = [
    (
        IngestionError,
        [
            "unsupported_type",
            "too_large",
            "empty",
            "corrupt",
            "encoding",
            "too_many_rows",
            "too_many_columns",
            "no_header",
        ],
    ),
    (QueryExecutionError, ["timeout", "out_of_memory", "binder", "conversion", "other"]),
    (LLMError, ["not_configured", "timeout", "rate_limited", "api_error", "invalid_output"]),
    (BudgetExceededError, ["llm_calls", "corrections"]),
]


@pytest.mark.parametrize(("cls", "categories"), CASES)
def test_every_category_has_its_own_user_message(cls, categories):
    messages = set()
    for category in categories:
        err = cls(category, "internal detail")
        assert isinstance(err, AnalystError)
        assert err.category == category
        assert err.user_message
        assert err.user_message != cls.fallback_message
        assert "internal detail" not in err.user_message
        messages.add(err.user_message)
    assert len(messages) == len(categories)
    assert set(cls.default_messages) == set(categories)


def test_detail_goes_to_str_not_user_message():
    err = QueryExecutionError("binder", 'Binder Error: column "wind" not found')
    assert "wind" in str(err)
    assert "wind" not in err.user_message


def test_user_message_override():
    err = IngestionError("too_large", user_message="Files must be 50 MB or smaller.")
    assert err.user_message == "Files must be 50 MB or smaller."


def test_str_without_detail():
    assert str(LLMError("timeout")) == "timeout"


def test_unknown_category_uses_fallback():
    err = AnalystError("mystery")
    assert err.user_message == AnalystError.fallback_message


def test_single_category_errors():
    assert RateLimitedError().category == "questions"
    assert "wait" in RateLimitedError().user_message
    assert SessionCorruptedError("table dropped").category == "catalog_changed"
    assert "upload" in SessionCorruptedError().user_message


def test_errors_can_be_raised_and_caught_as_base():
    with pytest.raises(AnalystError) as info:
        raise LLMError("rate_limited", "429")
    assert info.value.category == "rate_limited"
