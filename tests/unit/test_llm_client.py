"""The OpenAI client (T1.10) against a mock HTTP transport.

No test here touches the network or needs a key. Responses are raw JSON in the shape the
Responses API sends, so these tests exercise the real SDK parsing, request building and
error mapping, with only the transport replaced.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from pydantic import BaseModel, ConfigDict

from analyst.config.settings import ModelPrice, Settings
from analyst.core.errors import BudgetExceededError, LLMError
from analyst.core.logging import JsonFormatter
from analyst.core.models import GenerationResult, ModelExplanation
from analyst.llm import client as client_module
from analyst.llm.client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    CallBudget,
    LLMClient,
    Message,
    OpenAIClient,
    estimate_cost,
)
from analyst.llm.fake import FakeLLMClient

KEY = "sk-fake-fake-fake-fake"
PROMPT_TEXT = "CONFIDENTIAL-PROMPT-TEXT revenue for Acme Ltd"
ANSWER_TEXT = "CONFIDENTIAL-ANSWER-TEXT"
PRICES = {
    "gen-model": ModelPrice(input_per_mtok=2.0, output_per_mtok=8.0, cached_input_per_mtok=0.5),
}


class Answer(BaseModel):
    """A small response model for most tests."""

    model_config = ConfigDict(extra="forbid")

    value: int
    note: str


GOOD = {"value": 7, "note": ANSWER_TEXT}
USAGE = {
    "input_tokens": 1_000,
    "input_tokens_details": {"cached_tokens": 400},
    "output_tokens": 200,
    "output_tokens_details": {"reasoning_tokens": 50},
    "total_tokens": 1_200,
}


def body(
    text: str | None = None,
    *,
    status: str = "completed",
    content: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = USAGE,
    output: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    """A Responses API payload. ``text`` is the JSON the model wrote."""
    if content is None:
        text = json.dumps(GOOD) if text is None else text
        content = [{"type": "output_text", "text": text, "annotations": []}]
    if output is None:
        output = [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": content,
            }
        ]
    payload: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "gen-model",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
        "error": error,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
    }
    return payload


def ok(text: str | None = None, **kwargs: Any) -> httpx2.Response:
    return httpx2.Response(200, json=body(text, **kwargs), headers={"x-request-id": "req_abc"})


def failure(status: int, *, code: str | None = None, retry_after: str | None = None) -> Any:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    error = {"message": "nope", "type": "error", "code": code}
    return httpx2.Response(status, json={"error": error}, headers=headers)


class Harness:
    """A client wired to a scripted transport, with the clock and sleeps recorded."""

    def __init__(self, replies: list[Any], settings: Settings) -> None:
        self.replies = list(replies)
        self.requests: list[httpx2.Request] = []
        self.sleeps: list[float] = []
        self.ticks = iter(range(0, 10_000))
        sdk = openai.OpenAI(
            api_key=KEY,
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self._handle)),
        )
        self.client = OpenAIClient(
            settings,
            sdk=sdk,
            sleep=self.sleeps.append,
            jitter=lambda: 0.0,
            clock=lambda: next(self.ticks) * 0.25,  # each reading is 250 ms after the last
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def sent(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def complete(self, stage: Any = "generate", model: type[BaseModel] = Answer, **kwargs: Any):
        return self.client.complete(stage, [Message("user", PROMPT_TEXT)], model, **kwargs)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "openai_api_key": KEY,
        "llm_model_generate": "gen-model",
        "llm_model_explain": "explain-model",
        "llm_timeout_s": 12.0,
        "llm_max_retries": 2,
        "max_llm_calls": 3,
        "llm_prices": PRICES,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "LLM_MODEL_GENERATE", "LLM_MODEL_EXPLAIN", "LLM_PRICES"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def harness() -> Callable[..., Harness]:
    def build(*replies: Any, **overrides: Any) -> Harness:
        return Harness(list(replies), make_settings(**overrides))

    return build


@pytest.fixture
def records() -> Iterator[list[logging.LogRecord]]:
    """Log records from the client module, captured directly so other tests' logging
    configuration cannot hide them."""
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = Capture(level=logging.DEBUG)
    logger = client_module.log
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield captured
    logger.removeHandler(handler)
    logger.setLevel(previous)


def events(records: list[logging.LogRecord]) -> dict[str, Any]:
    return {record.getMessage(): record for record in records}


# -- the request -----------------------------------------------------------------------


def test_request_goes_to_the_responses_endpoint_with_a_bearer_key(harness):
    h = harness(ok())
    h.complete()
    request = h.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/responses"
    assert request.headers["authorization"] == f"Bearer {KEY}"


def test_request_body_asks_for_strict_structured_output_and_stores_nothing(harness):
    h = harness(ok())
    h.complete(max_output_tokens=1234)
    sent = h.sent()
    assert sent["store"] is False
    assert sent["max_output_tokens"] == 1234
    assert sent["input"] == [{"role": "user", "content": PROMPT_TEXT}]
    assert "temperature" not in sent
    fmt = sent["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["name"] == "Answer"
    assert fmt["schema"]["required"] == ["value", "note"]
    assert fmt["schema"]["additionalProperties"] is False


def test_default_output_limit_applies(harness):
    h = harness(ok())
    h.complete()
    assert h.sent()["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_messages_keep_their_roles_and_order(harness):
    h = harness(ok())
    messages = [Message("system", "rules"), Message("user", "q1"), Message("assistant", "a1")]
    h.client.complete("generate", messages, Answer)
    assert h.sent()["input"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


@pytest.mark.parametrize(
    ("stage", "model"),
    [("generate", "gen-model"), ("correct", "gen-model"), ("explain", "explain-model")],
)
def test_each_stage_uses_its_configured_model(harness, stage, model):
    h = harness(ok())
    result = h.complete(stage)
    assert h.sent()["model"] == model
    assert result.model == model


def test_the_sdk_is_built_once_with_the_timeout_and_no_hidden_retries(monkeypatch):
    built: list[dict[str, Any]] = []

    class SpySDK:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)
            self.responses = SimpleNamespace(
                with_raw_response=SimpleNamespace(parse=self.parse),
            )

        def parse(self, **_: Any) -> Any:
            raise openai.APITimeoutError(httpx2.Request("POST", "https://x"))

    monkeypatch.setattr(client_module.openai, "OpenAI", SpySDK)
    client = OpenAIClient(make_settings(llm_timeout_s=9.5))
    assert built == []  # nothing is built until the first call
    for _ in range(2):
        with pytest.raises(LLMError):
            client.complete("generate", [Message("user", "q")], Answer)
    assert built == [{"api_key": KEY, "timeout": 9.5, "max_retries": 0}]


def test_each_request_carries_the_configured_timeout(harness):
    h = harness(ok())
    h.complete()
    timeout = h.requests[0].extensions["timeout"]
    assert timeout["read"] == 12.0
    assert timeout["connect"] == 12.0


# -- the answer, usage and cost ---------------------------------------------------------


def test_returns_the_parsed_model_with_usage(harness):
    h = harness(ok())
    result = h.complete()
    assert result.parsed == Answer(**GOOD)
    assert isinstance(result.parsed, Answer)
    assert result.usage.llm_calls == 1
    assert result.usage.input_tokens == 1_000
    assert result.usage.output_tokens == 200
    # The clock reads 0 s when the call starts and 0.25 s when it ends.
    assert result.usage.latency_ms == pytest.approx(250.0)


def test_cost_is_computed_from_the_configured_price_with_cached_tokens(harness):
    h = harness(ok())
    result = h.complete()
    # 600 uncached x $2 + 400 cached x $0.5 + 200 output x $8, per million tokens.
    assert result.usage.cost_usd == pytest.approx((600 * 2.0 + 400 * 0.5 + 200 * 8.0) / 1e6)
    assert result.usage.cost_usd == pytest.approx(0.003)


def test_cost_is_zero_and_flagged_when_the_model_has_no_price(harness, records):
    h = harness(ok(), llm_prices={})
    result = h.complete()
    assert result.usage.cost_usd == 0.0
    assert result.usage.input_tokens == 1_000
    assert events(records)["llm_call"].cost_known is False


def test_a_missing_usage_block_counts_as_zero(harness):
    h = harness(ok(usage=None))
    result = h.complete()
    assert (result.usage.input_tokens, result.usage.output_tokens) == (0, 0)
    assert result.usage.cost_usd == 0.0
    assert result.usage.llm_calls == 1


def test_a_usage_block_without_detail_fields_still_works(harness):
    h = harness(ok(usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}))
    result = h.complete()
    assert (result.usage.input_tokens, result.usage.output_tokens) == (10, 5)
    assert result.usage.cost_usd == pytest.approx((10 * 2.0 + 5 * 8.0) / 1e6)


@pytest.mark.parametrize(
    ("price", "tokens", "expected"),
    [
        (None, (1_000, 500, 0), 0.0),
        (ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0), (1_000_000, 0, 0), 1.0),
        (ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0), (0, 1_000_000, 0), 4.0),
        # No cached price: cached tokens cost the normal input price.
        (ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0), (1_000_000, 0, 600_000), 1.0),
        (
            ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0, cached_input_per_mtok=0.25),
            (1_000_000, 0, 600_000),
            0.4 * 1.0 + 0.6 * 0.25,
        ),
        # Cached cannot exceed the input; a bad count must not give a negative cost.
        (
            ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0, cached_input_per_mtok=0.25),
            (1_000, 0, 5_000),
            1_000 * 0.25 / 1e6,
        ),
        (ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0), (9, 9, 9), 0.0),
    ],
)
def test_estimate_cost(price, tokens, expected):
    assert estimate_cost(price, *tokens) == pytest.approx(expected)


def test_real_output_models_survive_the_strict_schema_and_parse(harness):
    """GenerationResult and ModelExplanation are what T1.12 and T1.14 will request."""
    generation = {
        "resolved_question": "Total revenue by region",
        "intent": {
            "operation": "aggregate",
            "measures": [{"column": "revenue", "aggregation": "sum"}],
            "dimensions": ["region"],
            "filters": [],
            "time_range": None,
            "sort": [],
            "limit": None,
            "output": "table",
        },
        "assumptions": [],
        "ambiguities": [],
        "needs_clarification": False,
        "clarification_question": None,
        "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
        "explanation": "Sums revenue per region.",
        "chart": None,
    }
    explanation = {
        "answer_markdown": "North leads.",
        "key_numbers": [{"value": 1.5, "column": "revenue", "label": "North revenue"}],
        "caveats": [],
    }
    h = harness(ok(json.dumps(generation)), ok(json.dumps(explanation)))
    first = h.complete(model=GenerationResult)
    second = h.complete("explain", model=ModelExplanation)
    assert first.parsed.sql == generation["sql"]
    assert second.parsed.key_numbers[0].value == 1.5
    for index in (0, 1):
        schema = h.sent(index)["text"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


# -- answers that cannot be used --------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        pytest.param(
            ok(content=[{"type": "refusal", "refusal": "I can't help with that."}]),
            "refusal",
            id="refusal",
        ),
        pytest.param(
            ok("{", status="incomplete", incomplete_reason="max_output_tokens"),
            "incomplete:max_output_tokens",
            id="truncated",
        ),
        pytest.param(
            ok("", status="incomplete", content=[]),
            "incomplete:unknown",
            id="truncated-without-reason",
        ),
        pytest.param(ok("this is not json"), "does not fit the requested model", id="not-json"),
        pytest.param(
            ok(json.dumps({"value": "seven", "note": "x"})),
            "does not fit the requested model",
            id="wrong-type",
        ),
        pytest.param(
            ok(json.dumps({"value": 1})), "does not fit the requested model", id="missing-field"
        ),
        pytest.param(
            ok(json.dumps({"value": 1, "note": "x", "extra": True})),
            "does not fit the requested model",
            id="extra-field",
        ),
        pytest.param(ok(output=[]), "empty or unparsed output", id="no-output"),
    ],
)
def test_unusable_answers_are_invalid_output_and_not_retried(harness, reply, detail):
    h = harness(reply)
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "invalid_output"
    assert raised.value.detail == detail
    assert len(h.requests) == 1


def test_a_response_level_error_is_an_api_error(harness):
    h = harness(ok(status="failed", error={"code": "server_error", "message": "boom"}))
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "api_error"
    assert len(h.requests) == 1


def test_a_refusal_never_leaks_its_text_into_the_error(harness):
    refusal = "I cannot discuss Acme Ltd's confidential revenue."
    h = harness(ok(content=[{"type": "refusal", "refusal": refusal}]))
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert "Acme" not in str(raised.value)
    assert "Acme" not in raised.value.detail
    assert "Acme" not in raised.value.user_message


def test_a_call_that_reached_the_model_reports_what_it_cost_even_when_unusable(harness):
    """A refusal, a cut-off answer or a misfit was billed, so the caller can count it."""
    for reply in (
        ok(content=[{"type": "refusal", "refusal": "no"}]),
        ok("{", status="incomplete", incomplete_reason="max_output_tokens"),
        ok("not json"),
        ok(output=[]),
        ok(status="failed", error={"code": "server_error", "message": "x"}),
    ):
        h = harness(reply)
        with pytest.raises(LLMError) as raised:
            h.complete()
        usage = raised.value.usage
        assert usage is not None
        assert (usage.llm_calls, usage.input_tokens, usage.output_tokens) == (1, 1_000, 200)
        assert usage.cost_usd == pytest.approx(0.003)
        assert usage.latency_ms == pytest.approx(250.0)


def test_errors_before_the_model_was_reached_carry_no_usage(harness):
    for reply in (failure(401), failure(503), httpx2.ReadTimeout("slow")):
        h = harness(reply, llm_max_retries=0)
        with pytest.raises(LLMError) as raised:
            h.complete()
        assert raised.value.usage is None


@pytest.mark.parametrize("content", [b"<html>gateway page</html>", b"", b"[1, 2]", b"{}"])
def test_a_body_that_is_not_a_response_is_an_api_error_without_usage(harness, content):
    h = harness(httpx2.Response(200, content=content))
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "api_error"
    assert raised.value.detail == "unreadable response body"
    assert raised.value.usage is None
    assert "gateway" not in str(raised.value)
    assert len(h.requests) == 1


def test_a_json_body_with_nothing_useful_in_it_is_invalid_output(harness):
    h = harness(httpx2.Response(200, json={"id": "resp_1", "object": "response"}))
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "invalid_output"
    assert raised.value.detail == "empty or unparsed output"


def test_reasoning_items_before_the_message_are_skipped(harness):
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": []}
    message = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": json.dumps(GOOD), "annotations": []}],
    }
    h = harness(ok(output=[reasoning, message]))
    assert h.complete().parsed == Answer(**GOOD)


def test_a_refusal_after_reasoning_items_is_still_found(harness):
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": []}
    message = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "no"}],
    }
    h = harness(ok(output=[reasoning, message]))
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.detail == "refusal"


# -- retries -----------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_codes_are_retried_then_succeed(harness, status):
    h = harness(failure(status), ok())
    result = h.complete()
    assert result.parsed.value == 7
    assert len(h.requests) == 2
    assert h.sleeps == [0.5]  # first backoff: half of the 1 s ceiling, no jitter


def test_backoff_doubles_up_to_the_cap(harness):
    h = harness(failure(503), failure(503), failure(503), failure(503), ok(), llm_max_retries=4)
    h.complete()
    # Ceiling 1, 2, 4, 8 s; each wait is half the ceiling when the jitter is zero.
    assert h.sleeps == [0.5, 1.0, 2.0, 4.0]


def test_backoff_never_exceeds_the_cap(harness):
    replies = [failure(503)] * 5 + [ok()]
    h = harness(*replies, llm_max_retries=5)
    h.complete()
    # Ceilings 1, 2, 4, 8, then 16 capped to 8; each wait is half the ceiling.
    assert h.sleeps == [0.5, 1.0, 2.0, 4.0, 4.0]
    assert max(h.sleeps) <= client_module.BACKOFF_CAP_S


def test_jitter_adds_up_to_half_the_ceiling(harness):
    h = harness(failure(503), ok())
    h.client._jitter = lambda: 0.999
    h.complete()
    assert h.sleeps == [pytest.approx(0.5 + 0.999 * 0.5)]
    assert h.sleeps[0] < 1.0


def test_a_connection_failure_is_retried(harness):
    h = harness(httpx2.ConnectError("refused"), ok())
    result = h.complete()
    assert result.parsed.value == 7
    assert len(h.requests) == 2


def test_retries_stop_after_the_configured_number_and_raise_the_last_error(harness):
    h = harness(failure(429), failure(429), failure(429), ok())
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "rate_limited"
    assert len(h.requests) == 3  # one try and two retries
    assert len(h.sleeps) == 2


def test_server_errors_that_persist_are_api_errors(harness):
    h = harness(failure(500), failure(500), failure(500))
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "api_error"
    assert raised.value.detail.endswith(":500")


def test_persistent_connection_failures_are_api_errors(harness):
    h = harness(httpx2.ConnectError("x"), httpx2.ConnectError("x"), httpx2.ConnectError("x"))
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "api_error"
    assert raised.value.detail == "APIConnectionError"
    assert len(h.requests) == 3


def test_zero_retries_means_one_attempt(harness):
    h = harness(failure(503), ok(), llm_max_retries=0)
    with pytest.raises(LLMError):
        h.complete()
    assert len(h.requests) == 1
    assert h.sleeps == []


def test_retry_after_longer_than_the_backoff_is_honoured(harness):
    h = harness(failure(429, retry_after="3"), ok())
    h.complete()
    assert h.sleeps == [3.0]


def test_retry_after_shorter_than_the_backoff_does_not_shorten_it(harness):
    h = harness(failure(429, retry_after="0"), ok())
    h.complete()
    assert h.sleeps == [0.5]


def test_retry_after_longer_than_the_limit_fails_at_once(harness):
    h = harness(failure(429, retry_after="60"), ok())
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "rate_limited"
    assert len(h.requests) == 1
    assert h.sleeps == []


def test_retry_after_at_the_limit_is_still_honoured(harness):
    h = harness(failure(429, retry_after="20"), ok())
    h.complete()
    assert h.sleeps == [20.0]


def test_an_unreadable_retry_after_falls_back_to_the_backoff(harness):
    h = harness(failure(429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT"), ok())
    h.complete()
    assert h.sleeps == [0.5]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(harness, status):
    h = harness(failure(status), ok())
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "api_error"
    assert raised.value.detail.endswith(f":{status}")
    assert len(h.requests) == 1
    assert h.sleeps == []


def test_an_exhausted_quota_is_not_retried(harness):
    h = harness(failure(429, code="insufficient_quota"), ok())
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "api_error"
    assert raised.value.detail.endswith("insufficient_quota")
    assert len(h.requests) == 1


def test_a_timeout_is_not_retried(harness):
    h = harness(httpx2.ReadTimeout("slow"), ok())
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "timeout"
    assert len(h.requests) == 1
    assert h.sleeps == []


def test_unexpected_sdk_errors_are_wrapped(harness, monkeypatch):
    h = harness()

    def explode(**_: Any) -> Any:
        raise openai.OpenAIError("something odd with sk-secret in it")

    monkeypatch.setattr(h.client._sdk.responses.with_raw_response, "parse", explode)
    with pytest.raises(LLMError) as raised:
        h.complete()
    assert raised.value.category == "api_error"
    assert raised.value.detail == "OpenAIError"
    assert "sk-secret" not in str(raised.value)


def test_error_messages_shown_to_users_never_name_the_key_or_prompt(harness):
    h = harness(failure(401))
    with pytest.raises(LLMError) as raised:
        h.complete()
    shown = f"{raised.value} {raised.value.detail} {raised.value.user_message}"
    assert KEY not in shown
    assert PROMPT_TEXT not in shown


# -- configuration -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "stage"),
    [
        ({"openai_api_key": None}, "generate"),
        ({"openai_api_key": ""}, "generate"),
        ({"llm_model_generate": None}, "generate"),
        ({"llm_model_generate": None}, "correct"),
        ({"llm_model_explain": None}, "explain"),
        ({"llm_model_generate": ""}, "generate"),
    ],
)
def test_missing_configuration_is_reported_before_any_request(harness, overrides, stage):
    h = harness(ok(), **overrides)
    with pytest.raises(LLMError) as raised:
        h.complete(stage)
    assert raised.value.category == "not_configured"
    assert h.requests == []


def test_only_the_stage_that_lacks_a_model_fails(harness):
    h = harness(ok(), llm_model_explain=None)
    assert h.complete("generate").parsed.value == 7


def test_missing_configuration_does_not_use_up_a_budgeted_call(harness):
    h = harness(ok(), openai_api_key=None)
    budget = CallBudget(3)
    with pytest.raises(LLMError):
        h.complete(budget=budget)
    assert budget.used == 0


def test_a_client_with_no_key_never_builds_the_sdk(monkeypatch):
    def forbidden(**_: Any) -> Any:
        raise AssertionError("the SDK must not be built without a key")

    monkeypatch.setattr(client_module.openai, "OpenAI", forbidden)
    client = OpenAIClient(make_settings(openai_api_key=None))
    with pytest.raises(LLMError) as raised:
        client.complete("generate", [Message("user", "q")], Answer)
    assert raised.value.category == "not_configured"


# -- the call budget ---------------------------------------------------------------------


def test_each_call_uses_one_unit_of_budget(harness):
    h = harness(ok(), ok())
    budget = CallBudget(3)
    h.complete(budget=budget)
    h.complete(budget=budget)
    assert (budget.used, budget.remaining) == (2, 1)


def test_retries_do_not_use_extra_budget(harness):
    h = harness(failure(503), failure(503), ok())
    budget = CallBudget(3)
    h.complete(budget=budget)
    assert len(h.requests) == 3
    assert budget.used == 1


def test_a_failed_call_still_counts(harness):
    h = harness(ok("not json"))
    budget = CallBudget(3)
    with pytest.raises(LLMError):
        h.complete(budget=budget)
    assert budget.used == 1


def test_the_call_after_the_last_one_allowed_is_refused_before_any_request(harness):
    h = harness(ok(), ok())
    budget = CallBudget(1)
    h.complete(budget=budget)
    with pytest.raises(BudgetExceededError) as raised:
        h.complete(budget=budget)
    assert raised.value.category == "llm_calls"
    assert len(h.requests) == 1
    assert budget.used == 1


def test_no_budget_means_no_limit(harness):
    h = harness(*[ok() for _ in range(10)])
    for _ in range(10):
        h.complete()
    assert len(h.requests) == 10


def test_budget_from_settings_uses_max_llm_calls():
    budget = CallBudget.from_settings(make_settings(max_llm_calls=4))
    assert (budget.max_calls, budget.remaining) == (4, 4)


def test_budget_from_default_settings(monkeypatch):
    monkeypatch.setattr(client_module, "get_settings", lambda: make_settings(max_llm_calls=2))
    assert CallBudget.from_settings().max_calls == 2


def test_remaining_never_goes_negative():
    budget = CallBudget(1)
    budget.used = 5
    assert budget.remaining == 0


# -- logging -----------------------------------------------------------------------------


def test_a_successful_call_logs_one_line_of_metadata(harness, records):
    h = harness(ok())
    h.complete("explain", model=Answer)
    assert [r.getMessage() for r in records] == ["llm_call"]
    record = records[0]
    assert record.levelno == logging.INFO
    assert record.stage == "explain"
    assert record.model == "explain-model"
    assert record.schema == "Answer"
    assert record.input_tokens == 1_000
    assert record.output_tokens == 200
    assert record.cached_tokens == 400
    assert record.reasoning_tokens == 50
    assert record.attempts == 1
    assert record.request_id == "req_abc"
    assert record.latency_ms == 250.0
    assert record.cost_usd == 0.0  # explain-model has no price in PRICES
    assert record.cost_known is False


def test_a_priced_call_logs_its_cost(harness, records):
    h = harness(ok())
    h.complete()
    record = events(records)["llm_call"]
    assert record.cost_usd == 0.003
    assert record.cost_known is True


def test_retries_are_logged_and_counted(harness, records):
    h = harness(failure(429), failure(503), ok())
    h.complete()
    messages = [r.getMessage() for r in records]
    assert messages == ["llm_retry", "llm_retry", "llm_call"]
    assert records[0].category == "rate_limited"
    assert records[1].category == "api_error"
    assert (records[0].attempt, records[1].attempt) == (1, 2)
    assert records[2].attempts == 3


def test_a_failed_call_logs_its_category_and_attempts(harness, records):
    h = harness(failure(401))
    with pytest.raises(LLMError):
        h.complete()
    assert [r.getMessage() for r in records] == ["llm_call_failed"]
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.category == "api_error"
    assert record.error_type == "AuthenticationError:401"
    assert record.attempts == 1
    assert record.stage == "generate"


def test_an_unusable_answer_is_logged_as_a_failed_call_with_its_tokens(harness, records):
    h = harness(ok("not json"))
    with pytest.raises(LLMError):
        h.complete()
    assert [r.getMessage() for r in records] == ["llm_call_failed"]
    record = records[0]
    assert record.category == "invalid_output"
    assert record.error_type == "does not fit the requested model"
    assert record.attempts == 1
    assert record.request_id == "req_abc"
    assert (record.input_tokens, record.output_tokens) == (1_000, 200)
    assert record.cost_usd == 0.003


def test_a_failure_before_the_model_was_reached_logs_no_tokens(harness, records):
    h = harness(failure(401))
    with pytest.raises(LLMError):
        h.complete()
    assert not hasattr(records[0], "input_tokens")
    assert not hasattr(records[0], "cost_usd")


def test_logs_never_contain_prompts_answers_or_the_key(harness, records):
    good = harness(ok())
    good.complete()
    refused = harness(ok(content=[{"type": "refusal", "refusal": ANSWER_TEXT}]))
    with pytest.raises(LLMError):
        refused.complete()
    rejected = harness(failure(401))
    with pytest.raises(LLMError):
        rejected.complete()
    assert records
    formatter = JsonFormatter()
    text = "\n".join(formatter.format(record) for record in records)
    for secret in (KEY, PROMPT_TEXT, ANSWER_TEXT, "Acme", "confidential"):
        assert secret.lower() not in text.lower(), secret


# -- the shared interface ----------------------------------------------------------------


def test_both_clients_satisfy_the_protocol():
    assert isinstance(OpenAIClient(make_settings()), LLMClient)
    assert isinstance(FakeLLMClient(), LLMClient)


def test_the_client_reads_default_settings_when_none_are_given(monkeypatch):
    monkeypatch.setattr(client_module, "get_settings", lambda: make_settings(llm_timeout_s=3.0))
    assert OpenAIClient()._settings.llm_timeout_s == 3.0


def test_messages_are_immutable():
    message = Message("user", "hi")
    with pytest.raises(AttributeError):
        message.content = "changed"  # type: ignore[misc]
