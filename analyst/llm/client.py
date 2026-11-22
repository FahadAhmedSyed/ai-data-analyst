"""The language-model client (T1.10, ADR 0004).

Every model call in the application goes through ``LLMClient.complete``. The real
implementation, ``OpenAIClient``, uses the OpenAI Responses API with structured
outputs: the caller names a Pydantic model and gets an instance back, or an
``LLMError``. Around that one call it adds what each stage would otherwise repeat:

* **Configuration check.** No key or no model for the stage is ``not_configured``,
  raised before any network call. The key is a ``SecretStr`` and is never logged.
* **Per-question budget.** A ``CallBudget`` counts logical calls; the call after the
  last one allowed raises ``BudgetExceededError`` before anything is sent.
* **Privacy.** Every request sends ``store=false`` (ADR 0010).
* **Retries.** Rate limits (429) and server errors (5xx), and connection failures, are
  retried up to ``llm_max_retries`` times with exponential backoff and jitter, honouring
  ``Retry-After`` up to ``MAX_RETRY_WAIT_S``. Other 4xx errors are not retried, nor is a
  timeout (the user would wait several times as long), nor is an exhausted quota (a
  429 that retrying cannot fix). The SDK's own retries are switched off so this is the
  only place that retries and the only place that counts attempts.
* **Timeouts.** ``llm_timeout_s`` per attempt.
* **Cost and logging.** One ``llm_call`` log line per call with model, token counts,
  estimated cost, latency, attempts and the request id. Prompts, answers and
  dataset values are never logged (threat T10).

Errors raised are always ``LLMError`` (or ``BudgetExceededError``); an SDK exception
never escapes. ``invalid_output`` covers a refusal, a truncated answer and anything
that does not fit the requested model. It is not retried here: whether to ask again
is a decision for the stage that knows its budget. Such a call still cost tokens, so
the error carries them as ``LLMError.usage`` for the caller to add to the question's
total, and the ``llm_call_failed`` log line records them.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, TypeVar, runtime_checkable

import openai
from pydantic import BaseModel, ValidationError

from analyst.config.settings import ModelPrice, Settings, get_settings
from analyst.core.errors import BudgetExceededError, LLMCategory, LLMError
from analyst.core.logging import get_logger
from analyst.core.models import Usage

log = get_logger(__name__)

Stage = Literal["generate", "correct", "explain"]
Role = Literal["system", "user", "assistant"]

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_OUTPUT_TOKENS = 4_000
MAX_RETRY_WAIT_S = 20.0
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 8.0


@dataclass(frozen=True)
class Message:
    """One message of the prompt."""

    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse[T: BaseModel]:
    """A parsed answer with what it cost."""

    parsed: T
    usage: Usage
    model: str


class CallBudget:
    """Counts the model calls made for one question (``max_llm_calls`` in settings)."""

    def __init__(self, max_calls: int) -> None:
        """Allow ``max_calls`` calls."""
        self.max_calls = max_calls
        self.used = 0

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CallBudget:
        """A budget with the configured per-question limit."""
        return cls((settings or get_settings()).max_llm_calls)

    @property
    def remaining(self) -> int:
        """Calls still allowed."""
        return max(self.max_calls - self.used, 0)

    def charge(self) -> None:
        """Count one call, or raise ``BudgetExceededError`` if none is left."""
        if self.used >= self.max_calls:
            raise BudgetExceededError("llm_calls", f"limit of {self.max_calls} calls reached")
        self.used += 1


@runtime_checkable
class LLMClient(Protocol):
    """What the pipeline needs from a model client. ``FakeLLMClient`` is the test double."""

    def complete(
        self,
        stage: Stage,
        messages: Sequence[Message],
        response_model: type[T],
        *,
        budget: CallBudget | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> LLMResponse[T]:
        """Ask the model for an instance of ``response_model``."""
        ...


def estimate_cost(
    price: ModelPrice | None, input_tokens: int, output_tokens: int, cached_tokens: int = 0
) -> float:
    """Dollar cost of one call; 0.0 when the model's price is not configured.

    ``input_tokens`` includes the cached ones, which are billed at the cached price.
    Output tokens include reasoning tokens, which are billed as output.
    """
    if price is None:
        return 0.0
    cached = min(cached_tokens, input_tokens)
    cached_price = price.cached_input_per_mtok
    if cached_price is None:
        cached_price = price.input_per_mtok
    return (
        (input_tokens - cached) * price.input_per_mtok
        + cached * cached_price
        + output_tokens * price.output_per_mtok
    ) / 1_000_000


@dataclass
class _Call:
    """Bookkeeping for one logical call, which may take several attempts."""

    stage: Stage
    model: str
    schema: str
    started: float
    attempts: int = 0
    request_id: str | None = None


class _Failure(NamedTuple):
    """A failed attempt: the error to raise, whether another attempt may fix it, and any
    wait the server asked for."""

    error: LLMError
    retryable: bool
    retry_after: float | None = None


@dataclass(frozen=True)
class _Counts:
    input: int = 0
    output: int = 0
    cached: int = 0
    reasoning: int = 0


class OpenAIClient:
    """``LLMClient`` backed by the OpenAI Responses API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sdk: openai.OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create a client. Nothing is opened until the first call.

        Args:
            settings: Keys, models, timeout, retries and prices; defaults to the
                application settings.
            sdk: A ready ``openai.OpenAI`` object, for tests. It must have
                ``max_retries=0`` or retries would be doubled.
            sleep: Used between retries; replaced in tests.
            jitter: Returns a number in [0, 1); replaced in tests.
            clock: Monotonic seconds; replaced in tests.
        """
        self._settings = settings or get_settings()
        self._sdk = sdk
        self._sleep = sleep
        self._jitter = jitter
        self._clock = clock

    def complete(
        self,
        stage: Stage,
        messages: Sequence[Message],
        response_model: type[T],
        *,
        budget: CallBudget | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> LLMResponse[T]:
        """Ask the model for an instance of ``response_model``.

        Raises:
            LLMError: not configured, timed out, rate limited, API error, or an answer
                that is a refusal, truncated or does not fit ``response_model``.
            BudgetExceededError: the question has used all its model calls.
        """
        model = self._model_for(stage)  # not_configured, before the budget is touched
        if budget is not None:
            budget.charge()
        sdk = self._sdk_client()
        call = _Call(stage, model, response_model.__name__, started=self._clock())
        try:
            raw = self._request(sdk, call, messages, response_model, max_output_tokens)
            parsed, counts = self._read(raw, response_model, call)
        except LLMError as error:
            self._log_failure(call, error)
            raise
        usage = self._usage(counts, call)
        self._log_success(call, usage, counts)
        return LLMResponse(parsed=parsed, usage=usage, model=model)

    # -- configuration -----------------------------------------------------------------

    def _model_for(self, stage: Stage) -> str:
        settings = self._settings
        key = settings.openai_api_key
        if key is None or not key.get_secret_value():
            raise LLMError("not_configured", "no API key")
        model = settings.llm_model_explain if stage == "explain" else settings.llm_model_generate
        if not model:
            raise LLMError("not_configured", f"no model for stage {stage}")
        return model

    def _sdk_client(self) -> openai.OpenAI:
        if self._sdk is None:
            key = self._settings.openai_api_key
            assert key is not None  # noqa: S101 - checked by _model_for
            self._sdk = openai.OpenAI(
                api_key=key.get_secret_value(),
                timeout=self._settings.llm_timeout_s,
                max_retries=0,
            )
        return self._sdk

    # -- the call, with retries ----------------------------------------------------------

    def _request(
        self,
        sdk: openai.OpenAI,
        call: _Call,
        messages: Sequence[Message],
        response_model: type[BaseModel],
        max_output_tokens: int,
    ) -> Any:
        """Send the request until it gets an HTTP 200; return the SDK's raw response.

        The raw response is kept (rather than the SDK's parsed object) so that a body the
        SDK cannot parse can still be read for its status and token counts.
        """
        payload: list[Any] = [{"role": m.role, "content": m.content} for m in messages]
        max_attempts = 1 + self._settings.llm_max_retries
        for attempt in range(1, max_attempts + 1):
            call.attempts = attempt
            try:
                raw = sdk.responses.with_raw_response.parse(
                    model=call.model,
                    input=payload,
                    text_format=response_model,
                    store=False,
                    max_output_tokens=max_output_tokens,
                    timeout=self._settings.llm_timeout_s,
                )
            except openai.APITimeoutError as error:
                raise LLMError("timeout", type(error).__name__) from None
            except openai.APIConnectionError as error:
                failure = _Failure(LLMError("api_error", type(error).__name__), retryable=True)
            except openai.APIStatusError as error:
                call.request_id = error.request_id
                failure = self._status_failure(error)
            except openai.OpenAIError as error:
                raise LLMError("api_error", type(error).__name__) from None
            else:
                call.request_id = raw.headers.get("x-request-id")
                return raw
            if not failure.retryable or attempt == max_attempts:
                raise failure.error from None
            wait = self._wait(attempt, failure.retry_after)
            if wait is None:
                raise failure.error from None
            log.warning(
                "llm_retry",
                extra={
                    "stage": call.stage,
                    "category": failure.error.category,
                    "error_type": failure.error.detail,
                    "attempt": attempt,
                    "wait_s": round(wait, 2),
                },
            )
            self._sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _status_failure(error: openai.APIStatusError) -> _Failure:
        """Classify an HTTP error status. Only rate limits and server errors are retryable."""
        status = error.status_code
        detail = f"{type(error).__name__}:{status}"
        if status == 429:
            if getattr(error, "code", None) == "insufficient_quota":
                # Retrying cannot add credit.
                return _Failure(LLMError("api_error", f"{detail}:insufficient_quota"), False)
            return _Failure(LLMError("rate_limited", detail), True, _retry_after(error))
        if status >= 500:
            return _Failure(LLMError("api_error", detail), True, _retry_after(error))
        return _Failure(LLMError("api_error", detail), False)

    def _wait(self, attempt: int, retry_after: float | None) -> float | None:
        """Seconds to wait before the next attempt, or ``None`` if the server asks too long."""
        if retry_after is not None and retry_after > MAX_RETRY_WAIT_S:
            return None
        ceiling = min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_CAP_S)
        backoff = ceiling / 2 + self._jitter() * ceiling / 2
        return max(backoff, retry_after or 0.0)

    # -- reading the answer ------------------------------------------------------------------

    def _read(self, raw: Any, response_model: type[T], call: _Call) -> tuple[T, _Counts]:
        """Parse the raw response into ``response_model`` and count its tokens."""
        payload = _payload(raw)
        if not payload:  # for example a proxy's HTML page served with status 200
            raise LLMError("api_error", "unreadable response body")
        try:
            response = raw.parse()
        except ValidationError:
            # The text was cut off or does not fit. Say which, and what the call cost.
            counts = _counts(payload.get("usage"))
            if payload.get("status") == "incomplete":
                raise self._invalid(
                    f"incomplete:{_incomplete_reason(payload)}", counts, call
                ) from None
            raise self._invalid("does not fit the requested model", counts, call) from None
        counts = _counts(response.usage)
        if response.status == "incomplete":
            raise self._invalid(f"incomplete:{_incomplete_reason(response)}", counts, call)
        if response.error is not None:
            raise self._invalid("response_error", counts, call, category="api_error")
        for item in response.output or []:
            if getattr(item, "type", None) != "message":
                continue
            if any(getattr(part, "type", None) == "refusal" for part in item.content):
                raise self._invalid("refusal", counts, call)
        parsed = response.output_parsed
        if not isinstance(parsed, response_model):
            raise self._invalid("empty or unparsed output", counts, call)
        return parsed, counts

    def _invalid(
        self,
        detail: str,
        counts: _Counts,
        call: _Call,
        *,
        category: Literal["api_error", "invalid_output"] = "invalid_output",
    ) -> LLMError:
        """An error for a call that reached the model and so cost tokens."""
        return LLMError(category, detail, usage=self._usage(counts, call))

    def _usage(self, counts: _Counts, call: _Call) -> Usage:
        price = self._settings.llm_prices.get(call.model)
        return Usage(
            llm_calls=1,
            input_tokens=counts.input,
            output_tokens=counts.output,
            cost_usd=estimate_cost(price, counts.input, counts.output, counts.cached),
            latency_ms=(self._clock() - call.started) * 1000,
        )

    # -- logging -------------------------------------------------------------------------------

    def _log_success(self, call: _Call, usage: Usage, counts: _Counts) -> None:
        log.info(
            "llm_call",
            extra={
                "stage": call.stage,
                "model": call.model,
                "schema": call.schema,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_tokens": counts.cached,
                "reasoning_tokens": counts.reasoning,
                "cost_usd": round(usage.cost_usd, 6),
                "cost_known": call.model in self._settings.llm_prices,
                "latency_ms": round(usage.latency_ms, 1),
                "attempts": call.attempts,
                "request_id": call.request_id,
            },
        )

    def _log_failure(self, call: _Call, error: LLMError) -> None:
        fields: dict[str, Any] = {
            "stage": call.stage,
            "model": call.model,
            "schema": call.schema,
            "category": error.category,
            "error_type": error.detail,
            "latency_ms": round((self._clock() - call.started) * 1000, 1),
            "attempts": call.attempts,
            "request_id": call.request_id,
        }
        if error.usage is not None:  # the call reached the model, so it was billed
            fields["input_tokens"] = error.usage.input_tokens
            fields["output_tokens"] = error.usage.output_tokens
            fields["cost_usd"] = round(error.usage.cost_usd, 6)
        log.warning("llm_call_failed", extra=fields)


def _field(obj: Any, name: str) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` for an SDK object, ``None`` if absent."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _counts(usage: Any) -> _Counts:
    """Token counts from a usage block, whether a dict or an SDK object."""
    return _Counts(
        input=_field(usage, "input_tokens") or 0,
        output=_field(usage, "output_tokens") or 0,
        cached=_field(_field(usage, "input_tokens_details"), "cached_tokens") or 0,
        reasoning=_field(_field(usage, "output_tokens_details"), "reasoning_tokens") or 0,
    )


def _incomplete_reason(response: Any) -> str:
    return _field(_field(response, "incomplete_details"), "reason") or "unknown"


def _payload(raw: Any) -> dict[str, Any]:
    """The JSON body of a response the SDK could not parse; empty if it is not JSON."""
    try:
        payload = raw.http_response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _retry_after(error: openai.APIStatusError) -> float | None:
    """Seconds from a numeric ``Retry-After`` header, if there is one."""
    value = error.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "MAX_RETRY_WAIT_S",
    "CallBudget",
    "LLMCategory",
    "LLMClient",
    "LLMResponse",
    "Message",
    "OpenAIClient",
    "Role",
    "Stage",
    "estimate_cost",
]
