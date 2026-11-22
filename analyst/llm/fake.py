"""A scripted stand-in for the model client, for tests and for running without a key.

``FakeLLMClient`` answers each call with the next item of a script, so a test states
exactly what the "model" says and can check what it was asked. It follows the same
contract as ``OpenAIClient``: the call budget is charged first, a script item that does
not fit the requested model is ``LLMError("invalid_output")``, and any exception in
the script is raised as given. Running past the end of the script, or asking for a
stage the script did not expect, is a test bug and raises ``AssertionError``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from analyst.core.errors import LLMError
from analyst.core.models import Usage
from analyst.llm.client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    CallBudget,
    LLMResponse,
    Message,
    Stage,
    T,
)


@dataclass(frozen=True)
class FakeCall:
    """One request the fake received."""

    stage: Stage
    messages: tuple[Message, ...]
    response_model: type[BaseModel]
    max_output_tokens: int


# A script item is what the model "says" for one call:
# * a Pydantic instance of the requested model: returned as is;
# * a dict: validated against the requested model (a misfit is ``invalid_output``);
# * a str: parsed as raw JSON text, so malformed output can be simulated;
# * an exception: raised;
# * a callable: given the ``FakeCall``, returns any of the above;
# * a ``(stage, item)`` pair: the same, but the test fails if the call is for another stage.
ScriptItem = (
    BaseModel | Mapping[str, Any] | str | Exception | Callable[[FakeCall], Any] | tuple[Stage, Any]
)

DEFAULT_FAKE_USAGE = Usage(llm_calls=1, input_tokens=100, output_tokens=50)


class FakeLLMClient:
    """``LLMClient`` that replays a script."""

    def __init__(
        self,
        script: Sequence[ScriptItem] = (),
        *,
        usage: Usage | None = None,
        model: str = "fake-model",
    ) -> None:
        """Create a fake.

        Args:
            script: One item per expected call, in order.
            usage: What every successful call reports; one call, 100 tokens in and 50 out
                by default.
            model: Reported as ``LLMResponse.model``.
        """
        self._script = list(script)
        self._usage = usage or DEFAULT_FAKE_USAGE
        self._model = model
        self.calls: list[FakeCall] = []

    @property
    def remaining(self) -> int:
        """Script items not used yet."""
        return len(self._script)

    @property
    def stages(self) -> list[Stage]:
        """The stage of each call received, in order."""
        return [call.stage for call in self.calls]

    def add(self, *items: ScriptItem) -> None:
        """Append items to the script."""
        self._script.extend(items)

    def assert_exhausted(self) -> None:
        """Fail if the code under test made fewer calls than the script expected."""
        assert not self._script, f"{len(self._script)} scripted answers were never asked for"  # noqa: S101

    def complete(
        self,
        stage: Stage,
        messages: Sequence[Message],
        response_model: type[T],
        *,
        budget: CallBudget | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> LLMResponse[T]:
        """Record the call and answer it from the script."""
        if budget is not None:
            budget.charge()
        call = FakeCall(stage, tuple(messages), response_model, max_output_tokens)
        self.calls.append(call)
        if not self._script:
            raise AssertionError(f"unscripted {stage} call (call number {len(self.calls)})")
        item = self._script.pop(0)
        if isinstance(item, tuple):
            expected, item = item
            assert call.stage == expected, (  # noqa: S101
                f"call {len(self.calls)} was for stage {call.stage!r}, "
                f"but the script expected {expected!r}"
            )
        parsed = self._resolve(item, call, response_model)
        return LLMResponse(parsed=parsed, usage=self._usage, model=self._model)

    def _resolve(self, item: Any, call: FakeCall, response_model: type[T]) -> T:
        if isinstance(item, Exception):
            raise item
        if isinstance(item, BaseModel):
            if not isinstance(item, response_model):
                raise TypeError(
                    f"script item is a {type(item).__name__}, "
                    f"but the call asked for a {response_model.__name__}"
                )
            return item
        try:
            if isinstance(item, str):
                return response_model.model_validate_json(item)
            if isinstance(item, Mapping):
                return response_model.model_validate(item)
        except ValidationError:
            raise LLMError("invalid_output", "does not fit the requested model") from None
        if callable(item):
            return self._resolve(item(call), call, response_model)
        raise TypeError(f"unsupported script item: {type(item).__name__}")
