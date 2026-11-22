"""Explain a query result in words, and only in words the result supports (T1.14, ADR 0017).

``Explainer.explain`` asks the model what a result shows, then checks the answer in code
before anyone sees it:

1. The model is shown the question, how it was read, and the rows (``ContextBuilder.
   explanation_context``) and returns text, key numbers and caveats.
2. Markdown that could send data out or mislead is removed from that text (links, images,
   web addresses), and the text is checked with ``check_grounding``: every number in it must
   come from the result, the question or the assumptions. Numbers spelled as words from
   eleven up are refused too, because the check cannot read them.
3. If the check fails, the model is asked once more, told which numbers were not accepted.
4. If that fails too, or the call cannot be completed, the user gets the plain template
   answer written by code (``template_answer``) with the table, never an unchecked answer.

An empty result needs no model: its answer is written by code and names the filters.

Whatever happens, the question is not failed because the explanation was: the query already
ran, and its table is the answer. The reason for a fallback is recorded so the benchmark can
report the fallback rate (ADR 0017).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from analyst.core.errors import BudgetExceededError, LLMError
from analyst.core.logging import get_logger
from analyst.core.models import Intent, KeyNumber, ModelExplanation, QueryResult, Usage
from analyst.llm.client import CallBudget, LLMClient, Message
from analyst.pipeline.grounding import GroundingIssue, GroundingReport, check_grounding
from analyst.pipeline.result_notes import empty_result_answer, result_caveats, template_answer
from analyst.sql.context import ContextBuilder

log = get_logger(__name__)

MAX_ATTEMPTS = 2  # the first answer and one regeneration (ADR 0017)

FallbackReason = Literal["empty_result", "ungrounded", "llm_error", "budget"]


@dataclass(frozen=True)
class Explanation:
    """The explanation to show, and how it came about.

    Every number in ``answer`` was checked against the result, or was written by code
    from it, so there is nothing to flag as ungrounded.
    """

    answer: str  # Markdown, safe to render
    key_numbers: tuple[KeyNumber, ...]
    caveats: tuple[str, ...]
    source: Literal["model", "template"]
    fallback_reason: FallbackReason | None
    attempts: int  # model calls made
    usage: Usage
    issues: tuple[GroundingIssue, ...] = ()  # what the last rejected answer got wrong

    @property
    def fallback_used(self) -> bool:
        """True when the model's answer could not be used (an empty result is not a fallback)."""
        return self.fallback_reason not in {None, "empty_result"}


# -- text safety ---------------------------------------------------------------------------------

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)|!\[([^\]]*)\]\[[^\]]*\]")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)|\[([^\]]+)\]\[[^\]]*\]")
_REFERENCE_DEFINITION = re.compile(r"^[ ]{0,3}\[[^\]]+\]:[ ]*\S.*$", re.MULTILINE)
_AUTOLINK = re.compile(r"<(?:https?://|ftp://|mailto:)[^>\s]*>", re.IGNORECASE)
_ADDRESS = re.compile(
    r"(?:https?://|ftp://|www\.)\S*[^\s.,;:!?)\]\"']|\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE
)
_LINK_REMOVED = "[link removed]"


def sanitize_text(text: str) -> str:
    """Take links, images and web addresses out of text from the model.

    Streamlit renders images in Markdown, so an image address carrying data would send that
    data out as soon as the answer is shown (threat T12). Links are reduced to their text.
    HTML is not rendered by the UI, so it is left as text.
    """
    text = "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())
    text = _REFERENCE_DEFINITION.sub("", text)
    text = _IMAGE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _LINK.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _AUTOLINK.sub(_LINK_REMOVED, text)
    text = _ADDRESS.sub(_LINK_REMOVED, text)
    return text.strip()


_SPELLED_NUMBER = re.compile(
    r"(?<!\d )\b(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)


def spelled_numbers(text: str) -> list[str]:
    """Number words from eleven up, which ``check_grounding`` cannot read.

    ``1.2 million`` is a number the check reads, so a word that follows a digit is not
    counted. Words up to ten are not looked for: ``one`` and ``two`` are too common in
    ordinary sentences, and the prompt asks for digits.
    """
    return [m.group(0).lower() for m in _SPELLED_NUMBER.finditer(text)]


def _clean(explanation: ModelExplanation) -> ModelExplanation:
    return ModelExplanation(
        answer_markdown=sanitize_text(explanation.answer_markdown),
        key_numbers=[
            KeyNumber(value=k.value, column=k.column, label=sanitize_text(k.label))
            for k in explanation.key_numbers
        ],
        caveats=[c for c in (sanitize_text(c) for c in explanation.caveats) if c],
    )


# -- the explainer -------------------------------------------------------------------------------

_RETRY_GENERAL = (
    "Your answer was not accepted. Write it again, using only numbers that appear in the "
    "RESULT block, the question or the assumptions, and write every number as digits."
)
_RETRY_UNUSABLE = "Your last answer could not be used. Answer again in the format requested."


class Explainer:
    """Explains results for one session, with the session's context builder."""

    def __init__(self, client: LLMClient, builder: ContextBuilder | None = None) -> None:
        """Create an explainer.

        Args:
            client: The model client (``FakeLLMClient`` in tests).
            builder: The session's context builder, so the marker code is the same for the
                whole session; a new one is made if none is given.
        """
        self._client = client
        self._builder = builder or ContextBuilder()

    def explain(
        self,
        question: str,
        result: QueryResult,
        *,
        resolved_question: str | None = None,
        intent: Intent | None = None,
        sql: str | None = None,
        assumptions: Sequence[str] = (),
        budget: CallBudget | None = None,
    ) -> Explanation:
        """Explain ``result``, or fall back to the template answer.

        Args:
            question: The user's question.
            result: The executed result.
            resolved_question: The question restated on its own.
            intent: How the question was read.
            sql: The query that ran.
            assumptions: The assumptions shown with the answer.
            budget: The question's call budget; each model call is charged to it.
        """
        notes = tuple(result_caveats(result))
        if result.is_empty:
            return Explanation(
                answer=empty_result_answer(intent),
                key_numbers=(),
                caveats=notes,
                source="template",
                fallback_reason="empty_result",
                attempts=0,
                usage=Usage(),
            )
        context = self._builder.explanation_context(
            question,
            result,
            resolved_question=resolved_question,
            intent=intent,
            sql=sql,
            assumptions=assumptions,
        )
        opening = [Message("system", context.system), Message("user", context.user)]
        messages = list(opening)
        allowed_text = [a for a in (resolved_question, *assumptions) if a]
        usage = Usage()
        issues: tuple[GroundingIssue, ...] = ()
        attempts = 0
        reason: FallbackReason = "ungrounded"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.complete(
                    "explain", messages, ModelExplanation, budget=budget
                )
            except BudgetExceededError:
                reason = "budget"
                break
            except LLMError as error:
                attempts += 1
                usage += error.usage or Usage()
                if error.category == "invalid_output" and attempt < MAX_ATTEMPTS:
                    messages = [*opening, Message("user", _RETRY_UNUSABLE)]
                    continue
                reason = "llm_error"
                break
            attempts += 1
            usage += response.usage
            explanation = _clean(response.parsed)
            if not explanation.answer_markdown:
                reason = "llm_error"
                messages = [*opening, Message("user", _RETRY_UNUSABLE)]
                continue
            reason = "ungrounded"
            report = check_grounding(
                explanation,
                result,
                question=question,
                assumptions=allowed_text,
                extra_numbers=context.extra_numbers,
            )
            spelled = spelled_numbers(_written(explanation))
            if report.ok and not spelled:
                log.info(
                    "explanation accepted",
                    extra={"attempts": attempts, "regenerated": attempt > 1},
                )
                return Explanation(
                    answer=explanation.answer_markdown,
                    key_numbers=tuple(explanation.key_numbers),
                    caveats=_merge(notes, explanation.caveats),
                    source="model",
                    fallback_reason=None,
                    attempts=attempts,
                    usage=usage,
                )
            issues = report.issues
            messages = [
                *opening,
                Message("assistant", response.parsed.model_dump_json()),
                Message("user", _feedback(report, spelled)),
            ]
        log.warning(
            "explanation fell back to the template",
            extra={
                "reason": reason,
                "attempts": attempts,
                "ungrounded_numbers": len(issues),
            },
        )
        return Explanation(
            answer=template_answer(result, intent),
            key_numbers=(),
            caveats=notes,
            source="template",
            fallback_reason=reason,
            attempts=attempts,
            usage=usage,
            issues=issues,
        )


def _written(explanation: ModelExplanation) -> str:
    return "\n".join(
        [
            explanation.answer_markdown,
            *explanation.caveats,
            *(k.label for k in explanation.key_numbers),
        ]
    )


def _feedback(report: GroundingReport, spelled: list[str]) -> str:
    parts = [_RETRY_GENERAL]
    if not report.ok:
        parts.append(report.feedback())
    if spelled:
        parts.append("Write these as digits: " + ", ".join(dict.fromkeys(spelled)) + ".")
    return " ".join(parts)


def _merge(code_notes: tuple[str, ...], model_notes: list[str]) -> tuple[str, ...]:
    """Notes written by code first, then the model's, without repeats."""
    seen: set[str] = set()
    merged: list[str] = []
    for note in (*code_notes, *model_notes):
        key = note.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(note)
    return tuple(merged)
