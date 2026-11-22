"""Check that every number in an explanation comes from the executed result (ADR 0017).

The model is asked to explain a result, never to calculate. This module is the code that
enforces it. ``check_grounding`` reads every number out of the explanation's text and key
numbers and matches each one against what the pipeline can vouch for:

* every numeric value in the result, the result's row and column counts, and any extra
  numbers the caller supplies (for example summary statistics it showed the model);
* the numbers in the user's question and in the stated assumptions;
* the year, month and day of every date in the result, and the numbers written inside
  text values and column names (``"18-24"``, ``"Tier 2"``, ``revenue_2024``).

A number a derived calculation produced (a difference, a ratio, a share) is not on that
list unless a query produced it, so it is caught. That is intended: derived values must
come from SQL.

How a written number is matched
-------------------------------

* Formats read: ``1234``, ``1,234.5``, ``$1,234``, ``12%``, ``12 percent``, ``0.5``,
  ``.5``, ``1e3``, ``1.2 million``, ``3k``, ``2bn``. A leading minus is ignored: ``-3.2``
  matches 3.2 or -3.2, so the direction of a change is not checked here.
* **Rounding.** A number matches if it is within half a unit of its last written digit of
  a value on the list: ``1,234.57`` and ``1,235`` both match 1234.5678, and ``1.2 million``
  matches 1,234,567. Trailing zeros are not treated as rounding: ``1,200`` does not match
  1,234 unless 1,200 is on the list. This is deliberately strict, and the benchmark's
  fallback rate will show whether it needs relaxing.
* **Percent.** ``12%`` matches 12 or 0.12 (a fraction shown as a percentage).
* **Not read:** numbers spelled as words ("three regions"), ordinals ("2nd"), and digits
  that are part of an identifier ("Q3", "p95", "v2"). The explainer's prompt asks for
  digits.
* List numbering at the start of a line (``1.``, ``2)``) and markdown heading numbers are
  not counted.

The check says nothing about whether a matched number is used correctly, only that it was
not made up.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

import numpy as np
import pandas as pd

from analyst.core.models import KeyNumber, ModelExplanation, QueryResult

_NUMBER = re.compile(
    r"""
    (?<![A-Za-z_\d.])                # not part of an identifier such as Q3, p95 or v2.1
    (?:\d{1,3}(?:,\d{3})+|\d+)        # 1234 or 1,234
    (?:\.\d+)?                        # decimals
    (?:[eE][+-]?\d+)?                 # exponent
    |
    (?<![A-Za-z_\d])\.\d+             # .5
    """,
    re.VERBOSE,
)
_IDENTIFIER_DIGITS = re.compile(r"[A-Za-z_](\d+)")  # the 2024 in revenue_2024, the 3 in Q3
_ORDINAL = re.compile(r"(?:st|nd|rd|th)\b", re.IGNORECASE)
_PERCENT = re.compile(r"\s?(?:%|percent\b|per cent\b|pct\b)", re.IGNORECASE)
_SCALE_WORD = re.compile(r"\s?(thousand|million|billion|trillion)\b", re.IGNORECASE)
_SCALE_ATTACHED = re.compile(r"(k|m|bn|b)\b", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*(?:#+\s*)?(?:[-*+]|\d+[.)])\s+", re.MULTILINE)

_WORD_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_ATTACHED_SCALES = {"k": 1e3, "m": 1e6, "bn": 1e9, "b": 1e9}

# A candidate reading of a written number: the value it stands for, and how far a stored
# value may be from it while still counting as the same number after rounding.
Candidate = tuple[float, float]

_EPSILON = 1e-9


@dataclass(frozen=True)
class NumberToken:
    """A number found in text, as written."""

    text: str
    candidates: tuple[Candidate, ...]


def _decimals(literal: str) -> int:
    mantissa = re.split(r"[eE]", literal, maxsplit=1)[0]
    return len(mantissa.split(".", 1)[1]) if "." in mantissa else 0


def extract_numbers(text: str) -> list[NumberToken]:
    """Every number written in ``text``, with the values each could stand for."""
    text = _LIST_MARKER.sub("", text)
    tokens: list[NumberToken] = []
    for match in _NUMBER.finditer(text):
        literal = match.group(0)
        after = text[match.end() :]
        if _ORDINAL.match(after):
            continue
        value = float(literal.replace(",", ""))
        half = 0.5 * 10 ** -_decimals(literal) if "e" not in literal.lower() else 0.0
        scales: list[float] = [1.0]
        percent = False
        shown = literal
        if word := _SCALE_WORD.match(after):
            scales = [_WORD_SCALES[word.group(1).lower()]]
            shown = f"{literal}{word.group(0)}"
        elif pct := _PERCENT.match(after):
            percent = True
            shown = f"{literal}{pct.group(0)}"
        elif attached := _SCALE_ATTACHED.match(after):
            # "5m" may be millions or metres, "3k" thousands: accept either reading.
            scales = [1.0, _ATTACHED_SCALES[attached.group(1).lower()]]
            shown = f"{literal}{attached.group(0)}"
        candidates: list[Candidate] = []
        for scale in scales:
            candidates.append((value * scale, half * scale))
            if percent:
                candidates.append((value / 100, half / 100))
        tokens.append(NumberToken(text=shown, candidates=tuple(candidates)))
    return tokens


def _date_parts(value: date | datetime) -> list[float]:
    return [float(value.year), float(value.month), float(value.day)]


def _numbers_in_text(text: str) -> list[float]:
    """The values that numbers written in ``text`` stand for, exactly as written.

    Used for text the user or the data supplied, so digits inside identifiers count too:
    they are part of what the text says even though an answer may not quote them bare.
    """
    values: list[float] = []
    for token in extract_numbers(text):
        values.extend(target for target, _ in token.candidates)
    values.extend(float(digits) for digits in _IDENTIFIER_DIGITS.findall(text))
    return values


class AllowedNumbers:
    """The numbers an explanation may use, with a fast test for a written number."""

    def __init__(self) -> None:
        """Create an empty list."""
        self._chunks: list[np.ndarray] = []
        self._sorted: np.ndarray | None = None

    def add_values(self, values: Iterable[float]) -> None:
        """Allow these numbers (their sign is ignored)."""
        source = values if isinstance(values, np.ndarray) else list(values)
        array = np.abs(np.asarray(source, dtype="float64"))
        if array.size:
            self._chunks.append(array[np.isfinite(array)])
            self._sorted = None

    def add_text(self, text: str) -> None:
        """Allow the numbers written in ``text`` (a question, an assumption, a name)."""
        self.add_values(_numbers_in_text(text))

    def add_result(self, result: QueryResult) -> None:
        """Allow everything the executed result vouches for."""
        frame = result.dataframe
        self.add_values([result.row_count, len(frame.columns)])
        for name in frame.columns:
            self.add_text(str(name))
            self._add_column(frame[name])

    def _add_column(self, column: pd.Series) -> None:
        if pd.api.types.is_bool_dtype(column):
            return
        if pd.api.types.is_numeric_dtype(column):
            self.add_values(pd.to_numeric(column, errors="coerce").dropna().to_numpy(dtype=float))
            return
        if pd.api.types.is_datetime64_any_dtype(column):
            stamps = column.dropna().dt
            for part in (stamps.year, stamps.month, stamps.day):
                self.add_values(part.unique())
            return
        present = column.dropna()
        try:
            cells = present.unique()
        except TypeError:  # unhashable cells, such as lists from a LIST column
            cells = present.to_numpy()
        for value in cells:
            self._add_cell(value)

    def _add_cell(self, value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, str):
            self.add_text(value)
        elif isinstance(value, datetime | date):
            self.add_values(_date_parts(value))
        else:
            try:
                self.add_values([float(value)])  # type: ignore[arg-type]
            except (TypeError, ValueError, InvalidOperation):
                self.add_text(str(value))

    def matches(self, token: NumberToken) -> bool:
        """True if some allowed number is what ``token`` could stand for, after rounding."""
        if not self._chunks:
            return False
        if self._sorted is None:
            self._sorted = np.sort(np.concatenate(self._chunks))
        for target, half in token.candidates:
            tolerance = half + _EPSILON * max(1.0, abs(target))
            low = np.searchsorted(self._sorted, target - tolerance, side="left")
            if low < self._sorted.size and self._sorted[low] <= target + tolerance:
                return True
        return False


@dataclass(frozen=True)
class GroundingIssue:
    """One thing the check did not accept."""

    kind: Literal["number", "key_number"]
    text: str  # the number as written, or the key number's label
    reason: str


@dataclass(frozen=True)
class GroundingReport:
    """The verdict on one explanation."""

    issues: tuple[GroundingIssue, ...]

    @property
    def ok(self) -> bool:
        """True when every number is grounded."""
        return not self.issues

    def feedback(self) -> str:
        """A sentence for the model on regeneration. It names numbers, never data values."""
        numbers = [i.text for i in self.issues if i.kind == "number"]
        keys = [i.text for i in self.issues if i.kind == "key_number"]
        parts = []
        if numbers:
            parts.append("These numbers are not in the result: " + ", ".join(numbers) + ".")
        if keys:
            parts.append("These key numbers do not match their column: " + ", ".join(keys) + ".")
        return " ".join(parts)


def _written_decimals(value: float) -> int:
    try:
        exponent = Decimal(repr(float(value))).as_tuple().exponent
    except InvalidOperation:  # pragma: no cover - repr of a float is always a valid Decimal
        return 0
    return max(-exponent, 0) if isinstance(exponent, int) else 0


def _column_values(result: QueryResult, name: str) -> np.ndarray | None:
    frame = result.dataframe
    matches = [c for c in frame.columns if str(c) == name] or [
        c for c in frame.columns if str(c).casefold() == name.casefold()
    ]
    if not matches:
        return None
    column = frame[matches[0]]
    if pd.api.types.is_bool_dtype(column):
        return None
    numbers = pd.to_numeric(column, errors="coerce").dropna().to_numpy(dtype=float)
    return np.abs(numbers)


def _key_number_ok(key: KeyNumber, result: QueryResult) -> tuple[bool, str]:
    values = _column_values(result, key.column)
    if values is None:
        return False, f"no numeric column {key.column!r} in the result"
    target = abs(key.value)
    half = 0.5 * 10 ** -_written_decimals(key.value)
    for goal, tolerance in ((target, half), (target / 100, half / 100)):
        slack = tolerance + _EPSILON * max(1.0, abs(goal))
        if values.size and bool(np.any(np.abs(values - goal) <= slack)):
            return True, ""
    return False, f"{key.value:g} is not a value of {key.column!r}"


def check_grounding(
    explanation: ModelExplanation,
    result: QueryResult,
    *,
    question: str,
    assumptions: Sequence[str] = (),
    extra_numbers: Iterable[float] = (),
) -> GroundingReport:
    """Check every number of ``explanation`` against ``result`` and what the user said.

    Args:
        explanation: The model's answer, caveats and key numbers.
        result: The executed result the explanation is about.
        question: The user's question; its numbers are allowed.
        assumptions: The assumptions shown with the answer; their numbers are allowed.
        extra_numbers: Further values the model was shown by the pipeline, such as the
            summary statistics of a long result, and limits such as the row cap.

    Returns:
        A report; ``report.ok`` is true when nothing was made up.
    """
    allowed = AllowedNumbers()
    allowed.add_result(result)
    allowed.add_values(extra_numbers)
    for text in (question, *assumptions):
        allowed.add_text(text)

    issues: list[GroundingIssue] = []
    seen: set[str] = set()
    written = [
        explanation.answer_markdown,
        *explanation.caveats,
        *(k.label for k in explanation.key_numbers),
    ]
    for text in written:
        for token in extract_numbers(text):
            if token.text not in seen and not allowed.matches(token):
                seen.add(token.text)
                issues.append(GroundingIssue("number", token.text, "not in the result"))
    for key in explanation.key_numbers:
        good, reason = _key_number_ok(key, result)
        if not good:
            issues.append(GroundingIssue("key_number", key.label, reason))
    return GroundingReport(tuple(issues))
