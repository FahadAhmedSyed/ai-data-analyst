"""Benchmark cases: the YAML format of eval/README.md section 2, loaded and checked.

A case file that is wrong is refused whole, with every problem listed, so a typo cannot
quietly remove a case from a run. ``expected_result`` is not stored: the runner computes it
by running ``gold_sql``, so the cases stay right if the generator changes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from eval.generate_datasets import DATASETS

Compare = Literal["scalar", "ordered", "unordered"]
ExpectedStatus = Literal["answered", "clarification", "failed"]
Split = Literal["dev", "holdout", "all"]

# Category -> the letter its case ids start with.
CATEGORIES: dict[str, str] = {
    "simple_aggregation": "S",
    "filters": "F",
    "grouping_sorting": "G",
    "dates_seasons": "D",
    "nulls_duplicates": "N",
    "multi_aggregation": "M",
    "window_ranking": "W",
    "ambiguous": "A",
    "follow_up": "T",
    "empty_result": "E",
    "injection": "I",
}
# The names the report shows.
CATEGORY_TITLES: dict[str, str] = {
    "simple_aggregation": "Simple aggregation",
    "filters": "Filters",
    "grouping_sorting": "Grouping and sorting",
    "dates_seasons": "Dates and seasons",
    "nulls_duplicates": "NULLs and duplicates",
    "multi_aggregation": "Multiple aggregations",
    "window_ranking": "Window and ranking",
    "ambiguous": "Ambiguous (clarification expected)",
    "follow_up": "Multi-turn follow-ups",
    "empty_result": "Empty results",
    "injection": "Prompt injection and refusal",
}
_FIELDS = frozenset(
    {
        "id", "dataset", "category", "turns", "question", "gold_sql", "alt_gold_sql", "compare",
        "key_columns", "clarify", "clarify_about", "expected_status", "expected_assumptions",
        "answer_facts", "notes", "holdout",
    }
)  # fmt: skip
MAX_TURNS = 6


class CaseFileError(ValueError):
    """A case file that cannot be used. The message lists every problem found."""


@dataclass(frozen=True)
class Case:
    """One question to ask, and what a right answer looks like."""

    id: str
    dataset: str
    category: str
    question: str
    turns: tuple[str, ...] = ()
    gold_sql: str | None = None
    alt_gold_sql: str | None = None
    compare: Compare = "unordered"
    key_columns: tuple[str, ...] = ()
    clarify: bool = False
    clarify_about: str | None = None
    expected_status: ExpectedStatus = "answered"
    expected_assumptions: tuple[str, ...] = ()
    answer_facts: tuple[str, ...] = ()
    notes: str | None = None
    holdout: bool = False

    @property
    def golds(self) -> tuple[str, ...]:
        """The accepted queries: the gold one, then the alternative reading if there is one."""
        return tuple(sql for sql in (self.gold_sql, self.alt_gold_sql) if sql)

    @property
    def expects_answer(self) -> bool:
        """True when a right response is an answer (not a question back, not a refusal)."""
        return self.expected_status == "answered"


def _texts(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value or ())


def _build(raw: dict[str, Any]) -> Case:
    """A ``Case`` from one YAML mapping; field problems are raised as ``ValueError``."""
    clarify = bool(raw.get("clarify", False))
    return Case(
        id=str(raw["id"]),
        dataset=str(raw["dataset"]),
        category=str(raw["category"]),
        question=str(raw["question"]).strip(),
        turns=_texts(raw.get("turns")),
        gold_sql=_sql(raw.get("gold_sql")),
        alt_gold_sql=_sql(raw.get("alt_gold_sql")),
        compare=raw.get("compare", "unordered"),
        key_columns=_texts(raw.get("key_columns")),
        clarify=clarify,
        clarify_about=raw.get("clarify_about"),
        expected_status=raw.get("expected_status", "clarification" if clarify else "answered"),
        expected_assumptions=_texts(raw.get("expected_assumptions")),
        answer_facts=_texts(raw.get("answer_facts")),
        notes=raw.get("notes"),
        holdout=bool(raw.get("holdout", False)),
    )


def _sql(value: Any) -> str | None:
    text = " ".join(str(value).split()) if value else ""
    return text or None


def problems_with(case: Case) -> list[str]:
    """What is wrong with one case on its own; empty when it is usable."""
    found: list[str] = []
    if case.dataset not in DATASETS:
        found.append(f"unknown dataset {case.dataset!r}")
    letter = CATEGORIES.get(case.category)
    if letter is None:
        found.append(f"unknown category {case.category!r}")
    elif not case.id.startswith(letter) or not case.id[1:].isdigit():
        found.append(f"id should be {letter} followed by a number for {case.category}")
    if not case.question:
        found.append("no question")
    if case.compare not in get_args(Compare):
        found.append(f"compare must be one of {', '.join(get_args(Compare))}")
    if case.expected_status not in get_args(ExpectedStatus):
        found.append(f"expected_status must be one of {', '.join(get_args(ExpectedStatus))}")
    if case.clarify != (case.expected_status == "clarification"):
        found.append("clarify and expected_status disagree")
    if case.expects_answer and not case.gold_sql:
        found.append("a case that expects an answer needs gold_sql")
    if not case.expects_answer and (case.gold_sql or case.alt_gold_sql):
        found.append("a case that expects a question back or a refusal has no gold_sql")
    if case.alt_gold_sql and not case.gold_sql:
        found.append("alt_gold_sql without gold_sql")
    if case.category == "follow_up" and not case.turns:
        found.append("a follow-up case needs earlier turns")
    if len(case.turns) > MAX_TURNS:
        found.append(f"at most {MAX_TURNS} earlier turns")
    if case.clarify and not case.clarify_about:
        found.append("a clarification case says what the question should be about")
    if case.expected_assumptions and not case.expects_answer:
        found.append("expected_assumptions only make sense when an answer is expected")
    return found


def check_cases(cases: Sequence[Case]) -> list[str]:
    """Every problem in a set of cases, each named by case id."""
    found = [f"{case.id}: {text}" for case in cases for text in problems_with(case)]
    found += [
        f"{id_}: id used {n} times" for id_, n in Counter(c.id for c in cases).items() if n > 1
    ]
    return found


def parse_cases(text: str, *, source: str = "cases") -> list[Case]:
    """Cases from the text of a YAML file with a top-level ``cases`` list.

    Raises:
        CaseFileError: The file is not valid YAML, a case has a missing or unknown field,
            or any case fails ``problems_with``. Every problem is listed.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        msg = f"{source}: not valid YAML ({type(error).__name__})"
        raise CaseFileError(msg) from None
    items = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(items, list):
        msg = f"{source}: expected a top-level 'cases' list"
        raise CaseFileError(msg)
    cases: list[Case] = []
    found: list[str] = []
    for position, raw in enumerate(items, 1):
        name = f"{source} item {position}"
        if not isinstance(raw, dict):
            found.append(f"{name}: not a mapping")
            continue
        name = str(raw.get("id", name))
        unknown = sorted(set(raw) - _FIELDS)
        missing = [f for f in ("id", "dataset", "category", "question") if not raw.get(f)]
        if unknown or missing:
            found.append(f"{name}: unknown fields {unknown}, missing fields {missing}")
            continue
        cases.append(_build(raw))
    found += check_cases(cases)
    if found:
        raise CaseFileError(f"{source}: " + "; ".join(found))
    return cases


def load_cases(path: Path | str) -> list[Case]:
    """Cases from one YAML file, or from every ``*.yaml`` file in a folder (by file name).

    Ids must be unique across all the files.
    """
    path = Path(path)
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    if not files:
        msg = f"{path}: no .yaml case files"
        raise CaseFileError(msg)
    cases: list[Case] = []
    for file in files:
        cases += parse_cases(file.read_text(encoding="utf-8"), source=file.name)
    found = check_cases(cases)
    if found:
        raise CaseFileError(f"{path}: " + "; ".join(found))
    return cases


def select(
    cases: Iterable[Case], split: Split = "dev", ids: Iterable[str] | None = None
) -> list[Case]:
    """The cases of a split (``dev`` is every case that is not held out), optionally only
    the named ids. An id that is not in the split is an error, so a typo is not a skipped case."""
    chosen = [
        c for c in cases if split == "all" or (c.holdout if split == "holdout" else not c.holdout)
    ]
    if ids:
        wanted = list(dict.fromkeys(ids))
        known = {c.id for c in chosen}
        unknown = [i for i in wanted if i not in known]
        if unknown:
            msg = f"not in the {split} split: {', '.join(unknown)}"
            raise CaseFileError(msg)
        chosen = [c for c in chosen if c.id in set(wanted)]
    return chosen
