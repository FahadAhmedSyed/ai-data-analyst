"""Run benchmark cases through the real pipeline and judge each response.

Each dataset is generated and loaded once, through the application's own ingestion. Each case
then gets a new ``Analyst`` (so a new, empty conversation), is asked its earlier turns and
then its question, and is scored. The response is the real one: the validator, executor,
explainer and recommender all run; only the model client is supplied by the caller (the
OpenAI client for a real run, the oracle or a scripted fake for tests).

A run never changes a case. A gold query that the validator refuses or that fails to run
makes the case ``invalid`` and the model is not asked (so no money is spent on it).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from analyst.config.settings import Settings, get_settings
from analyst.core.errors import SessionCorruptedError
from analyst.core.logging import get_logger
from analyst.core.models import AnalystResponse, ModelExplanation
from analyst.database.executor import QueryExecutor
from analyst.database.session_db import SessionDatabase
from analyst.llm.client import LLMClient
from analyst.pipeline.analyst import Analyst
from analyst.pipeline.grounding import check_grounding
from analyst.pipeline.result_digest import digest_result
from analyst.session.dataset import LoadedDataset, load_dataset
from analyst.sql.validator import SqlValidator
from eval.cases import Case
from eval.generate_datasets import DATASETS, SEED, generate
from eval.gold import GoldError, GoldRunner
from eval.scoring import (
    Expected,
    assumption_found,
    compare_results,
    is_read_only,
    linking_scores,
)

log = get_logger(__name__)

ClientFactory = Callable[[Case, str], LLMClient]
"""Makes the model client for one case, given the name of the table its data is in."""

ANSWER_CHARS = 1_500  # how much of an answer a result keeps, for the report


# -- the datasets ----------------------------------------------------------------------------------


@dataclass
class PreparedDataset:
    """One dataset, loaded and ready for questions."""

    name: str
    db: SessionDatabase
    loaded: LoadedDataset
    gold: GoldRunner

    @property
    def table(self) -> str:
        """The table the data is in."""
        return self.db.table_name

    @property
    def columns(self) -> frozenset[str]:
        """The table's column names."""
        return self.loaded.profile.column_names


class DatasetPool:
    """Loads each benchmark dataset on first use and keeps it for the run."""

    def __init__(self, settings: Settings | None = None, seed: int = SEED) -> None:
        """Create an empty pool; nothing is generated until ``get``."""
        self._settings = settings or get_settings()
        self._seed = seed
        self._loaded: dict[str, PreparedDataset] = {}

    def get(self, name: str) -> PreparedDataset:
        """The dataset ``name``, generated and loaded the first time it is asked for."""
        if name not in self._loaded:
            self._loaded[name] = self._load(name)
        return self._loaded[name]

    def reload(self, name: str) -> PreparedDataset:
        """Load ``name`` again from scratch, for when a case left its database damaged."""
        previous = self._loaded.pop(name, None)
        if previous is not None:
            previous.db.close()
        return self.get(name)

    def close(self) -> None:
        """Close every database."""
        for dataset in self._loaded.values():
            dataset.db.close()
        self._loaded.clear()

    def _load(self, name: str) -> PreparedDataset:
        spec = DATASETS[name]
        db = SessionDatabase(self._settings)
        loaded = load_dataset(
            db, generate(name, self._seed), spec.load_name, spec.options, settings=self._settings
        )
        validator = SqlValidator.for_database(db, max_rows=self._settings.max_result_rows)
        return PreparedDataset(
            name, db, loaded, GoldRunner(validator, QueryExecutor(db, self._settings))
        )

    def __enter__(self) -> DatasetPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# -- results ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """How one case went. Everything the report shows comes from here."""

    id: str
    dataset: str
    category: str
    holdout: bool
    question: str
    expected_status: str
    status: str  # answered | clarification | failed | invalid
    correct: bool
    executed: bool = False  # the final query ran and gave a result
    matched: str | None = None  # "gold" or "alt" when the result matched
    reason: str = ""
    precision: float | None = None
    recall: float | None = None
    assumptions_expected: int = 0
    assumptions_found: int = 0
    fallback_used: bool = False
    ungrounded: int = 0
    safe: bool = True
    attempted_unsafe: bool = False  # the model wrote SQL that is not one query (it was stopped)
    gold_blocked: bool = False
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # of the question itself
    latency_s: float = 0.0  # of the question itself
    spent_usd: float = 0.0  # of the whole case, earlier turns included
    rows: int | None = None
    sql: str | None = None
    answer: str = ""
    assumptions: tuple[str, ...] = ()
    turn_statuses: tuple[str, ...] = ()

    @property
    def invalid(self) -> bool:
        """True when the case could not be run because its gold query did not."""
        return self.status == "invalid"


@dataclass
class BenchmarkRun:
    """A finished (or cost-capped) run."""

    results: list[CaseResult]
    not_run: list[str]
    started: datetime
    seconds: float
    model: str
    seed: int
    schema_only: bool
    stopped: str | None = None
    notes: list[str] = field(default_factory=list)


# -- judging one case ------------------------------------------------------------------------------


def ungrounded_numbers(response: AnalystResponse) -> int:
    """Numbers in a model-written answer that are not in the result, the question or the
    assumptions: the pipeline's own check, run again on what was actually shown."""
    result = response.result
    if response.status != "answered" or result is None or result.is_empty or response.fallback_used:
        return 0
    allowed = [a for a in (response.resolved_question, *response.assumptions) if a]
    report = check_grounding(
        ModelExplanation(answer_markdown=response.message, key_numbers=[], caveats=[]),
        result,
        question=response.question,
        assumptions=allowed,
        extra_numbers=digest_result(result).numbers,
    )
    return len(report.issues)


def _judge(
    case: Case, response: AnalystResponse, expected: Sequence[Expected]
) -> tuple[bool, str | None, str]:
    """Whether the response is right for the case: (correct, which gold matched, why not)."""
    if case.expected_status == "clarification":
        ok = response.status == "clarification"
        return (
            ok,
            None,
            "" if ok else f"answered {response.status}, a clarifying question was right",
        )
    if case.expected_status == "failed":
        ok = response.status == "failed" and response.result is None
        return ok, None, "" if ok else f"ended {response.status}, a refusal was right"
    if response.status != "answered" or response.result is None:
        return False, None, f"ended {response.status}, an answer was expected"
    first_reason = ""
    for position, gold in enumerate(expected):
        verdict = compare_results(response.result.dataframe, gold, case.compare)
        if verdict.matched:
            return True, "gold" if position == 0 else "alt", ""
        first_reason = first_reason or verdict.reason
    return False, None, first_reason


def run_case(
    case: Case,
    data: PreparedDataset,
    client: LLMClient,
    *,
    settings: Settings,
    schema_only: bool = False,
) -> tuple[CaseResult, bool]:
    """Ask one case and judge the response.

    Returns:
        The result, and whether the dataset's database was left damaged (the caller loads
        it again before the next case).
    """
    base = {
        "id": case.id,
        "dataset": case.dataset,
        "category": case.category,
        "holdout": case.holdout,
        "question": case.question,
        "expected_status": case.expected_status,
    }
    expected: list[Expected] = []
    try:
        expected = [data.gold.expected(sql) for sql in case.golds]
    except GoldError as error:
        result = CaseResult(
            **base, status="invalid", correct=False, reason=str(error), gold_blocked=error.blocked
        )
        return result, False

    analyst = Analyst(
        data.db, data.loaded.profile, client, settings=settings, schema_only=schema_only
    )
    statuses: list[str] = []
    spent = 0.0
    for turn in case.turns:
        earlier = analyst.ask(turn)
        statuses.append(earlier.status)
        spent += earlier.usage.cost_usd
    response = analyst.ask(case.question)
    spent += response.usage.cost_usd

    correct, matched, reason = _judge(case, response, expected)
    precision = recall = None
    if response.status == "answered" and response.sql and expected:
        best = expected[0] if matched in (None, "gold") else expected[1]
        precision, recall = linking_scores(response.sql, best.sql, data.columns)
    found = sum(assumption_found(a, response.assumptions) for a in case.expected_assumptions)

    damaged = False
    try:
        data.db.verify_catalog()
    except SessionCorruptedError:
        damaged = True
    ran_sql = response.sql if response.result is not None else None
    safe = not damaged and is_read_only(ran_sql)

    usage = response.usage
    result = CaseResult(
        **base,
        status=response.status,
        correct=correct,
        executed=response.status == "answered" and response.result is not None,
        matched=matched,
        reason=reason,
        precision=precision,
        recall=recall,
        assumptions_expected=len(case.expected_assumptions),
        assumptions_found=found,
        fallback_used=response.fallback_used,
        ungrounded=ungrounded_numbers(response),
        safe=safe,
        attempted_unsafe=not is_read_only(response.sql),
        llm_calls=usage.llm_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        latency_s=usage.latency_ms / 1000,
        spent_usd=spent,
        rows=response.result.row_count if response.result is not None else None,
        sql=response.sql,
        answer=response.message[:ANSWER_CHARS],
        assumptions=tuple(response.assumptions),
        turn_statuses=tuple(statuses),
    )
    return result, damaged


# -- running many ----------------------------------------------------------------------------------


def run_benchmark(
    cases: Sequence[Case],
    client_factory: ClientFactory,
    *,
    model: str,
    settings: Settings | None = None,
    seed: int = SEED,
    schema_only: bool = False,
    max_cost_usd: float | None = None,
    on_result: Callable[[CaseResult, int, int], None] | None = None,
) -> BenchmarkRun:
    """Run ``cases`` one after another.

    Args:
        cases: The cases to run, in order.
        client_factory: Makes the model client for each case.
        model: How the report names the model.
        settings: Limits and region; the application settings by default.
        seed: The datasets' seed.
        schema_only: Run in the privacy mode that sends no values from the file.
        max_cost_usd: Stop before a case that could take the spend over this. A case's
            worst case is taken to be the costliest one so far, so the cap is a guide: the
            first case always runs, and one case can overshoot if it costs more than any
            before it.
        on_result: Called after each case with the result, its number and the total.
    """
    settings = settings or get_settings()
    started = datetime.now(UTC)
    clock = time.perf_counter()
    results: list[CaseResult] = []
    spent = 0.0
    costliest = 0.0
    stopped: str | None = None
    with DatasetPool(settings, seed) as pool:
        for number, case in enumerate(cases, 1):
            if max_cost_usd is not None and spent + costliest > max_cost_usd:
                stopped = f"stopped before {case.id}: spent ${spent:.4f} of ${max_cost_usd:.2f}"
                break
            data = pool.get(case.dataset)
            client = client_factory(case, data.table)
            result, damaged = run_case(
                case, data, client, settings=settings, schema_only=schema_only
            )
            if damaged:
                pool.reload(case.dataset)
            results.append(result)
            spent += result.spent_usd
            costliest = max(costliest, result.spent_usd)
            if on_result is not None:
                on_result(result, number, len(cases))
    ran = {r.id for r in results}
    return BenchmarkRun(
        results=results,
        not_run=[c.id for c in cases if c.id not in ran],
        started=started,
        seconds=time.perf_counter() - clock,
        model=model,
        seed=seed,
        schema_only=schema_only,
        stopped=stopped,
    )
