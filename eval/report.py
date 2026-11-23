"""Turn a finished run into numbers and a report (eval/README.md sections 4 and 5).

``compute_metrics`` is pure arithmetic over the case results. ``threshold_rows`` lays the
numbers against ADR 0013 and says pass, fail or "not measured"; it never changes a threshold
(that takes a new ADR). ``render_markdown`` and ``render_csv`` produce the two files a run
writes, and ``write_reports`` puts them in ``eval/reports/``.
"""

from __future__ import annotations

import csv
import io
import random
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from eval.cases import CATEGORIES, CATEGORY_TITLES, Case
from eval.runner import BenchmarkRun, CaseResult
from eval.scoring import percentile, ratio

REVIEW_SAMPLE = 30
SIMPLE_CATEGORIES = ("simple_aggregation", "filters")
LATENCY_LIMIT_S = 15.0
COST_TARGET_USD = 0.02


# -- numbers ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """The measurements of one run. ``None`` means there was nothing to measure."""

    cases_run: int
    cases_not_run: int
    invalid_cases: int
    gold_blocked: int
    execution_success: float | None
    accuracy: float | None
    accuracy_simple_and_filters: float | None
    clarification_recall: float | None
    false_ask_rate: float | None
    multi_turn_correct: float | None
    answers_with_ungrounded_number: int
    fallback_rate: float | None
    schema_precision: float | None
    schema_recall: float | None
    assumptions_found: float | None
    unsafe_cases: int
    unsafe_attempts: int
    latency_p50_s: float | None
    latency_p95_s: float | None
    cost_median_usd: float | None
    cost_p95_usd: float | None
    spent_usd: float
    by_category: dict[str, tuple[int, float | None, float | None]]  # n, ran, right


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def compute_metrics(run: BenchmarkRun) -> Metrics:
    """Every measurement the report needs, from the results of ``run``."""
    results = [r for r in run.results if not r.invalid]
    answerable = [r for r in results if r.expected_status == "answered"]
    asked_back = [r for r in results if r.expected_status == "clarification"]
    simple = [r for r in answerable if r.category in SIMPLE_CATEGORIES]
    follow_ups = [r for r in answerable if r.category == "follow_up"]
    answered = [r for r in results if r.status == "answered"]
    linked = [r for r in answered if r.precision is not None and r.recall is not None]
    expected_assumptions = sum(r.assumptions_expected for r in answered)
    by_category: dict[str, tuple[int, float | None, float | None]] = {}
    for category in CATEGORIES:
        group = [r for r in results if r.category == category]
        if group:
            asked = [r for r in group if r.expected_status == "answered"]
            by_category[category] = (
                len(group),
                ratio(sum(r.executed for r in asked), len(asked)),
                ratio(sum(r.correct for r in group), len(group)),
            )
    costs = [r.cost_usd for r in results if r.cost_usd > 0]
    return Metrics(
        cases_run=len(run.results),
        cases_not_run=len(run.not_run),
        invalid_cases=sum(r.invalid for r in run.results),
        gold_blocked=sum(r.gold_blocked for r in run.results),
        execution_success=ratio(sum(r.executed for r in answerable), len(answerable)),
        accuracy=ratio(sum(r.correct for r in answerable), len(answerable)),
        accuracy_simple_and_filters=ratio(sum(r.correct for r in simple), len(simple)),
        clarification_recall=ratio(
            sum(r.status == "clarification" for r in asked_back), len(asked_back)
        ),
        false_ask_rate=ratio(sum(r.status == "clarification" for r in answerable), len(answerable)),
        multi_turn_correct=ratio(sum(r.correct for r in follow_ups), len(follow_ups)),
        answers_with_ungrounded_number=sum(r.ungrounded > 0 for r in answered),
        fallback_rate=ratio(sum(r.fallback_used for r in answered), len(answered)),
        schema_precision=_mean([r.precision for r in linked if r.precision is not None]),
        schema_recall=_mean([r.recall for r in linked if r.recall is not None]),
        assumptions_found=ratio(sum(r.assumptions_found for r in answered), expected_assumptions),
        unsafe_cases=sum(not r.safe for r in results),
        unsafe_attempts=sum(r.attempted_unsafe for r in results),
        latency_p50_s=percentile([r.latency_s for r in results], 0.5),
        latency_p95_s=percentile([r.latency_s for r in results], 0.95),
        cost_median_usd=percentile(costs, 0.5) if costs else None,
        cost_p95_usd=percentile(costs, 0.95) if costs else None,
        spent_usd=sum(r.spent_usd for r in run.results),
        by_category=by_category,
    )


# -- ADR 0013 --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdRow:
    """One line of ADR 0013's table laid against the run."""

    metric: str
    threshold: str
    measured: str
    verdict: str  # pass | fail | not measured | measured elsewhere


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _at_least(metric: str, value: float | None, floor: float) -> ThresholdRow:
    verdict = "not measured" if value is None else ("pass" if value >= floor else "fail")
    return ThresholdRow(metric, f"{floor:.0%} or more", _percent(value), verdict)


def _at_most(metric: str, value: float | None, ceiling: float) -> ThresholdRow:
    verdict = "not measured" if value is None else ("pass" if value <= ceiling else "fail")
    return ThresholdRow(metric, f"{ceiling:.0%} or less", _percent(value), verdict)


def threshold_rows(metrics: Metrics) -> list[ThresholdRow]:
    """ADR 0013, line by line. Lines the benchmark does not measure say where they are."""
    seconds = metrics.latency_p95_s
    cost = metrics.cost_median_usd
    return [
        ThresholdRow(
            "Adversarial SQL blocked",
            "100%",
            "the validator's own test corpus (tests/unit/sql_cases.yaml)",
            "measured elsewhere",
        ),
        ThresholdRow(
            "Gold SQL wrongly blocked by the validator",
            "0",
            str(metrics.gold_blocked),
            "pass" if metrics.gold_blocked == 0 else "fail",
        ),
        _at_least("Execution success, non-ambiguous cases", metrics.execution_success, 0.95),
        _at_least("Result accuracy, all non-ambiguous cases", metrics.accuracy, 0.85),
        _at_least(
            "Result accuracy, simple aggregation and filter cases",
            metrics.accuracy_simple_and_filters,
            0.95,
        ),
        _at_least("Clarification recall on ambiguous cases", metrics.clarification_recall, 0.80),
        _at_most("Clarification requested on clear cases", metrics.false_ask_rate, 0.10),
        _at_least("Multi-turn sequences fully correct", metrics.multi_turn_correct, 0.80),
        ThresholdRow(
            "Answers shown with an ungrounded number",
            "0",
            str(metrics.answers_with_ungrounded_number),
            "pass" if metrics.answers_with_ungrounded_number == 0 else "fail",
        ),
        ThresholdRow(
            "Profile values matching pandas",
            "100%",
            "the profiler's tests (tests/unit/test_profiler.py)",
            "measured elsewhere",
        ),
        ThresholdRow(
            "Question latency, p95",
            f"{LATENCY_LIMIT_S:g} s or less",
            "n/a" if seconds is None else f"{seconds:.1f} s",
            "not measured"
            if seconds is None
            else ("pass" if seconds <= LATENCY_LIMIT_S else "fail"),
        ),
        ThresholdRow(
            "Median cost per question",
            f"under US${COST_TARGET_USD:.2f} (target; the ADR fixes it after spike T0.4 (c))",
            "n/a (no model cost recorded)" if cost is None else f"US${cost:.4f}",
            "not measured" if cost is None else ("pass" if cost < COST_TARGET_USD else "fail"),
        ),
    ]


# -- files -----------------------------------------------------------------------------------------


def git_revision() -> str:
    """The short commit id, with ``-dirty`` if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return f"{sha}-dirty" if dirty else sha


def _one_line(text: str | None, limit: int = 200) -> str:
    flat = " ".join((text or "").split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def review_sample(
    run: BenchmarkRun, cases: Sequence[Case], count: int = REVIEW_SAMPLE
) -> list[tuple[CaseResult, Case]]:
    """Up to ``count`` answered cases for a person to read against ``answer_facts``.
    The pick is a function of the seed and the case ids, so two reports of the same cases
    ask for the same reading."""
    by_id = {c.id: c for c in cases}
    pool = sorted(r.id for r in run.results if r.status == "answered" and r.id in by_id)
    chosen = sorted(random.Random(f"review:{run.seed}").sample(pool, min(count, len(pool))))  # noqa: S311
    results = {r.id: r for r in run.results}
    return [(results[i], by_id[i]) for i in chosen]


def render_markdown(
    run: BenchmarkRun,
    cases: Sequence[Case],
    *,
    revision: str,
    split: str,
    milestone: str | None = None,
    oracle: bool = False,
) -> str:
    """The report as Markdown."""
    metrics = compute_metrics(run)
    out: list[str] = ["# Benchmark report", ""]
    if oracle:
        out += [
            '> **ORACLE RUN.** The "model" here is a script that returns the gold SQL. These',
            "> figures prove that the harness, the scoring and the validator work together;",
            "> they say nothing about how well a language model answers.",
            "",
        ]
    out += [
        f"- Date: {run.started:%Y-%m-%d %H:%M} UTC",
        f"- Commit: {revision}",
        f"- Split: {split}" + (f" (milestone {milestone})" if milestone else ""),
        f"- Cases: {metrics.cases_run} run, {metrics.cases_not_run} not run, "
        f"{metrics.invalid_cases} with a gold query that did not run",
        f"- Model: {run.model}",
        f"- Privacy mode: {'schema only' if run.schema_only else 'full'}",
        f"- Dataset seed: {run.seed}",
        f"- Run time: {run.seconds:.1f} s; spend: US${metrics.spent_usd:.4f}",
    ]
    if run.stopped:
        out.append(f"- **Stopped early:** {run.stopped}")
    out += ["", "## ADR 0013 thresholds", "", "| Metric | Threshold | Measured | Result |"]
    out.append("| --- | --- | --- | --- |")
    out += [
        f"| {row.metric} | {row.threshold} | {row.measured} | {row.verdict} |"
        for row in threshold_rows(metrics)
    ]
    out += [
        "",
        "Thresholds apply to the full benchmark at a milestone, held-out cases included. "
        "A partial run shows where it stands.",
        "",
        "## By category",
        "",
        "| Category | Cases | Query ran (where an answer is expected) | Right response |",
        "| --- | --- | --- | --- |",
    ]
    out += [
        f"| {CATEGORY_TITLES[name]} | {n} | {_percent(ran)} | {_percent(right)} |"
        for name, (n, ran, right) in metrics.by_category.items()
    ]
    out += ["", "## Other measurements", ""]
    out += [
        f"- Schema linking (answered cases): precision {_percent(metrics.schema_precision)}, "
        f"recall {_percent(metrics.schema_recall)}",
        "- Explanation fell back to the plain summary: "
        f"{_percent(metrics.fallback_rate)} of answers",
        f"- Expected assumptions found in the response (keyword match, indicative only): "
        f"{_percent(metrics.assumptions_found)}",
        f"- Cases where something unsafe ran or the data was changed: {metrics.unsafe_cases}",
        f"- Cases where the model wrote SQL that is not one query (stopped by the validator): "
        f"{metrics.unsafe_attempts}",
        "- Latency (per question, model calls included): p50 "
        f"{_seconds(metrics.latency_p50_s)}, p95 {_seconds(metrics.latency_p95_s)}",
        f"- Cost per question: median {_dollars(metrics.cost_median_usd)}, "
        f"p95 {_dollars(metrics.cost_p95_usd)}",
    ]
    wrong = [r for r in run.results if not r.correct or not r.safe]
    out += ["", "## Cases that were not right", ""]
    if wrong:
        out += [
            "| Case | Category | Question | What went wrong | SQL that ran |",
            "| --- | --- | --- | --- | --- |",
        ]
        out += [
            f"| {r.id} | {CATEGORY_TITLES[r.category]} | {_one_line(r.question, 120)} | "
            f"{_one_line(r.reason or ('unsafe' if not r.safe else ''))} | {_one_line(r.sql, 160)} |"
            for r in wrong
        ]
    else:
        out.append("None.")
    if run.not_run:
        out += ["", "## Not run", "", ", ".join(run.not_run)]
    sample = review_sample(run, cases)
    out += [
        "",
        f"## Manual review ({len(sample)} answered cases)",
        "",
        "Read each answer against the facts it should state, and tick it. A person does this, "
        "not the harness (ADR 0017).",
        "",
    ]
    for result, case in sample:
        facts = (
            "; ".join(case.answer_facts) or "no specific facts listed; check it matches the table"
        )
        out += [
            f"- [ ] **{result.id}** {_one_line(result.question, 140)}",
            f"  - Answer: {_one_line(result.answer, 500)}",
            f"  - Should state: {facts}",
        ]
        if case.expected_assumptions:
            shown = "; ".join(result.assumptions) or "none shown"
            out.append(
                f"  - Assumptions expected: {'; '.join(case.expected_assumptions)}. Shown: {shown}"
            )
    unrun = [r for r in run.results if r.invalid]
    if unrun:
        out += ["", "## Gold queries that did not run", ""]
        out += [f"- {r.id}: {_one_line(r.reason)}" for r in unrun]
    return "\n".join(out) + "\n"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def _dollars(value: float | None) -> str:
    return "n/a" if value is None else f"US${value:.4f}"


def render_csv(run: BenchmarkRun) -> str:
    """One row per case, every field of its result."""
    names = [f.name for f in fields(CaseResult)]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(names)
    for result in run.results:
        row = []
        for name in names:
            value = getattr(result, name)
            row.append(" | ".join(value) if isinstance(value, tuple) else value)
        writer.writerow(row)
    return buffer.getvalue()


def report_stem(run: BenchmarkRun, revision: str, split: str, *, oracle: bool = False) -> str:
    """``<date>-<git sha>-<split>``, with ``-oracle`` for a run that is not a model's."""
    return f"{run.started:%Y-%m-%d}-{revision}-{split}" + ("-oracle" if oracle else "")


def write_reports(
    run: BenchmarkRun,
    cases: Sequence[Case],
    directory: Path,
    *,
    split: str,
    milestone: str | None = None,
    oracle: bool = False,
    revision: str | None = None,
) -> tuple[Path, Path]:
    """Write ``<stem>.md`` and ``<stem>.csv`` into ``directory``; returns both paths."""
    revision = revision or git_revision()
    directory.mkdir(parents=True, exist_ok=True)
    stem = report_stem(run, revision, split, oracle=oracle)
    markdown = directory / f"{stem}.md"
    table = directory / f"{stem}.csv"
    markdown.write_text(
        render_markdown(
            run, cases, revision=revision, split=split, milestone=milestone, oracle=oracle
        ),
        encoding="utf-8",
    )
    table.write_text(render_csv(run), encoding="utf-8")
    return markdown, table
