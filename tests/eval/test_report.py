"""Numbers, ADR 0013 verdicts and the two report files, from results made by hand."""

from __future__ import annotations

import csv
import io
import re
import subprocess
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval.cases import Case
from eval.report import (
    REVIEW_SAMPLE,
    Metrics,
    compute_metrics,
    git_revision,
    render_csv,
    render_markdown,
    report_stem,
    review_sample,
    threshold_rows,
    write_reports,
)
from eval.runner import BenchmarkRun, CaseResult
from eval.scoring import percentile

ROOT = Path(__file__).resolve().parents[2]


def res(id_="S01", category="simple_aggregation", **kw) -> CaseResult:
    fields_ = {
        "id": id_, "dataset": "sales", "category": category, "holdout": False,
        "question": f"Question {id_}?", "expected_status": "answered", "status": "answered",
        "correct": True, "executed": True,
    }  # fmt: skip
    return CaseResult(**{**fields_, **kw})


def run_of(results: list[CaseResult], **kw) -> BenchmarkRun:
    fields_ = {
        "results": results, "not_run": [], "started": datetime(2026, 11, 23, 9, 5, tzinfo=UTC),
        "seconds": 12.3, "model": "a model", "seed": 1, "schema_only": False,
    }  # fmt: skip
    return BenchmarkRun(**{**fields_, **kw})


def mixed() -> BenchmarkRun:
    clarified = {"status": "clarification", "executed": False}
    return run_of(
        [
            res("S01", latency_s=1.0, cost_usd=0.01, spent_usd=0.01, precision=1.0, recall=1.0,
                assumptions_expected=2, assumptions_found=1),
            res("S02", correct=False, latency_s=2.0, cost_usd=0.03, spent_usd=0.03,
                fallback_used=True, reason="expected 1.0, got 2.0"),
            res("F01", "filters", latency_s=3.0, cost_usd=0.02, spent_usd=0.02, ungrounded=2,
                precision=1.0, recall=0.5, assumptions_expected=1, assumptions_found=1),
            res("G01", "grouping_sorting", status="failed", correct=False, executed=False,
                latency_s=4.0),
            res("A01", "ambiguous", expected_status="clarification", executed=False,
                latency_s=0.5, **{"status": "clarification"}),
            res("A02", "ambiguous", expected_status="clarification", correct=False,
                latency_s=5.0, cost_usd=0.04, spent_usd=0.04),
            res("T01", "follow_up", latency_s=1.0),
            res("T02", "follow_up", correct=False, latency_s=1.0, **clarified),
            res("I01", "injection", expected_status="failed", status="failed", executed=False,
                latency_s=0.5),
            res("I02", "injection", expected_status="failed", correct=False, safe=False,
                attempted_unsafe=True, latency_s=0.5),
            res("X01", "nulls_duplicates", status="invalid", correct=False, executed=False,
                gold_blocked=True, reason="the validator refused it"),
        ]
    )  # fmt: skip


# -- numbers ------------------------------------------------------------------------------------


def test_the_numbers_of_a_mixed_run() -> None:
    m = compute_metrics(mixed())
    assert (m.cases_run, m.cases_not_run, m.invalid_cases, m.gold_blocked) == (11, 0, 1, 1)
    # Answerable: S01 S02 F01 G01 T01 T02 (6). The invalid case is left out of every ratio.
    assert m.execution_success == pytest.approx(4 / 6)
    assert m.accuracy == pytest.approx(3 / 6)
    assert m.accuracy_simple_and_filters == pytest.approx(2 / 3)
    assert m.clarification_recall == pytest.approx(1 / 2)
    assert m.false_ask_rate == pytest.approx(1 / 6)
    assert m.multi_turn_correct == pytest.approx(1 / 2)
    assert m.answers_with_ungrounded_number == 1
    assert m.fallback_rate == pytest.approx(1 / 6)  # of the six that were answered
    assert (m.schema_precision, m.schema_recall) == (1.0, 0.75)
    assert m.assumptions_found == pytest.approx(2 / 3)
    assert (m.unsafe_cases, m.unsafe_attempts) == (1, 1)
    assert m.cost_median_usd == pytest.approx(0.025)
    assert m.cost_p95_usd == pytest.approx(0.0385)
    assert m.spent_usd == pytest.approx(0.10)
    results = [r for r in mixed().results if not r.invalid]
    assert m.latency_p50_s == percentile([r.latency_s for r in results], 0.5)
    assert m.latency_p95_s == percentile([r.latency_s for r in results], 0.95)


def test_the_numbers_by_category_count_a_query_only_where_an_answer_was_expected() -> None:
    by = compute_metrics(mixed()).by_category
    assert by["simple_aggregation"] == (2, 1.0, 0.5)
    assert by["filters"] == (1, 1.0, 1.0)
    assert by["grouping_sorting"] == (1, 0.0, 0.0)
    assert by["ambiguous"] == (2, None, 0.5)
    assert by["follow_up"] == (2, 0.5, 0.5)
    assert by["injection"] == (2, None, 0.5)
    assert "nulls_duplicates" not in by, "the only case there is invalid"
    assert list(by) == [
        "simple_aggregation", "filters", "grouping_sorting", "ambiguous", "follow_up", "injection",
    ]  # fmt: skip


def test_nothing_to_measure_is_none_not_zero() -> None:
    m = compute_metrics(run_of([]))
    assert m.execution_success is None and m.accuracy is None
    assert m.clarification_recall is None and m.false_ask_rate is None
    assert m.latency_p95_s is None and m.cost_median_usd is None
    assert m.spent_usd == 0.0
    assert m.by_category == {}


def test_a_run_with_no_cost_recorded_has_no_cost_figures() -> None:
    m = compute_metrics(run_of([res(), res("S02")]))
    assert (m.cost_median_usd, m.cost_p95_usd) == (None, None)


def test_the_not_run_cases_are_counted() -> None:
    assert compute_metrics(run_of([res()], not_run=["S02", "S03"])).cases_not_run == 2


# -- ADR 0013 --------------------------------------------------------------------------------------


def perfect() -> Metrics:
    asked_back = res("A01", "ambiguous", expected_status="clarification", status="clarification")
    results = [
        res("S01", latency_s=2.0, cost_usd=0.005, spent_usd=0.005),
        res("T01", "follow_up"),
        asked_back,
    ]
    return compute_metrics(run_of(results))


def verdicts(m: Metrics) -> dict[str, str]:
    return {row.metric: row.verdict for row in threshold_rows(m)}


def test_a_perfect_run_passes_every_line_the_benchmark_measures() -> None:
    got = verdicts(perfect())
    assert len(got) == 12
    assert set(got.values()) == {"pass", "measured elsewhere"}
    assert [m for m, v in got.items() if v == "measured elsewhere"] == [
        "Adversarial SQL blocked",
        "Profile values matching pandas",
    ]


@pytest.mark.parametrize(
    ("change", "metric"),
    [
        ({"gold_blocked": 1}, "Gold SQL wrongly blocked by the validator"),
        ({"execution_success": 0.94}, "Execution success, non-ambiguous cases"),
        ({"accuracy": 0.84}, "Result accuracy, all non-ambiguous cases"),
        (
            {"accuracy_simple_and_filters": 0.94},
            "Result accuracy, simple aggregation and filter cases",
        ),
        ({"clarification_recall": 0.79}, "Clarification recall on ambiguous cases"),
        ({"false_ask_rate": 0.11}, "Clarification requested on clear cases"),
        ({"multi_turn_correct": 0.79}, "Multi-turn sequences fully correct"),
        ({"answers_with_ungrounded_number": 1}, "Answers shown with an ungrounded number"),
        ({"latency_p95_s": 15.1}, "Question latency, p95"),
        ({"cost_median_usd": 0.02}, "Median cost per question"),
    ],
)
def test_each_threshold_fails_just_past_its_limit(change: dict, metric: str) -> None:
    assert verdicts(replace(perfect(), **change))[metric] == "fail"


@pytest.mark.parametrize(
    ("change", "metric"),
    [
        ({"execution_success": 0.95}, "Execution success, non-ambiguous cases"),
        ({"accuracy": 0.85}, "Result accuracy, all non-ambiguous cases"),
        (
            {"accuracy_simple_and_filters": 0.95},
            "Result accuracy, simple aggregation and filter cases",
        ),
        ({"clarification_recall": 0.8}, "Clarification recall on ambiguous cases"),
        ({"false_ask_rate": 0.1}, "Clarification requested on clear cases"),
        ({"multi_turn_correct": 0.8}, "Multi-turn sequences fully correct"),
        ({"latency_p95_s": 15.0}, "Question latency, p95"),
        ({"cost_median_usd": 0.0199}, "Median cost per question"),
    ],
)
def test_each_threshold_passes_exactly_at_its_limit(change: dict, metric: str) -> None:
    assert verdicts(replace(perfect(), **change))[metric] == "pass"


def test_a_figure_with_nothing_behind_it_is_not_measured_rather_than_passed() -> None:
    blank = compute_metrics(run_of([]))
    got = verdicts(blank)
    not_measured = {m for m, v in got.items() if v == "not measured"}
    assert "Result accuracy, all non-ambiguous cases" in not_measured
    assert "Question latency, p95" in not_measured
    assert "Median cost per question" in not_measured
    assert got["Gold SQL wrongly blocked by the validator"] == "pass"


def test_the_threshold_text_is_the_adrs() -> None:
    rows = {r.metric: r.threshold for r in threshold_rows(perfect())}
    assert rows["Execution success, non-ambiguous cases"] == "95% or more"
    assert rows["Clarification requested on clear cases"] == "10% or less"
    assert rows["Question latency, p95"] == "15 s or less"
    assert rows["Gold SQL wrongly blocked by the validator"] == "0"
    assert rows["Median cost per question"].startswith("under US$0.02")


# -- the Markdown ---------------------------------------------------------------------------------


def case_for(id_: str, **kw) -> Case:
    return Case(id=id_, dataset="sales", category="simple_aggregation", question="Q?", **kw)


def markdown(run: BenchmarkRun, cases: list[Case] | None = None, **kw) -> str:
    return render_markdown(run, cases or [], revision="abc1234", split="dev", **kw)


def test_the_report_says_what_was_run_and_when() -> None:
    text = markdown(mixed(), milestone="M4")
    assert text.startswith("# Benchmark report")
    for line in (
        "- Date: 2026-11-23 09:05 UTC",
        "- Commit: abc1234",
        "- Split: dev (milestone M4)",
        "- Cases: 11 run, 0 not run, 1 with a gold query that did not run",
        "- Model: a model",
        "- Privacy mode: full",
        "- Dataset seed: 1",
        "- Run time: 12.3 s; spend: US$0.1000",
    ):
        assert line in text
    assert "ORACLE" not in text
    assert "Stopped early" not in text


def test_an_oracle_report_says_so_at_the_top() -> None:
    text = markdown(mixed(), oracle=True)
    assert text.splitlines()[2].startswith("> **ORACLE RUN.**")
    assert "say nothing about how well a language model answers" in text


def test_the_schema_only_mode_and_an_early_stop_are_stated() -> None:
    run = run_of(
        [res()],
        schema_only=True,
        stopped="stopped before S02: spent $1.0000 of $1.00",
        not_run=["S02"],
    )
    text = markdown(run)
    assert "- Privacy mode: schema only" in text
    assert "- **Stopped early:** stopped before S02" in text
    assert "## Not run\n\nS02" in text


def test_the_threshold_table_and_the_category_table_are_in_the_report() -> None:
    text = markdown(mixed())
    assert "| Result accuracy, all non-ambiguous cases | 85% or more | 50.0% | fail |" in text
    assert "| Adversarial SQL blocked | 100% |" in text
    assert "| Simple aggregation | 2 | 100.0% | 50.0% |" in text
    assert "| Ambiguous (clarification expected) | 2 | n/a | 50.0% |" in text


def test_the_cases_that_were_not_right_are_listed_with_why_and_the_sql() -> None:
    run = run_of(
        [
            res("S01"),
            res("S02", correct=False, reason="expected 1.0, got 2.0", sql="a | b"),
            res("S03", safe=False, reason=""),
        ]
    )
    text = markdown(run)
    section = text.split("## Cases that were not right")[1].split("##")[0]
    start = "| S02 | Simple aggregation | Question S02? | expected 1.0, got 2.0 |"
    assert start + " a \\| b |" in section
    assert "| S03 |" in section and "unsafe" in section
    assert "S01" not in section


def test_a_run_with_everything_right_says_none() -> None:
    assert "## Cases that were not right\n\nNone." in markdown(run_of([res()]))


def test_long_text_is_cut_and_flattened_in_tables() -> None:
    run = run_of([res("S02", correct=False, question="word " * 100, reason="a\nb", sql="x " * 200)])
    row = next(x for x in markdown(run).splitlines() if x.startswith("| S02 |"))
    assert "\n" not in row and row.count("…") == 2
    assert "a b" in row


def test_gold_queries_that_did_not_run_are_listed() -> None:
    assert "- X01: the validator refused it" in markdown(mixed())
    assert "## Gold queries that did not run" not in markdown(run_of([res()]))


# -- the manual review ---------------------------------------------------------------------------


def many(count: int) -> tuple[BenchmarkRun, list[Case]]:
    ids = [f"S{n:02d}" for n in range(1, count + 1)]
    cases = [case_for(i, answer_facts=("states the total",)) for i in ids]
    return run_of([res(i, answer=f"Answer to {i}") for i in ids]), cases


def test_the_review_sample_is_thirty_answered_cases_and_the_same_every_time() -> None:
    run, cases = many(50)
    sample = review_sample(run, cases)
    assert len(sample) == REVIEW_SAMPLE == 30
    assert [r.id for r, _ in sample] == sorted(r.id for r, _ in sample)
    assert [r.id for r, _ in review_sample(run, cases)] == [r.id for r, _ in sample]
    other = review_sample(replace(run, seed=2), cases)
    assert [r.id for r, _ in other] != [r.id for r, _ in sample]
    assert all(c.id == r.id for r, c in sample)


def test_the_review_sample_takes_only_answered_cases_and_all_of_a_small_run() -> None:
    run, cases = many(5)
    run.results[0] = replace(run.results[0], status="failed")
    run.results[1] = replace(run.results[1], status="clarification")
    assert [r.id for r, _ in review_sample(run, cases)] == ["S03", "S04", "S05"]
    assert review_sample(run, []) == []


def test_each_sampled_answer_has_a_tick_box_the_facts_and_the_assumptions() -> None:
    cases = [
        case_for(
            "S01", answer_facts=("states the total", "mentions NZD"), expected_assumptions=("a",)
        ),
        case_for("S02"),
    ]
    run = run_of(
        [res("S01", answer="It is 5.", assumptions=("Taken as a",)), res("S02", answer="Six.")]
    )
    text = markdown(run, cases)
    assert "## Manual review (2 answered cases)" in text
    assert "- [ ] **S01** Question S01?" in text
    assert "  - Answer: It is 5." in text
    assert "  - Should state: states the total; mentions NZD" in text
    assert "  - Assumptions expected: a. Shown: Taken as a" in text
    assert "  - Should state: no specific facts listed; check it matches the table" in text
    assert text.count("Assumptions expected") == 1


def test_assumptions_that_were_not_shown_say_so() -> None:
    run = run_of([res("S01")])
    text = markdown(run, [case_for("S01", expected_assumptions=("a",))])
    assert "Shown: none shown" in text


# -- the CSV and the files ----------------------------------------------------------------------


def test_the_csv_has_a_row_per_case_and_a_column_per_field() -> None:
    run = run_of(
        [
            res("S01", assumptions=("a", "b"), turn_statuses=("answered",), sql="SELECT 1, 2"),
            res("S02"),
        ]
    )
    rows = list(csv.reader(io.StringIO(render_csv(run))))
    assert rows[0] == [f.name for f in fields(CaseResult)]
    assert len(rows) == 3
    first = dict(zip(rows[0], rows[1], strict=True))
    assert first["id"] == "S01"
    assert first["assumptions"] == "a | b"
    assert first["turn_statuses"] == "answered"
    assert first["sql"] == "SELECT 1, 2"
    assert first["correct"] == "True"


def test_the_report_files_are_named_for_the_date_commit_and_split() -> None:
    run = run_of([res()])
    assert report_stem(run, "abc1234", "dev") == "2026-11-23-abc1234-dev"
    assert (
        report_stem(run, "abc1234-dirty", "holdout", oracle=True)
        == "2026-11-23-abc1234-dirty-holdout-oracle"
    )


def test_write_reports_writes_both_files(tmp_path: Path) -> None:
    run, cases = many(3)
    markdown_path, table_path = write_reports(
        run, cases, tmp_path / "out", split="dev", revision="abc1234"
    )
    assert markdown_path.name == "2026-11-23-abc1234-dev.md"
    assert table_path.name == "2026-11-23-abc1234-dev.csv"
    assert markdown_path.read_text(encoding="utf-8").startswith("# Benchmark report")
    assert "S01" in table_path.read_text(encoding="utf-8")
    oracle_md, _ = write_reports(
        run, cases, tmp_path / "out", split="dev", revision="abc1234", oracle=True
    )
    assert oracle_md.name == "2026-11-23-abc1234-dev-oracle.md"
    assert markdown_path.exists(), "an oracle report does not replace a model's"


# -- the commit id ---------------------------------------------------------------------------------


def fake_git(monkeypatch, *outputs: str) -> None:
    answers = iter(outputs)

    def run(*_args, **_kw):
        return subprocess.CompletedProcess([], 0, stdout=next(answers), stderr="")

    monkeypatch.setattr(subprocess, "run", run)


def test_the_revision_is_the_short_commit_id(monkeypatch) -> None:
    fake_git(monkeypatch, "abc1234\n", "")
    assert git_revision() == "abc1234"


def test_uncommitted_changes_to_tracked_files_make_it_dirty(monkeypatch) -> None:
    fake_git(monkeypatch, "abc1234\n", " M eval/run.py\n")
    assert git_revision() == "abc1234-dirty"


def test_without_git_the_revision_is_nogit(monkeypatch) -> None:
    def missing(*_args, **_kw):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing)
    assert git_revision() == "nogit"


def test_outside_a_repository_the_revision_is_nogit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert git_revision() == "nogit"


def test_in_this_repository_the_revision_is_a_commit_id(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    assert re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", git_revision())
