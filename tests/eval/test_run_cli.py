"""The command line: ``python -m eval.run``."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from analyst.config.settings import get_settings
from analyst.llm.fake import FakeLLMClient
from eval import run as cli
from eval.oracle import OracleClient, replies_for
from eval.runner import CaseResult
from tests.eval.conftest import CASES_DIR

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _keep_the_loggers_state():
    """The command line configures logging; put it back so other tests' log capture works."""
    logger = logging.getLogger("analyst")
    level, handlers, propagate = logger.level, list(logger.handlers), logger.propagate
    yield
    logger.setLevel(level)
    logger.handlers[:] = handlers
    logger.propagate = propagate


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "reports"


def oracle(*extra: str, out: Path, **kw) -> list[str]:
    return ["--oracle", "--cases", str(CASES_DIR), "--out", str(out), *extra]


def test_an_oracle_run_writes_both_reports_and_exits_zero(out: Path, capsys) -> None:
    code = cli.main(oracle("--ids", "S02", "F04", "A04", out=out))
    printed = capsys.readouterr().out
    assert code == 0
    assert "[1/3] ok   S02" in printed and "[3/3] ok   A04" in printed
    assert "pass               Gold SQL wrongly blocked by the validator: 0" in printed
    assert "Report: " in printed and "Table:  " in printed
    names = sorted(p.name for p in out.iterdir())
    assert len(names) == 2
    assert all(n.endswith("-dev-oracle.md") or n.endswith("-dev-oracle.csv") for n in names)
    report = next(out.glob("*.md")).read_text(encoding="utf-8")
    assert "ORACLE RUN" in report


def test_the_oracle_may_run_the_held_out_cases_without_a_milestone(out: Path) -> None:
    assert cli.main(oracle("--split", "holdout", "--ids", "S03", out=out)) == 0
    assert any(p.name.endswith("-holdout-oracle.md") for p in out.iterdir())


def test_an_oracle_run_that_is_not_perfect_exits_one(tmp_path: Path, out: Path) -> None:
    bad = {"cases": [{
        "id": "S90", "dataset": "sales", "category": "simple_aggregation",
        "question": "How many orders?", "gold_sql": "DELETE FROM sales", "compare": "scalar",
    }]}  # fmt: skip
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    assert cli.main(["--oracle", "--cases", str(path), "--out", str(out)]) == 1
    report = next(out.glob("*.md")).read_text(encoding="utf-8")
    assert "| Gold SQL wrongly blocked by the validator | 0 | 1 | fail |" in report


def test_a_bad_case_file_is_reported_and_exits_two(tmp_path: Path, out: Path, capsys) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("cases:\n  - id: S1\n")
    assert cli.main(["--cases", str(path), "--out", str(out)]) == 2
    assert "missing fields" in capsys.readouterr().err
    assert not out.exists()


def test_an_unknown_case_id_is_an_error_not_a_skipped_case(out: Path, capsys) -> None:
    assert cli.main(oracle("--ids", "S02", "S777", out=out)) == 2
    assert "not in the dev split: S777" in capsys.readouterr().err


def test_a_split_with_no_cases_says_so_and_exits_two(tmp_path: Path, out: Path, capsys) -> None:
    only_dev = {"cases": [{
        "id": "S90", "dataset": "sales", "category": "simple_aggregation",
        "question": "How many orders?", "gold_sql": "SELECT COUNT(*) FROM sales",
        "compare": "scalar",
    }]}  # fmt: skip
    path = tmp_path / "dev_only.yaml"
    path.write_text(yaml.safe_dump(only_dev))
    code = cli.main(["--oracle", "--cases", str(path), "--split", "holdout", "--out", str(out)])
    assert code == 2
    assert "No cases to run." in capsys.readouterr().err
    assert not out.exists()


def test_a_case_line_shows_the_cost_only_when_there_was_one(capsys) -> None:
    def result(cost: float) -> CaseResult:
        return CaseResult(
            id="S02", dataset="sales", category="simple_aggregation", holdout=False,
            question="q", expected_status="answered", status="answered", correct=True,
            cost_usd=cost, latency_s=1.5,
        )  # fmt: skip

    cli._line(result(0.01234), 1, 2)
    cli._line(result(0.0), 2, 2)
    first, second = capsys.readouterr().out.splitlines()
    assert first == "[1/2] ok   S02  answered        1.5 s $0.0123"
    assert second == "[2/2] ok   S02  answered        1.5 s"


def test_held_out_cases_need_a_milestone_for_a_model_run(out: Path, capsys) -> None:
    for split in ("holdout", "all"):
        code = cli.main(["--cases", str(CASES_DIR), "--split", split, "--out", str(out)])
        assert code == 2
        assert "--milestone" in capsys.readouterr().err


def test_a_cost_cap_must_be_more_than_zero(out: Path, capsys) -> None:
    assert cli.main(oracle("--max-cost-usd", "0", out=out)) == 2
    assert "more than zero" in capsys.readouterr().err


def test_a_model_run_without_a_configured_model_says_how_to_fix_it(out: Path, capsys) -> None:
    assert cli.main(["--cases", str(CASES_DIR), "--out", str(out), "--ids", "S02"]) == 2
    error = capsys.readouterr().err
    assert "No model is configured" in error
    assert "OPENAI_API_KEY" in error and "--oracle" in error


@pytest.fixture
def configured(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL_GENERATE", "gen-model")
    monkeypatch.setenv("LLM_MODEL_EXPLAIN", "exp-model")
    get_settings.cache_clear()


def test_a_cost_cap_needs_prices_for_both_models(configured, out: Path, capsys) -> None:
    args = ["--cases", str(CASES_DIR), "--out", str(out), "--ids", "S02", "--max-cost-usd", "1"]
    assert cli.main(args) == 2
    assert "needs a price for both models" in capsys.readouterr().err


def test_the_real_client_is_the_openai_client_and_the_report_names_both_models(
    configured, monkeypatch
) -> None:
    made = []

    class Stub:
        def __init__(self, settings) -> None:
            made.append(settings)

    monkeypatch.setattr(cli, "OpenAIClient", Stub)
    factory, label = cli._real_client(get_settings(), None)
    assert label == "gen-model (generate), exp-model (explain)"
    assert factory(None, "t") is factory(None, "t") is not None
    assert len(made) == 1, "one client is shared by every case"


def test_a_model_run_writes_a_report_and_exits_zero_even_when_cases_are_wrong(
    out: Path, monkeypatch, capsys
) -> None:
    def factory(case, table):
        replies = replies_for(case, table)
        wrong = [r.model_copy(update={"sql": "SELECT 1 AS wrong"}) if r.sql else r for r in replies]
        return OracleClient(wrong)

    monkeypatch.setattr(cli, "_real_client", lambda settings, cap: (factory, "a pretend model"))
    code = cli.main(["--cases", str(CASES_DIR), "--out", str(out), "--ids", "S02", "S05"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "[1/2] MISS S02" in printed
    report = next(out.glob("*.md")).read_text(encoding="utf-8")
    assert "- Model: a pretend model" in report
    assert "ORACLE" not in report
    assert "-dev.md" in next(out.glob("*.md")).name
    assert "| S02 |" in report.split("## Cases that were not right")[1]


def test_a_cost_cap_stops_the_run_and_the_report_says_where(out: Path, monkeypatch, capsys) -> None:
    from analyst.core.models import ModelExplanation, Usage

    def factory(case, table):
        script = []
        for reply in replies_for(case, table):
            script += [
                ("generate", reply),
                ("explain", ModelExplanation(answer_markdown="Done.", key_numbers=[], caveats=[])),
            ]
        return FakeLLMClient(script, usage=Usage(llm_calls=1, cost_usd=0.01))

    monkeypatch.setattr(cli, "_real_client", lambda settings, cap: (factory, "priced"))
    args = ["--cases", str(CASES_DIR), "--out", str(out), "--ids", "S02", "S05", "F04"]
    assert cli.main([*args, "--max-cost-usd", "0.03"]) == 0
    printed = capsys.readouterr().out
    assert "stopped before S05" in printed
    report = next(out.glob("*.md")).read_text(encoding="utf-8")
    assert "**Stopped early:** stopped before S05" in report
    assert "## Not run\n\nS05, F04" in report


def test_the_module_runs_from_the_command_line(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "eval.run", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 0
    assert "--oracle" in done.stdout and "--max-cost-usd" in done.stdout
    assert "--seed" not in done.stdout
