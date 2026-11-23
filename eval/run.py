"""The command line: ``uv run python -m eval.run`` (eval/README.md section 5).

    uv run python -m eval.run --split dev                 # a model run on the dev cases
    uv run python -m eval.run --split holdout --milestone M4
    uv run python -m eval.run --max-cost-usd 2.00         # stop before the spend passes $2
    uv run python -m eval.run --oracle                    # prove the harness, no model, no key

A model run needs ``OPENAI_API_KEY`` and the two model names in the environment (never in a
file that is committed). The held-out cases are for milestone reports: asking for them
needs ``--milestone``, so the report says which milestone it is for.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from analyst.config.settings import Settings, get_settings
from analyst.core.logging import configure_logging
from analyst.llm.client import OpenAIClient
from eval.cases import Case, CaseFileError, load_cases, select
from eval.generate_datasets import SEED
from eval.oracle import MODEL_NAME, OracleClient
from eval.report import compute_metrics, git_revision, threshold_rows, write_reports
from eval.runner import CaseResult, ClientFactory, run_benchmark

EXIT_USAGE = 2


class SetupError(Exception):
    """The run cannot start as asked; the message says what to change."""


def build_parser() -> argparse.ArgumentParser:
    """The command line's options."""
    parser = argparse.ArgumentParser(prog="eval.run", description="Run the benchmark.")
    parser.add_argument("--cases", type=Path, default=Path("eval/cases"), help="a file or a folder")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--ids", nargs="+", metavar="ID", help="only these case ids")
    parser.add_argument("--max-cost-usd", type=float, help="stop before a case would pass this")
    parser.add_argument("--milestone", help="name of the milestone, required for holdout and all")
    parser.add_argument("--oracle", action="store_true", help="answer with the gold SQL, no model")
    parser.add_argument(
        "--schema-only", action="store_true", help="the privacy mode with no values"
    )
    parser.add_argument("--out", type=Path, default=Path("eval/reports"), help="report folder")
    parser.add_argument("--seed", type=int, default=SEED, help=argparse.SUPPRESS)
    return parser


def _real_client(settings: Settings, max_cost_usd: float | None) -> tuple[ClientFactory, str]:
    """The OpenAI client for every case, and how the report names the models."""
    if not settings.llm_configured:
        msg = (
            "No model is configured. Set OPENAI_API_KEY, LLM_MODEL_GENERATE and "
            "LLM_MODEL_EXPLAIN in the environment (see .env.example), or use --oracle to "
            "run the harness without a model."
        )
        raise SetupError(msg)
    names = [settings.llm_model_generate or "", settings.llm_model_explain or ""]
    if max_cost_usd is not None and not all(name in settings.llm_prices for name in names):
        msg = (
            "--max-cost-usd needs a price for both models in LLM_PRICES, or the cost of a "
            "case is unknown and the cap cannot work."
        )
        raise SetupError(msg)
    client = OpenAIClient(settings)
    return (lambda _case, _table: client), f"{names[0]} (generate), {names[1]} (explain)"


def _line(result: CaseResult, number: int, total: int) -> None:
    mark = "ok  " if result.correct else "MISS"
    cost = f" ${result.cost_usd:.4f}" if result.cost_usd else ""
    head = f"[{number}/{total}] {mark} {result.id:<4} {result.status:<13}"
    print(f"{head} {result.latency_s:5.1f} s{cost}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its report. Returns the exit code."""
    args = build_parser().parse_args(argv)
    configure_logging("WARNING", stream=sys.stderr)
    try:
        cases: list[Case] = select(load_cases(args.cases), args.split, args.ids)
    except CaseFileError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    if not cases:
        print("No cases to run.", file=sys.stderr)
        return EXIT_USAGE
    if args.split != "dev" and not args.oracle and not args.milestone:
        print(
            "Held-out cases are for milestone reports only: add --milestone M3 (or M4, M5), "
            "or run --split dev.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.max_cost_usd is not None and args.max_cost_usd <= 0:
        print("--max-cost-usd must be more than zero.", file=sys.stderr)
        return EXIT_USAGE

    settings = get_settings()
    if args.oracle:
        factory: ClientFactory = lambda case, table: OracleClient.for_case(case, table)  # noqa: E731
        model = f"{MODEL_NAME} (not a model: returns the gold SQL)"
    else:
        try:
            factory, model = _real_client(settings, args.max_cost_usd)
        except SetupError as error:
            print(error, file=sys.stderr)
            return EXIT_USAGE

    run = run_benchmark(
        cases,
        factory,
        model=model,
        settings=settings,
        seed=args.seed,
        schema_only=args.schema_only,
        max_cost_usd=args.max_cost_usd,
        on_result=_line,
    )
    markdown, table = write_reports(
        run,
        cases,
        args.out,
        split=args.split,
        milestone=args.milestone,
        oracle=args.oracle,
        revision=git_revision(),
    )
    metrics = compute_metrics(run)
    for row in threshold_rows(metrics):
        print(f"{row.verdict:<18} {row.metric}: {row.measured}")
    if run.stopped:
        print(run.stopped)
    print(f"Report: {markdown}\nTable:  {table}")
    if args.oracle:  # the oracle is perfect by construction: any miss is a bug in the harness
        perfect = all(r.correct and r.safe for r in run.results) and not run.not_run
        return 0 if perfect else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
