"""Shared fixtures for the benchmark's tests.

The datasets are generated and loaded once for the whole test session, and so is one full run
of every case with the oracle (about ten seconds), because most of these tests only read them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analyst.config.settings import Settings
from eval.cases import Case, load_cases
from eval.oracle import OracleClient
from eval.runner import BenchmarkRun, DatasetPool, run_benchmark

CASES_DIR = Path(__file__).resolve().parents[2] / "eval" / "cases"


@pytest.fixture(scope="session")
def bench_settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="session")
def pool(bench_settings: Settings):
    with DatasetPool(bench_settings) as datasets:
        yield datasets


@pytest.fixture(scope="session")
def shipped_cases() -> list[Case]:
    return load_cases(CASES_DIR)


@pytest.fixture(scope="session")
def oracle_run(shipped_cases: list[Case], bench_settings: Settings) -> BenchmarkRun:
    return run_benchmark(
        shipped_cases,
        lambda case, table: OracleClient.for_case(case, table),
        model="oracle",
        settings=bench_settings,
    )
