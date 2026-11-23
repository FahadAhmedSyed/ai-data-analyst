"""The shipped cases: 80 of them, in the planned categories, each one usable."""

from __future__ import annotations

from collections import Counter

from eval.cases import CATEGORIES, CATEGORY_TITLES
from eval.generate_datasets import DATASETS
from eval.scoring import tied_at_cutoff

SHIPPED = {
    "simple_aggregation": 10, "filters": 9, "grouping_sorting": 9, "dates_seasons": 11,
    "nulls_duplicates": 6, "multi_aggregation": 6, "window_ranking": 6, "ambiguous": 8,
    "follow_up": 7, "empty_result": 3, "injection": 5,
}  # fmt: skip


def test_the_shipped_cases_are_eighty_in_the_planned_categories(shipped_cases) -> None:
    assert len(shipped_cases) == 80
    assert dict(Counter(c.category for c in shipped_cases)) == SHIPPED
    assert set(CATEGORIES) == set(CATEGORY_TITLES) == set(SHIPPED)


def test_the_shipped_cases_have_unique_ids_that_match_their_category(shipped_cases) -> None:
    assert len({c.id for c in shipped_cases}) == 80
    assert all(c.id[0] == CATEGORIES[c.category] for c in shipped_cases)


def test_every_dataset_is_used(shipped_cases) -> None:
    assert {c.dataset for c in shipped_cases} == set(DATASETS)


def test_about_thirty_percent_are_held_out(shipped_cases) -> None:
    held = sum(c.holdout for c in shipped_cases)
    assert 0.28 <= held / len(shipped_cases) <= 0.35
    for category, count in SHIPPED.items():
        if count >= 5:
            splits = {c.holdout for c in shipped_cases if c.category == category}
            assert splits == {True, False}, f"{category} needs cases in both splits"


def test_ambiguous_cases_ask_and_injection_cases_that_refuse_have_no_gold(shipped_cases) -> None:
    ambiguous = [c for c in shipped_cases if c.category == "ambiguous"]
    assert all(c.clarify and c.clarify_about and not c.golds for c in ambiguous)
    refused = [c for c in shipped_cases if c.expected_status == "failed"]
    assert {c.id for c in refused} == {"I02", "I04", "I05"}
    assert all(not c.golds and c.category == "injection" for c in refused)


def test_only_follow_ups_have_earlier_turns(shipped_cases) -> None:
    with_turns = [c for c in shipped_cases if c.turns]
    assert {c.category for c in with_turns} == {"follow_up"}
    assert len(with_turns) == 7
    assert all(1 <= len(c.turns) <= 2 for c in with_turns)


def test_every_gold_query_runs_through_the_real_validator_and_executor(oracle_run) -> None:
    assert [r.id for r in oracle_run.results if r.invalid] == []
    assert not any(r.gold_blocked for r in oracle_run.results)
    assert len(oracle_run.results) == 80


def test_the_gold_results_are_well_formed(shipped_cases, pool) -> None:
    problems = []
    for c in shipped_cases:
        for sql in c.golds:
            expected = pool.get(c.dataset).gold.expected(sql)
            frame = expected.frame
            if c.compare == "scalar" and frame.shape != (1, 1):
                problems.append(f"{c.id}: scalar but {frame.shape}")
            if c.compare == "ordered" and not expected.sort_columns:
                problems.append(f"{c.id}: ordered but no sort key to compare on")
            if not set(c.key_columns) <= {str(x) for x in frame.columns}:
                problems.append(f"{c.id}: key_columns not in the result")
            if frame.empty and c.category != "empty_result":
                problems.append(f"{c.id}: empty result")
            if tied_at_cutoff(expected):
                problems.append(f"{c.id}: tie at the LIMIT cut-off")
            if len(frame) > 400:
                problems.append(f"{c.id}: {len(frame)} rows is a lot to read")
    assert problems == []
