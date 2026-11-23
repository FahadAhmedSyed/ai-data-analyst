"""The case files: the format and its checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval.cases import (
    Case,
    CaseFileError,
    load_cases,
    parse_cases,
    problems_with,
    select,
)


def case(**changes) -> dict:
    fields = {
        "id": "S99",
        "dataset": "sales",
        "category": "simple_aggregation",
        "question": "How many orders are there?",
        "gold_sql": "SELECT COUNT(*) AS orders FROM sales",
        "compare": "scalar",
    }
    return {k: v for k, v in {**fields, **changes}.items() if v != "DROP"}


def text(*cases: dict) -> str:
    return yaml.safe_dump({"cases": list(cases)})


def parsed(**changes) -> Case:
    return parse_cases(text(case(**changes)))[0]


# -- the format ------------------------------------------------------------------------------------


def test_a_case_gets_its_defaults() -> None:
    c = parsed(compare="DROP")
    assert c.compare == "unordered"
    assert (c.turns, c.alt_gold_sql, c.holdout, c.clarify) == ((), None, False, False)
    assert c.expected_status == "answered"
    assert c.golds == ("SELECT COUNT(*) AS orders FROM sales",)
    assert c.expects_answer


def test_a_clarification_case_expects_a_clarification() -> None:
    c = parsed(
        id="A99", category="ambiguous", gold_sql=None, clarify=True, clarify_about="what is meant"
    )
    assert c.expected_status == "clarification"
    assert not c.expects_answer
    assert c.golds == ()


def test_sql_is_put_on_one_line_and_blank_sql_is_none() -> None:
    c = parsed(gold_sql="SELECT COUNT(*)\n   AS orders\nFROM sales\n", alt_gold_sql="  ")
    assert c.gold_sql == "SELECT COUNT(*) AS orders FROM sales"
    assert c.alt_gold_sql is None


def test_alt_gold_is_accepted_after_gold() -> None:
    c = parsed(alt_gold_sql="SELECT COUNT(order_id) AS orders FROM sales")
    assert c.golds == (c.gold_sql, "SELECT COUNT(order_id) AS orders FROM sales")


def test_turns_and_lists_are_read_as_text() -> None:
    c = parsed(
        id="T99", category="follow_up", turns=["First?", 2024], expected_assumptions=["a", 1]
    )
    assert c.turns == ("First?", "2024")
    assert c.expected_assumptions == ("a", "1")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dataset": "weather"}, "unknown dataset 'weather'"),
        ({"category": "joins"}, "unknown category 'joins'"),
        ({"id": "F01"}, "id should be S followed by a number"),
        ({"id": "Sx"}, "id should be S followed by a number"),
        ({"id": "SA1"}, "id should be S followed by a number"),
        ({"compare": "approximately"}, "compare must be one of"),
        ({"expected_status": "maybe"}, "expected_status must be one of"),
        ({"gold_sql": "DROP"}, "needs gold_sql"),
        ({"clarify": True, "expected_status": "answered"}, "clarify and expected_status disagree"),
        ({"expected_status": "failed"}, "has no gold_sql"),
        (
            {"alt_gold_sql": "SELECT 1", "gold_sql": "DROP", "expected_status": "failed"},
            "alt_gold_sql without gold_sql",
        ),
        ({"category": "follow_up", "id": "T99"}, "needs earlier turns"),
        ({"turns": list("abcdefg")}, "at most 6 earlier turns"),
        (
            {"expected_status": "clarification", "clarify": True, "clarify_about": "x"},
            "has no gold_sql",
        ),
    ],
)
def test_a_case_with_a_problem_is_refused_and_the_message_says_what(changes, message) -> None:
    with pytest.raises(CaseFileError, match=message):
        parse_cases(text(case(**changes)))


def test_six_earlier_turns_are_allowed() -> None:
    assert len(parsed(turns=list("abcdef")).turns) == 6


def test_a_clarification_case_must_say_what_it_is_about() -> None:
    c = case(id="A99", category="ambiguous", gold_sql="DROP", clarify=True)
    with pytest.raises(CaseFileError, match="says what the question should be about"):
        parse_cases(text(c))


def test_assumptions_need_an_answer() -> None:
    c = case(
        id="A99", category="ambiguous", gold_sql="DROP", clarify=True, clarify_about="x",
        expected_assumptions=["something"],
    )  # fmt: skip
    with pytest.raises(CaseFileError, match="only make sense when an answer is expected"):
        parse_cases(text(c))


def test_problems_with_a_good_case_is_empty() -> None:
    assert problems_with(parsed()) == []


def test_a_question_that_is_blank_is_refused() -> None:
    with pytest.raises(CaseFileError, match="no question"):
        parse_cases(text(case(question=" ")))


def test_every_problem_in_a_file_is_listed_at_once() -> None:
    bad = text(case(dataset="x"), case(id="S98", category="y"))
    with pytest.raises(CaseFileError) as error:
        parse_cases(bad)
    assert "S99: unknown dataset" in str(error.value)
    assert "S98: unknown category" in str(error.value)


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(CaseFileError, match="S99: id used 2 times"):
        parse_cases(text(case(), case()))


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("cases: [unclosed", "not valid YAML"),
        ("just text", "expected a top-level 'cases' list"),
        ("cases: 5", "expected a top-level 'cases' list"),
        ("cases:\n  - just text", "mine.yaml item 1: not a mapping"),
        (
            text({"id": "S99", "dataset": "sales", "category": "filters"}),
            "missing fields \\['question'\\]",
        ),
        (text(case(colour="red")), "unknown fields \\['colour'\\]"),
    ],
)
def test_a_file_that_is_not_a_case_file_is_refused(document: str, message: str) -> None:
    with pytest.raises(CaseFileError, match=message):
        parse_cases(document, source="mine.yaml")


# -- folders and files -----------------------------------------------------------------------------


def test_a_folder_loads_every_yaml_file_by_name(tmp_path: Path) -> None:
    (tmp_path / "b.yaml").write_text(text(case(id="S02")))
    (tmp_path / "a.yaml").write_text(text(case(id="S01")))
    (tmp_path / "notes.txt").write_text("not a case file")
    assert [c.id for c in load_cases(tmp_path)] == ["S01", "S02"]
    assert [c.id for c in load_cases(tmp_path / "b.yaml")] == ["S02"]


def test_ids_must_be_unique_across_files(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(text(case()))
    (tmp_path / "b.yaml").write_text(text(case()))
    with pytest.raises(CaseFileError, match="S99: id used 2 times"):
        load_cases(tmp_path)


def test_a_folder_without_case_files_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CaseFileError, match=r"no \.yaml case files"):
        load_cases(tmp_path)


# -- splits ----------------------------------------------------------------------------------------


def three() -> list[Case]:
    return parse_cases(
        text(case(id="S01"), case(id="S02", holdout=True), case(id="S03", holdout=False))
    )


def test_dev_is_everything_not_held_out_and_holdout_is_the_rest() -> None:
    assert [c.id for c in select(three())] == ["S01", "S03"]
    assert [c.id for c in select(three(), "dev")] == ["S01", "S03"]
    assert [c.id for c in select(three(), "holdout")] == ["S02"]
    assert [c.id for c in select(three(), "all")] == ["S01", "S02", "S03"]


def test_named_ids_pick_cases_in_file_order() -> None:
    assert [c.id for c in select(three(), "all", ["S03", "S01", "S03"])] == ["S01", "S03"]


def test_an_id_that_is_not_in_the_split_is_an_error_not_a_skipped_case() -> None:
    with pytest.raises(CaseFileError, match="not in the dev split: S02"):
        select(three(), "dev", ["S01", "S02"])
    with pytest.raises(CaseFileError, match="not in the all split: S77"):
        select(three(), "all", ["S77"])
