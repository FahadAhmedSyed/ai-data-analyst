"""Conversation memory (T1.16)."""

from __future__ import annotations

import pytest

from analyst.core.models import ResultSummary, Turn
from analyst.memory import ConversationMemory
from analyst.memory.conversation import (
    MAX_ASSUMPTION_CHARS,
    MAX_CARRIED,
    MAX_TURNS,
    WINDOW_TURNS,
)


def turn(n: int, status="answered", assumptions=(), **kw) -> Turn:
    return Turn(question=f"question {n}", status=status, assumptions=assumptions, **kw)


def filled(count: int) -> ConversationMemory:
    memory = ConversationMemory()
    for n in range(1, count + 1):
        memory.record(turn(n))
    return memory


def test_defaults_match_the_architecture_notes():
    assert (WINDOW_TURNS, MAX_TURNS, MAX_CARRIED) == (3, 50, 8)


def test_a_new_conversation_is_empty():
    memory = ConversationMemory()
    assert len(memory) == 0
    assert memory.turns == ()
    assert memory.recent() == ()
    assert memory.last_turn is None
    assert memory.carried_assumptions() == ()


def test_turns_are_kept_in_order():
    memory = filled(3)
    assert [t.question for t in memory.turns] == ["question 1", "question 2", "question 3"]
    assert len(memory) == 3
    assert memory.last_turn.question == "question 3"


@pytest.mark.parametrize(
    ("count", "shown"), [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (6, [4, 5, 6])]
)
def test_recent_is_the_last_three_turns_oldest_first(count, shown):
    memory = filled(count)
    assert [t.question for t in memory.recent()] == [f"question {n}" for n in shown]


def test_the_window_does_not_forget_older_turns():
    memory = filled(6)
    assert len(memory) == 6
    assert memory.turns[0].question == "question 1"


def test_a_custom_window():
    memory = ConversationMemory(window=2)
    for n in range(1, 5):
        memory.record(turn(n))
    assert [t.question for t in memory.recent()] == ["question 3", "question 4"]


def test_the_oldest_turns_are_dropped_beyond_the_limit():
    memory = ConversationMemory(window=2, max_turns=4)
    for n in range(1, 8):
        memory.record(turn(n))
    assert [t.question for t in memory.turns] == [f"question {n}" for n in (4, 5, 6, 7)]
    assert [t.question for t in memory.recent()] == ["question 6", "question 7"]


@pytest.mark.parametrize(
    "kwargs", [{"window": 0}, {"max_carried": 0}, {"window": 5, "max_turns": 4}]
)
def test_impossible_limits_are_refused(kwargs):
    with pytest.raises(ValueError, match="at least"):
        ConversationMemory(**kwargs)


def test_only_turns_can_be_recorded():
    memory = ConversationMemory()
    with pytest.raises(TypeError, match="Turn"):
        memory.record({"question": "x"})  # type: ignore[arg-type]
    assert len(memory) == 0


def test_turns_keep_their_result_summary_for_the_context_builder():
    summary = ResultSummary(columns=("a",), row_count=1, preview_rows=({"a": 1},))
    memory = ConversationMemory()
    memory.record(turn(1, result_summary=summary))
    assert memory.recent()[0].result_summary == summary


# -- carried assumptions ----------------------------------------------------------------------


def test_assumptions_of_answered_turns_carry_forward_oldest_first():
    memory = ConversationMemory()
    memory.record(turn(1, assumptions=("Revenue means the revenue column.",)))
    memory.record(turn(2, assumptions=("Last year means 2024.", "Regions are the region column.")))
    assert memory.carried_assumptions() == (
        "Revenue means the revenue column.",
        "Last year means 2024.",
        "Regions are the region column.",
    )


def test_assumptions_outlive_the_window_and_the_turns_they_came_from():
    memory = ConversationMemory(window=2, max_turns=3)
    memory.record(turn(1, assumptions=("Sales means revenue.",)))
    for n in range(2, 9):
        memory.record(turn(n))
    assert "question 1" not in [t.question for t in memory.turns]
    assert memory.carried_assumptions() == ("Sales means revenue.",)


@pytest.mark.parametrize("status", ["clarification", "failed"])
def test_turns_that_settled_nothing_carry_nothing(status):
    memory = ConversationMemory()
    memory.record(turn(1, status=status, assumptions=("Not agreed.",)))
    assert memory.carried_assumptions() == ()
    assert len(memory) == 1  # the turn itself is still remembered


def test_a_repeated_assumption_is_kept_once_and_counts_as_the_latest():
    memory = ConversationMemory()
    memory.record(turn(1, assumptions=("A is x.", "B is y.")))
    memory.record(turn(2, assumptions=("  a   IS x. ",)))
    assert memory.carried_assumptions() == ("B is y.", "a IS x.")


def test_a_later_conflicting_assumption_comes_after_the_earlier_one():
    memory = ConversationMemory()
    memory.record(turn(1, assumptions=("Revenue means gross revenue.",)))
    memory.record(turn(2, assumptions=("Revenue means net revenue.",)))
    assert memory.carried_assumptions()[-1] == "Revenue means net revenue."


def test_only_the_most_recent_assumptions_are_kept():
    memory = ConversationMemory(max_carried=3)
    for n in range(1, 7):
        memory.record(turn(n, assumptions=(f"Assumption {n}.",)))
    assert memory.carried_assumptions() == ("Assumption 4.", "Assumption 5.", "Assumption 6.")


def test_a_repeat_is_not_pushed_out_by_newer_ones():
    memory = ConversationMemory(max_carried=3)
    memory.record(turn(1, assumptions=("Keep me.",)))
    memory.record(turn(2, assumptions=("Two.",)))
    memory.record(turn(3, assumptions=("Keep me.",)))  # restated, so it is the newest
    memory.record(turn(4, assumptions=("Four.",)))
    assert memory.carried_assumptions() == ("Two.", "Keep me.", "Four.")


def test_assumptions_are_flattened_and_cut():
    memory = ConversationMemory()
    memory.record(turn(1, assumptions=("two\nlines\tand   spaces", "x" * 1_000, "   ", "")))
    first, second = memory.carried_assumptions()
    assert first == "two lines and spaces"
    assert len(second) == MAX_ASSUMPTION_CHARS
    assert second.endswith("…")


def test_an_assumption_of_exactly_the_limit_is_kept_whole():
    memory = ConversationMemory()
    exact = "y" * MAX_ASSUMPTION_CHARS
    memory.record(turn(1, assumptions=(exact, exact + "z")))
    whole, cut = memory.carried_assumptions()
    assert whole == exact
    assert cut == "y" * (MAX_ASSUMPTION_CHARS - 1) + "…"


# -- reset ------------------------------------------------------------------------------------


def test_reset_forgets_everything():
    memory = filled(5)
    memory.record(turn(6, assumptions=("Something.",)))
    memory.reset()
    assert len(memory) == 0
    assert memory.recent() == ()
    assert memory.last_turn is None
    assert memory.carried_assumptions() == ()


def test_memory_works_normally_after_a_reset():
    memory = filled(4)
    memory.reset()
    memory.record(turn(1, assumptions=("Fresh.",)))
    assert [t.question for t in memory.recent()] == ["question 1"]
    assert memory.carried_assumptions() == ("Fresh.",)


def test_two_sessions_do_not_share_memory():
    first, second = ConversationMemory(), ConversationMemory()
    first.record(turn(1, assumptions=("Only in the first.",)))
    assert len(second) == 0
    assert second.carried_assumptions() == ()
