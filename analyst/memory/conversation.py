"""What one session remembers of the conversation (T1.16, ADR 0011).

``ConversationMemory`` is a plain in-process list of ``Turn`` objects, one per question,
kept for the life of the browser session and never written anywhere (privacy.md,
section 2). It does three things for the pipeline:

* **A window.** ``recent()`` returns the last three turns, which is all that follow-up
  questions are resolved against (FR-7.1). The context builder shows the model these
  and the preview of the most recent result.
* **Carried assumptions.** An assumption the pipeline stated on an answered turn is one the
  user has seen and not contradicted, so it keeps applying for the rest of the session even
  after its turn has left the window (ADR 0011). ``carried_assumptions()`` returns them,
  oldest first, so that when two conflict the later one is the one to follow. Turns that
  asked for clarification or failed carry nothing: nothing was settled on them.
* **A reset.** ``reset()`` empties everything. It serves both "start a new conversation"
  and "replace the dataset" (FR-7.3, FR-7.4); the interface asks for confirmation before
  the second.

A new question always runs a new query, so nothing here is a cache of results.

Memory is not thread-safe. Each session has its own object and Streamlit handles one
interaction at a time per session.
"""

from __future__ import annotations

from collections import deque

from analyst.core.models import Turn

WINDOW_TURNS = 3
MAX_TURNS = 50
MAX_CARRIED = 8
MAX_ASSUMPTION_CHARS = 300


def _key(assumption: str) -> str:
    return " ".join(assumption.split()).casefold()


def _tidy(assumption: str) -> str:
    text = " ".join(assumption.split())
    return text if len(text) <= MAX_ASSUMPTION_CHARS else text[: MAX_ASSUMPTION_CHARS - 1] + "…"


class ConversationMemory:
    """The turns of one session and the assumptions that carry forward."""

    def __init__(
        self,
        *,
        window: int = WINDOW_TURNS,
        max_turns: int = MAX_TURNS,
        max_carried: int = MAX_CARRIED,
    ) -> None:
        """Create empty memory.

        Args:
            window: How many recent turns ``recent`` returns.
            max_turns: Turns kept; the oldest are dropped beyond this, which bounds memory
                in a long session. Carried assumptions outlive the turns they came from.
            max_carried: Assumptions carried; when there are more the oldest are dropped.
        """
        if window < 1 or max_turns < window or max_carried < 1:
            raise ValueError("window and max_carried must be at least 1, max_turns at least window")
        self._window = window
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._carried: dict[str, str] = {}  # normalised text to display text, oldest first
        self._max_carried = max_carried

    def __len__(self) -> int:
        """Number of turns kept."""
        return len(self._turns)

    @property
    def turns(self) -> tuple[Turn, ...]:
        """Every kept turn, oldest first."""
        return tuple(self._turns)

    @property
    def last_turn(self) -> Turn | None:
        """The most recent turn, or ``None`` at the start of a conversation."""
        return self._turns[-1] if self._turns else None

    def recent(self) -> tuple[Turn, ...]:
        """The last few turns, oldest first, for resolving a follow-up question."""
        return tuple(self._turns)[-self._window :]

    def carried_assumptions(self) -> tuple[str, ...]:
        """Assumptions from answered turns, oldest first, without repeats."""
        return tuple(self._carried.values())

    def record(self, turn: Turn) -> None:
        """Add a finished turn.

        Raises:
            TypeError: ``turn`` is not a ``Turn``.
        """
        if not isinstance(turn, Turn):
            raise TypeError(f"expected a Turn, got {type(turn).__name__}")
        self._turns.append(turn)
        if turn.status == "answered":
            for assumption in turn.assumptions:
                self._carry(assumption)

    def reset(self) -> None:
        """Forget every turn and carried assumption."""
        self._turns.clear()
        self._carried.clear()

    def _carry(self, assumption: str) -> None:
        text = _tidy(assumption)
        if not text:
            return
        key = _key(text)
        self._carried.pop(key, None)  # a repeat moves to the end: it is the latest word
        self._carried[key] = text
        while len(self._carried) > self._max_carried:
            del self._carried[next(iter(self._carried))]
