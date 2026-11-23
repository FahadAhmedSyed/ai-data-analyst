"""The question rate limit (FR-10.2, ADR 0012): 20 questions per 10 minutes per session.

A sliding window: a question counts for ``window_s`` seconds after it was asked. The limit
belongs to the session, not to the dataset, so replacing the file does not reset it. It
is a guard on the OpenAI budget for a small group of testers, not a defence against
someone who opens a new session each time (the shared password and the spend cap on the
OpenAI project are for that).
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable

from analyst.core.errors import RateLimitedError
from analyst.session.display import count_text


def _duration(seconds: float) -> str:
    """A length of time for a message: whole seconds under a minute, else whole minutes."""
    whole = max(1, math.ceil(seconds))
    if whole < 60:
        return f"{whole} second{'s' if whole != 1 else ''}"
    minutes = math.ceil(whole / 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


class RateLimiter:
    """Allows ``max_events`` events in any ``window_s`` seconds."""

    def __init__(
        self,
        max_events: int,
        window_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter.

        Args:
            max_events: Events allowed in one window.
            window_s: The window, in seconds.
            clock: Monotonic seconds; replaced in tests.
        """
        if max_events < 1 or window_s <= 0:
            raise ValueError("max_events must be at least 1 and window_s positive")
        self._max = max_events
        self._window = window_s
        self._clock = clock
        self._events: deque[float] = deque()

    def _forget_old(self, now: float) -> None:
        while self._events and now - self._events[0] >= self._window:
            self._events.popleft()

    def remaining(self) -> int:
        """How many more events are allowed right now."""
        self._forget_old(self._clock())
        return self._max - len(self._events)

    def acquire(self) -> None:
        """Count one event, or raise ``RateLimitedError`` saying how long to wait.

        A refused event is not counted, so waiting is never made longer by trying again.
        """
        now = self._clock()
        self._forget_old(now)
        if len(self._events) >= self._max:
            wait = self._events[0] + self._window - now
            raise RateLimitedError(
                f"{len(self._events)} events in {self._window:g} s",
                user_message=(
                    f"You've reached the limit of {count_text(self._max, 'question')} in "
                    f"{_duration(self._window)}. Please try again in about {_duration(wait)}."
                ),
            )
        self._events.append(now)
