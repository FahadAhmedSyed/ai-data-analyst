"""The question rate limit (FR-10.2): a sliding window, with a message that says when to retry."""

from __future__ import annotations

import pytest

from analyst.core.errors import RateLimitedError
from analyst.session.limiter import RateLimiter, _duration


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def limiter(max_events: int = 3, window: float = 600.0) -> tuple[RateLimiter, Clock]:
    clock = Clock()
    return RateLimiter(max_events, window, clock=clock), clock


def test_events_up_to_the_limit_are_allowed() -> None:
    rl, _ = limiter(3)
    for _ in range(3):
        rl.acquire()
    assert rl.remaining() == 0


def test_the_event_after_the_limit_is_refused() -> None:
    rl, _ = limiter(3)
    for _ in range(3):
        rl.acquire()
    with pytest.raises(RateLimitedError) as caught:
        rl.acquire()
    assert caught.value.category == "questions"


def test_remaining_counts_down() -> None:
    rl, _ = limiter(3)
    assert rl.remaining() == 3
    rl.acquire()
    assert rl.remaining() == 2


def test_an_event_stops_counting_when_the_window_has_passed() -> None:
    rl, clock = limiter(2, window=600)
    rl.acquire()
    clock.now += 100
    rl.acquire()
    clock.now += 499.9  # the first is 599.9 s old: still counted
    assert rl.remaining() == 0
    clock.now += 0.1  # exactly one window old: free
    assert rl.remaining() == 1
    rl.acquire()
    with pytest.raises(RateLimitedError):
        rl.acquire()


def test_the_window_slides_event_by_event() -> None:
    rl, clock = limiter(2, window=10)
    rl.acquire()  # t=0
    clock.now += 6
    rl.acquire()  # t=6
    clock.now += 5  # t=11: the first has gone, the second has not
    assert rl.remaining() == 1


def test_a_refused_event_is_not_counted_and_does_not_lengthen_the_wait() -> None:
    rl, clock = limiter(1, window=600)
    rl.acquire()
    clock.now += 60
    for _ in range(5):
        with pytest.raises(RateLimitedError):
            rl.acquire()
    clock.now += 540  # 600 s after the one that counted
    rl.acquire()  # allowed: the refusals did not count


def test_the_message_names_the_limit_the_window_and_the_wait() -> None:
    rl, clock = limiter(20, window=600)
    for _ in range(20):
        rl.acquire()
    clock.now += 250  # the oldest was asked 250 s ago, so 350 s to go
    with pytest.raises(RateLimitedError) as caught:
        rl.acquire()
    message = caught.value.user_message
    assert "20 questions in 10 minutes" in message
    assert "about 6 minutes" in message  # 350 s rounded up


def test_a_short_wait_is_in_seconds() -> None:
    rl, clock = limiter(1, window=600)
    rl.acquire()
    clock.now += 570
    with pytest.raises(RateLimitedError) as caught:
        rl.acquire()
    assert "about 30 seconds" in caught.value.user_message


def test_the_technical_detail_is_for_logs_and_has_no_message_text() -> None:
    rl, _ = limiter(1, window=600)
    rl.acquire()
    with pytest.raises(RateLimitedError) as caught:
        rl.acquire()
    assert caught.value.detail == "1 events in 600 s"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0.2, "1 second"),
        (1, "1 second"),
        (1.01, "2 seconds"),
        (59, "59 seconds"),
        (59.5, "1 minute"),
        (60, "1 minute"),
        (61, "2 minutes"),
        (120, "2 minutes"),
        (600, "10 minutes"),
    ],
)
def test_durations_are_rounded_up_to_whole_units(seconds: float, text: str) -> None:
    assert _duration(seconds) == text


@pytest.mark.parametrize(("max_events", "window"), [(0, 60), (-1, 60), (1, 0), (1, -5)])
def test_a_limiter_that_could_never_allow_anything_or_never_forget_is_refused(
    max_events: int, window: float
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RateLimiter(max_events, window)


def test_it_uses_the_monotonic_clock_by_default() -> None:
    rl = RateLimiter(1, 60)
    rl.acquire()
    with pytest.raises(RateLimitedError):
        rl.acquire()


@pytest.mark.parametrize(("max_events", "window"), [(1, 1), (1, 0.5), (1, 0.001), (2, 60)])
def test_the_smallest_limits_that_make_sense_are_accepted(max_events: int, window: float) -> None:
    limiter = RateLimiter(max_events, window)
    assert limiter.remaining() == max_events
