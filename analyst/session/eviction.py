"""Removing the data of sessions that have been idle too long (ADR 0014: 60 minutes).

The privacy notice promises that a file is deleted after 60 minutes of inactivity. A session
that nobody touches runs no code, so something has to look: ``start_sweeper`` runs one daemon
thread per process that wakes every minute and asks each live session to expire itself if it
is idle (``AnalystSession.expire_if_idle``). A user who comes back after the sweeper has
run finds the data gone and is asked to upload the file again; a user who comes back before
it runs is caught by the same check at the start of their next interaction.

This is the one piece of module-level state that touches sessions, and it is held weakly:
the registry is a ``WeakSet``, so it never keeps a session (or its data) alive, and the only
thing the sweeper can do with a session is ask it whether it is idle.
"""

from __future__ import annotations

import threading
import weakref
from typing import Protocol

from analyst.core.logging import get_logger

log = get_logger(__name__)

SWEEP_INTERVAL_S = 60.0


class Expirable(Protocol):
    """What the sweeper needs from a session."""

    def expire_if_idle(self) -> bool:
        """Remove the session's data if it is idle; return whether it did."""
        ...


_sessions: weakref.WeakSet[Expirable] = weakref.WeakSet()
_registry_lock = threading.Lock()
_sweeper: threading.Thread | None = None
_stop = threading.Event()


def register(session: Expirable) -> None:
    """Make ``session`` known to the sweeper, until it is garbage collected."""
    with _registry_lock:
        _sessions.add(session)


def sweep() -> int:
    """Expire every idle session once. Returns how many were expired."""
    with _registry_lock:
        sessions = list(_sessions)
    expired = 0
    for session in sessions:
        try:
            expired += bool(session.expire_if_idle())
        except Exception as error:  # one bad session must not stop the others being checked
            log.error("sweep_failed", extra={"error_type": type(error).__name__})
    if expired:
        log.info("sessions_swept", extra={"expired": expired})
    return expired


def _run(stop: threading.Event, interval_s: float) -> None:
    while not stop.wait(interval_s):
        sweep()


def start_sweeper(interval_s: float = SWEEP_INTERVAL_S) -> threading.Thread:
    """Start the sweeper thread if it is not running, and return it. Safe to call often."""
    global _sweeper, _stop
    with _registry_lock:
        if _sweeper is None or not _sweeper.is_alive():
            _stop = threading.Event()
            _sweeper = threading.Thread(
                target=_run, args=(_stop, interval_s), name="session-sweeper", daemon=True
            )
            _sweeper.start()
        return _sweeper


def stop_sweeper() -> None:
    """Stop the sweeper thread and wait for it. Used by tests; the process ending does it too."""
    global _sweeper
    with _registry_lock:
        thread, _sweeper = _sweeper, None
    _stop.set()
    if thread is not None:
        thread.join(timeout=5)
