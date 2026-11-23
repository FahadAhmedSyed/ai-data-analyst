"""One browser session: its database, dataset, conversation and rate limit (architecture section 3).

``AnalystSession`` is what the interface keeps in ``st.session_state``, one per browser
session and never shared (T8, NFR-4). It owns the session's DuckDB database and everything
derived from the data in it, so that forgetting a dataset, resetting a conversation or
evicting an idle session is one call that cannot leave a piece behind.

What is deliberately **not** reset by loading another file: the rate limiter. Replacing the
dataset must not give a session a fresh allowance of questions (FR-10.2).

Like the orchestrator, ``load`` and ``ask`` do not raise for a failure of the file or the
question: a failed load is recorded in ``load_failure`` and a question that could not run is
a ``ChatEntry`` with a notice. The interface shows what it is given.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from analyst.config.settings import Settings, get_settings
from analyst.core.errors import IngestionError, RateLimitedError
from analyst.core.logging import get_logger
from analyst.core.models import AnalystResponse
from analyst.database.session_db import SessionDatabase
from analyst.llm.client import LLMClient, OpenAIClient
from analyst.memory.conversation import ConversationMemory
from analyst.pipeline.analyst import Analyst, ProgressCallback
from analyst.session.dataset import LoadedDataset, ReadOptions, load_dataset
from analyst.session.display import csv_bytes
from analyst.session.eviction import register
from analyst.session.limiter import RateLimiter

log = get_logger(__name__)


@dataclass
class ChatEntry:
    """One question as the conversation shows it.

    ``response`` is ``None`` when the question was not run, which today means the session
    was over its rate limit; ``notice`` then says why. ``download`` is the result as a CSV
    file, built the first time it is asked for and kept, so that drawing the conversation
    again does not rebuild every file.
    """

    question: str
    #: Not in ``repr``: the response holds the result's rows, and a repr ends up in logs
    #: and error messages.
    response: AnalystResponse | None = field(default=None, repr=False)
    notice: str = ""
    #: Whether the question was asked in schema-only mode, so the interface can say whether
    #: the conversation has already sent values from the file.
    schema_only: bool = False
    #: Things the interface built from the response (such as a drawn chart), kept so that
    #: drawing the conversation again is cheap. Never read by this package.
    drawn: dict[object, object] = field(default_factory=dict, init=False, repr=False)
    _download: bytes | None = field(default=None, init=False, repr=False)

    def download(self) -> bytes | None:
        """The result as a CSV file (FR-8.3), or ``None`` when there is no result."""
        if self.response is None or self.response.result is None:
            return None
        if self._download is None:
            self._download = csv_bytes(self.response.result.dataframe)
        return self._download


@dataclass(frozen=True)
class LoadFailure:
    """Why a file could not be loaded, and which choice of file and settings it was."""

    key: object
    message: str


def _default_client_factory(settings: Settings) -> Callable[[], LLMClient]:
    return lambda: OpenAIClient(settings)


class AnalystSession:
    """The state of one browser session."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client_factory: Callable[[], LLMClient] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty session.

        Args:
            settings: Limits and region; the application settings by default.
            client_factory: Makes the model client for a dataset; OpenAI's by default and a
                scripted fake in tests.
            clock: Monotonic seconds, for the rate limit and the idle timer.
        """
        self.settings = settings or get_settings()
        self._client_factory = client_factory or _default_client_factory(self.settings)
        #: Short random label for log lines. It is not derived from anything about the user.
        self.id = secrets.token_hex(4)
        self.db = SessionDatabase(self.settings, clock=clock)
        self.limiter = RateLimiter(
            self.settings.rate_limit_questions, self.settings.rate_limit_window_s, clock=clock
        )
        self.memory = ConversationMemory()
        self.history: list[ChatEntry] = []
        self.dataset: LoadedDataset | None = None
        #: Whatever the interface used to tell one choice of file and settings from another.
        self.source_key: object | None = None
        self.load_failure: LoadFailure | None = None
        self.schema_only = False
        #: True after the data was removed for being idle, until a file is loaded again.
        self.expired = False
        self._analyst: Analyst | None = None
        register(self)

    @property
    def has_dataset(self) -> bool:
        """True when a file is loaded and questions can be asked."""
        return self._analyst is not None

    # -- the dataset ---------------------------------------------------------------------------

    def load(
        self, data: bytes, filename: str, options: ReadOptions | None = None, *, key: object = None
    ) -> bool:
        """Load a file, replacing the dataset and clearing the conversation.

        Returns ``True`` when it loaded. On failure nothing is raised: ``load_failure`` says
        why, and the dataset that was loaded before stays if the file could not even be read.
        ``key`` is stored with the result either way, so the interface can tell a file it has
        already tried from a new choice.
        """
        try:
            dataset = load_dataset(self.db, data, filename, options, settings=self.settings)
        except IngestionError as error:
            message = error.user_message
            log.info("file_rejected", extra={"session": self.id, "category": error.category})
        except Exception as error:  # a bug must not reach the user as a stack trace
            message = IngestionError.fallback_message
            log.error("file_failed", extra={"session": self.id, "error_type": type(error).__name__})
        else:
            self._install(dataset, key)
            return True
        if not self.db.is_loaded:
            self._forget_dataset()  # the table went with the failed load; what used it must too
        self.load_failure = LoadFailure(key, message)
        return False

    def _install(self, dataset: LoadedDataset, key: object) -> None:
        self.memory = ConversationMemory()
        self.history = []
        self.dataset = dataset
        self.source_key = key
        self.load_failure = None
        self.expired = False
        self._analyst = Analyst(
            self.db,
            dataset.profile,
            self._client_factory(),
            memory=self.memory,
            settings=self.settings,
            schema_only=self.schema_only,
        )

    def _forget_dataset(self) -> None:
        self.memory = ConversationMemory()
        self.history = []
        self.dataset = None
        self.source_key = None
        self._analyst = None

    def clear(self) -> None:
        """Remove the data and the conversation. The rate limit stays."""
        self.db.close()
        self._forget_dataset()

    # -- the conversation ----------------------------------------------------------------------

    def ask(self, question: str, on_progress: ProgressCallback | None = None) -> ChatEntry:
        """Ask a question about the loaded dataset and add it to the conversation.

        Raises:
            RuntimeError: No dataset is loaded.
        """
        analyst = self._analyst
        if analyst is None:
            raise RuntimeError("No dataset is loaded in this session")
        try:
            self.limiter.acquire()
        except RateLimitedError as error:
            log.info("rate_limited", extra={"session": self.id})
            entry = ChatEntry(
                question=question.strip(), notice=error.user_message, schema_only=self.schema_only
            )
        else:
            analyst.schema_only = self.schema_only
            response = analyst.ask(question, on_progress)
            entry = ChatEntry(
                question=question.strip(), response=response, schema_only=self.schema_only
            )
            log.info("question_asked", extra={"session": self.id, "status": response.status})
        self.history.append(entry)
        return entry

    def reset_conversation(self) -> None:
        """Start a new conversation about the same dataset (FR-7.3)."""
        self.memory.reset()
        self.history = []

    @property
    def values_may_have_been_sent(self) -> bool:
        """True when a question was run with schema-only mode off in this conversation.

        Its text, SQL or assumptions can quote values from the file, and they stay in the
        history the model is shown, whatever the mode is now (security.md, section 3).
        """
        return any(e.response is not None and not e.schema_only for e in self.history)

    # -- time ----------------------------------------------------------------------------------

    def touch(self) -> None:
        """Record that the user is here, which resets the idle timer."""
        self.db.touch()

    def expire_if_idle(self) -> bool:
        """Remove the data and conversation if the session has been idle too long.

        Returns ``True`` when it did. Called when the user comes back and, for sessions whose
        tab is left open, by the sweeper (``analyst.session.eviction``).
        """
        if not self.has_dataset or not self.db.is_expired():
            return False
        self.clear()
        self.expired = True
        log.info("session_expired", extra={"session": self.id})
        return True
