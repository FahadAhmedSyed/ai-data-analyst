"""A way to use the app from a test: upload, load and ask, with a scripted model.

Each dataset the app loads gets the next script from the ones given, so a test says exactly
what the model will answer, and ``clients`` shows what it was asked.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

from analyst.llm.fake import FakeLLMClient
from analyst.session.state import AnalystSession
from analyst.ui.session_store import SESSION_KEY

SALES = (
    b"Order ID,Region,Revenue,Order Date\n"
    b"1,North,10.5,2024-01-01\n2,South,20,2024-01-02\n3,North,30,2024-01-03\n4,East,40,2024-01-04\n"
)
OTHER = b"Name,Score\nAnn,5\nBo,7\n"


def app(client_factory) -> None:
    """The script the tests run: the real page, with the model replaced."""
    from analyst.ui.app import run

    run(client_factory=client_factory)


class Page:
    """One browser session running the app."""

    def __init__(self, *scripts, session: AnalystSession | None = None) -> None:
        queue = list(scripts)
        self.clients: list[FakeLLMClient] = []

        def factory() -> FakeLLMClient:
            client = FakeLLMClient(queue.pop(0) if queue else [])
            self.clients.append(client)
            return client

        self.at = AppTest.from_function(app, kwargs={"client_factory": factory}, default_timeout=60)
        if session is not None:
            self.at.session_state[SESSION_KEY] = session
        self.at.run()
        assert not self.at.exception

    # -- doing things -----------------------------------------------------------------------

    def upload(self, name: str, data: bytes) -> Page:
        self.at.file_uploader[0].set_value((name, data, "application/octet-stream")).run()
        return self

    def load(self) -> Page:
        self.at.button(key="load_file").click().run()
        return self

    def open_file(self, name: str = "sales.csv", data: bytes = SALES) -> Page:
        return self.upload(name, data).load()

    def ask(self, question: str) -> Page:
        self.at.chat_input[0].set_value(question).run()
        return self

    # -- looking ----------------------------------------------------------------------------

    @property
    def session(self) -> AnalystSession:
        return self.at.session_state[SESSION_KEY]

    def texts(self, *kinds: str) -> list[str]:
        """The text of every element of these kinds, anywhere on the page."""
        kinds = kinds or ("markdown", "caption", "info", "warning", "error", "success")
        return [element.value for kind in kinds for element in getattr(self.at, kind)]

    def page_text(self) -> str:
        return "\n".join(self.texts())

    def answer(self, index: int = -1) -> Block:
        """The assistant's message at ``index`` among the assistant messages."""
        return [m for m in self.at.chat_message if m.name == "assistant"][index]

    def questions(self) -> list[str]:
        return [m.markdown[0].value for m in self.at.chat_message if m.name == "user"]
