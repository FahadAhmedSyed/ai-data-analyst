"""Text from a file or from the model is never turned into markup (security.md, T12).

The checks look at what the page tells the browser to show: every markdown, caption and
message element, and the flag that would let it carry HTML.
"""

from __future__ import annotations

import re

from analyst.core.models import FilterCondition
from analyst.session.display import escape_markdown
from tests.e2e.driver import Page
from tests.unit.test_analyst import explains
from tests.unit.test_context import INTENT
from tests.unit.test_generator import answer, asks, refuses

IMAGE = "![x](http://evil.example/c.png)"
LINK = "[click](http://evil.example/v)"
HTML = "<img src=x onerror=alert(1)>"
FILE_NAME = "<b>sales</b>.csv"
HOSTILE_CSV = (
    f"Region,{IMAGE},Note\n{HTML},1,:red[loud]\n{LINK},,$$x$$\n{LINK},3,<b>bold</b>\n"
).encode()

KINDS = ("markdown", "caption", "info", "warning", "error", "success")


def shown(page: Page) -> list[str]:
    """Everything the page draws as text, from the main area, the sidebar and every panel."""
    return page.texts(*KINDS)


def literal(text: str) -> str:
    """What a browser would show of markdown text: escaped characters count as plain text."""
    return re.sub(r"\\.", "", text)


def assert_no_markup_from_outside(page: Page) -> None:
    for text in shown(page):
        plain = literal(text)  # what is left is markdown that could do something
        for forming_markup in ("![", "](", "<", "://", "evil.example"):
            assert forming_markup not in plain, text
        # Streamlit linkifies a bare address after the escapes are read (seen in a browser)
        read = re.sub(r"\\(.)", r"\1", text)
        assert not re.search(r"(?i)\w+://|\bwww\.|@[\w-]", read), text


def assert_html_never_allowed(page: Page) -> None:
    for kind in ("markdown", "caption"):
        for element in getattr(page.at, kind):
            assert element.proto.allow_html is False


# -- the file --------------------------------------------------------------------------------------


def test_a_hostile_file_name_column_name_and_value_are_shown_as_text() -> None:
    page = Page().open_file(FILE_NAME, HOSTILE_CSV)
    assert page.session.has_dataset
    assert_no_markup_from_outside(page)
    assert_html_never_allowed(page)
    sidebar = [m.value for m in page.at.sidebar.markdown]
    assert f"**{escape_markdown(FILE_NAME)}**" in sidebar


def test_the_quality_notes_name_a_hostile_column_literally() -> None:
    page = Page().open_file(FILE_NAME, HOSTILE_CSV)
    notes = [t for t in page.texts("markdown") if t.startswith("- ") and "missing" in t]
    assert any(escape_markdown(IMAGE) in note for note in notes)


def test_the_tables_hold_the_hostile_text_as_data_not_as_markup() -> None:
    page = Page().open_file(FILE_NAME, HOSTILE_CSV)
    preview = page.at.dataframe[0].value
    assert IMAGE in list(preview.columns)
    assert HTML in preview["Region"].tolist()
    assert all(element.proto.allow_html is False for element in page.at.markdown)


# -- the model -------------------------------------------------------------------------------------


def test_links_images_and_addresses_in_what_the_model_says_do_not_reach_the_page() -> None:
    hostile = (
        f"Revenue is shown. {IMAGE} {LINK} <http://evil.example/auto> "
        "http://evil.example/bare?d=secret"
    )
    intent = INTENT.model_copy(
        update={"filters": [FilterCondition(column="region", operator="=", values=[IMAGE])]}
    )
    generated = answer(
        resolved_question=f"Revenue {LINK}",
        assumptions=[f"Assumed {IMAGE}", HTML],
        intent=intent,
    )
    page = Page([("generate", generated), ("explain", explains(hostile))]).open_file()
    page.ask("Total revenue by region?")
    assert not page.at.exception
    assert page.answer().markdown
    assert_no_markup_from_outside(page)
    assert_html_never_allowed(page)


def test_a_clarifying_question_with_links_and_images_is_shown_without_them() -> None:
    page = Page([("generate", asks(f"Do you mean {IMAGE} or {LINK}?"))]).open_file()
    page.ask("What is best?")
    (info,) = [i.value for i in page.answer().info]
    assert "evil.example" not in info
    assert_no_markup_from_outside(page)


def test_a_refusal_that_carries_markup_is_shown_without_it() -> None:
    page = Page([("generate", refuses(f"I cannot change data. {IMAGE} {LINK}"))]).open_file()
    page.ask("Delete the table")
    assert_no_markup_from_outside(page)
    assert_html_never_allowed(page)


def test_what_the_user_types_is_shown_as_text_too() -> None:
    page = Page([("generate", asks("Which?"))]).open_file()
    page.ask(f"{IMAGE} {HTML} $5 and $10 :red[x]")
    (typed,) = page.questions()
    assert typed == escape_markdown(f"{IMAGE} {HTML} $5 and $10 :red[x]")
    assert_no_markup_from_outside(page)


# -- the page does not rely on the layers before it ------------------------------------------------


def test_the_page_escapes_what_it_shows_even_if_earlier_layers_let_markup_through() -> None:
    """Defence in depth: the response is rewritten as if the pipeline had cleaned nothing."""
    intent = INTENT.model_copy(
        update={"filters": [FilterCondition(column="region", operator="=", values=[IMAGE])]}
    )
    page = Page([("generate", answer()), ("explain", explains("Revenue is shown."))]).open_file()
    page.ask("Total revenue by region?")
    entry = page.session.history[0]
    response = entry.response
    assert response is not None
    entry.response = response.model_copy(
        update={
            "resolved_question": f"Revenue {LINK} {IMAGE}",
            "assumptions": (IMAGE, HTML),
            "caveats": (LINK, HTML),
            "turn": response.turn.model_copy(update={"intent": intent}),
        }
    )
    entry.question = f"{IMAGE} {HTML}"
    page.at.run()
    assert not page.at.exception
    assert_no_markup_from_outside(page)
    assert_html_never_allowed(page)
    assert escape_markdown(f"Revenue {LINK} {IMAGE}") in " ".join(page.texts("caption"))
