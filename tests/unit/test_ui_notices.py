"""The words the interface says on its own account (FR-10.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from analyst.ui.notices import WELCOME, privacy_notice

PRIVACY_DOC = Path(__file__).resolve().parents[2] / "docs" / "privacy.md"


def doc_notice() -> str:
    """The blockquote under 'Summary for users' in docs/privacy.md, as one paragraph."""
    lines = PRIVACY_DOC.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## 1. Summary for users"))
    quote = []
    for line in lines[start:]:
        if line.startswith(">"):
            quote.append(line.removeprefix(">").strip())
        elif quote:
            break
    return " ".join(quote)


def test_the_notice_shown_is_the_notice_in_the_privacy_document() -> None:
    assert privacy_notice(60) == doc_notice()


def test_the_idle_time_in_the_notice_is_the_one_configured() -> None:
    assert "after 15 minutes of inactivity" in privacy_notice(15)


@pytest.mark.parametrize(
    "promise",
    [
        "deleted when you close the tab",
        "column names and types",
        "three sample rows",
        "your question",
        "rows of each result",
        "Schema-only mode",
        "results are still sent",
        "does not use API data to train",
    ],
)
def test_the_notice_says_what_is_sent_and_what_schema_only_mode_does_not_stop(promise: str) -> None:
    assert promise in privacy_notice(60)


def test_the_welcome_points_to_the_sidebar_and_promises_visible_sql() -> None:
    assert "sidebar" in WELCOME
    assert "see the SQL" in WELCOME
