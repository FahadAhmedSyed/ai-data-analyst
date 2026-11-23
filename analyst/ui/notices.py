"""The words the interface says on its own account: the privacy notice and the welcome."""

from __future__ import annotations


def privacy_notice(idle_minutes: int) -> str:
    """The notice shown on upload (FR-10.3). ``docs/privacy.md`` section 1 holds the same text,
    and a test keeps the two equal."""
    return (
        "Your file stays in this browser session's memory and is deleted when you close the "
        f"tab or after {idle_minutes} minutes of inactivity. To answer questions, the app sends "
        "OpenAI the column names and types, summary statistics, a few common values per column, "
        "three sample rows, and your question, and the rows of each result so that they can be "
        "explained. Schema-only mode leaves out the sample rows and common values, but results "
        "are still sent. OpenAI does not use API data to train its models by default."
    )


WELCOME = """\
Ask questions about a spreadsheet in plain English.

1. **Upload** a CSV or Excel file in the sidebar.
2. **Check** the preview, the columns and the data quality notes.
3. **Ask** a question, such as *"What was the average rainfall at each site in summer?"*

Every number in an answer comes from a query run on your data, and you can always see the SQL.
"""
