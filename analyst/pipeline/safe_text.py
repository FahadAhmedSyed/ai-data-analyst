"""Make text from the model safe to show (threat T12).

The model's words are shown as Markdown, which Streamlit renders with images. An image
address that carries data (``![x](https://host/?d=<data>)``) would send that data out as soon as
the text was shown, and a link can lead a user somewhere they did not mean to go. Text
the model writes for the user (explanations, clarifying questions, assumptions) therefore
goes through ``sanitize_text`` before it is checked or shown.
"""

from __future__ import annotations

import re

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)|!\[([^\]]*)\]\[[^\]]*\]")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)|\[([^\]]+)\]\[[^\]]*\]")
_REFERENCE_DEFINITION = re.compile(r"^[ ]{0,3}\[[^\]]+\]:[ ]*\S.*$", re.MULTILINE)
_AUTOLINK = re.compile(r"<(?:https?://|ftp://|mailto:)[^>\s]*>", re.IGNORECASE)
_ADDRESS = re.compile(
    r"(?:https?://|ftp://|www\.)\S*[^\s.,;:!?)\]\"']|\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE
)
_LINK_REMOVED = "[link removed]"


def sanitize_text(text: str) -> str:
    """Take links, images and web addresses out of text from the model.

    Streamlit renders images in Markdown, so an image address carrying data would send that
    data out as soon as the answer is shown (threat T12). Links are reduced to their text.
    HTML is not rendered by the UI, so it is left as text.
    """
    text = "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())
    text = _REFERENCE_DEFINITION.sub("", text)
    text = _IMAGE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _LINK.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _AUTOLINK.sub(_LINK_REMOVED, text)
    text = _ADDRESS.sub(_LINK_REMOVED, text)
    return text.strip()
