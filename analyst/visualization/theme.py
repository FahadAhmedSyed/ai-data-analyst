"""Chart colours and type (T1.15).

The categorical palette is the validated eight-slot default from the data-visualisation
method used for this project, checked with its validator in both modes against the
surfaces below:

* adjacent slots: worst colour-blind distance (OKLab x 100) 9.1 light and 8.4 dark, worst
  normal-vision distance 19.6 and 19.3 (floors 6 and 15);
* the first three slots hold up under all pairs (9.2 and 9.4; 24.0 and 20.9), which is why
  scatter plots use at most three groups;
* three light-mode slots (aqua, yellow, magenta) are under 3:1 contrast on the surface, so
  every chart with more than one series carries a legend and a hover label, and the result
  is always shown as a table too.

Slots are used in this order and never cycled: a ninth group is not drawn. Text never takes a
series colour; it uses the ink colours.
"""

from __future__ import annotations

from dataclasses import dataclass

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


@dataclass(frozen=True)
class Theme:
    """The colours of one mode."""

    surface: str
    ink: str  # primary text
    ink_secondary: str  # axis titles, legend, labels
    muted: str  # tick labels
    grid: str
    baseline: str
    series: tuple[str, ...]


LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    series=(
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua
        "#eda100",  # yellow
        "#e87ba4",  # magenta
        "#008300",  # green
        "#6250d6",  # violet
        "#e34948",  # red
    ),
)

DARK = Theme(
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    series=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#0ca30c",
        "#9085e9",
        "#e66767",
    ),
)
