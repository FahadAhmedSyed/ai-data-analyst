"""Draw a chart from a result and a spec that fits it (T1.15, FR-8.2).

``build_figure`` returns a Plotly figure. It follows a few fixed rules so that every chart
looks the same and reads the same way:

* **One axis.** Value columns whose sizes differ by more than ten times are not drawn on
  one axis (the second would look flat, or the pairing would suggest a link that is not
  there). They get one panel each, stacked, sharing the x axis.
* **Colour follows the thing, in fixed slot order.** Series are coloured from the palette in
  ``theme.py`` in order, never cycled. Groups are ordered by name, so the same set of groups
  gets the same colours each time. One series is one colour: a bar is not shaded by its size.
* **Thin, quiet marks.** Lines are 2px with 8px end dots, bars are rounded and slim, a 2px
  gap in the surface colour separates touching marks, and the grid is one hairline step off
  the surface.
* **A legend whenever there are two or more series**, none for one. With four series or
  fewer the line ends are also labelled, unless the labels would collide.
* **No numbers on every point.** Single-series bars with twelve bars or fewer show their
  values at the tip; everything else relies on the axis and the hover label.
* **Text from the file cannot become markup.** Plotly reads a small subset of HTML in labels
  (including links), so every name and category from the data is cut, flattened and has
  ``&``, ``<`` and ``>`` escaped (and ``%{`` broken up, which Plotly reads as a hover
  placeholder) before it reaches the figure.

Texture (opt-in, for print or colour-blind readers) draws bars with 45 and 135 degree
hatching in place of plain fills.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyst.core.models import ChartSpec, QueryResult
from analyst.ingestion.schema_card import clean_text
from analyst.visualization.recommender import column_kind, fit_problem
from analyst.visualization.theme import DARK, FONT_FAMILY, LIGHT, Theme

DEFAULT_HEIGHT = 380
PANEL_HEIGHT = 230
MAX_LABEL_CHARS = 60
MAX_END_LABEL_CHARS = 24
MAX_VALUE_LABELS = 12
SCALE_RATIO_LIMIT = 10.0
BAR_PIXELS = 24  # target bar thickness at a typical chart width
ASSUMED_PLOT_WIDTH = 640
MAX_FULL_MARKERS = 12
SI_FROM = 10_000

_PATTERNS = ("/", "\\")


def plot_text(value: object, limit: int = MAX_LABEL_CHARS) -> str:
    """Text from the data made safe for a Plotly label: one line, cut, and no markup."""
    text = clean_text(str(value), limit)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("%{", "% {")  # Plotly fills "%{x}" in hover text from the data


@dataclass(frozen=True)
class _Series:
    name: str
    x: list[Any]
    y: list[float | None]
    color: str


def _numbers(column: pd.Series) -> list[float | None]:
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return [None if not math.isfinite(v) else float(v) for v in values]


def _axis_values(column: pd.Series) -> list[Any]:
    """Values for an axis: dates as ISO text, numbers as floats, categories as safe text."""
    kind = column_kind(column)
    if kind == "date":
        stamps = pd.to_datetime(column)
        midnight = bool((stamps.dropna() == stamps.dropna().dt.normalize()).all())
        fmt = "%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S"
        return [None if pd.isna(s) else s.strftime(fmt) for s in stamps]
    if kind == "numeric":
        return _numbers(column)
    return [None if pd.isna(v) else plot_text(v) for v in column]


def _value_format(values: Sequence[float | None]) -> str:
    present = [v for v in values if v is not None]
    if present and all(float(v).is_integer() for v in present):
        return ",d"
    if present and max(abs(v) for v in present) < 1:
        return ".4g"
    return ",.2f"


def _label(value: float | None, fmt: str) -> str | None:
    """A value written for a bar tip, using the format its series hovers with."""
    if value is None:
        return None
    return format(int(value) if fmt == ",d" else value, fmt)


def _tick_format(values: Sequence[Sequence[float | None]]) -> str:
    biggest = max((abs(v) for series in values for v in series if v is not None), default=0.0)
    return "~s" if biggest >= SI_FROM else ",~g"


def _compatible(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    """True if the columns are of one size, so that one axis serves them all."""
    sizes = []
    for name in columns:
        numbers = [abs(v) for v in _numbers(frame[name]) if v]
        if numbers:
            sizes.append(float(np.median(numbers)))
    return not sizes or max(sizes) / min(sizes) <= SCALE_RATIO_LIMIT


def _series_for(frame: pd.DataFrame, spec: ChartSpec, theme: Theme) -> list[_Series]:
    assert spec.x is not None  # noqa: S101 - spec.type is not "none"
    if spec.type == "line":
        frame = frame.sort_values(spec.x, kind="stable")
    if spec.color is None:
        x = _axis_values(frame[spec.x])
        return [
            _Series(plot_text(y), x, _numbers(frame[y]), theme.series[i])
            for i, y in enumerate(spec.y)
        ]
    groups = sorted(frame[spec.color].dropna().unique(), key=str)
    out: list[_Series] = []
    categories = _axis_values(frame[spec.x])
    order = list(dict.fromkeys(c for c in categories if c is not None))
    for i, group in enumerate(groups):
        part = frame[frame[spec.color] == group]
        if spec.type == "bar":  # align every group to the same categories
            cells = dict(zip(_axis_values(part[spec.x]), _numbers(part[spec.y[0]]), strict=True))
            out.append(
                _Series(plot_text(group), order, [cells.get(c) for c in order], theme.series[i])
            )
        else:
            out.append(
                _Series(
                    plot_text(group),
                    _axis_values(part[spec.x]),
                    _numbers(part[spec.y[0]]),
                    theme.series[i],
                )
            )
    return out


def _base_layout(fig: go.Figure, theme: Theme, height: int, legend: bool) -> None:
    axis = {
        "linecolor": theme.baseline,
        "tickfont": {"color": theme.muted, "size": 12},
        "title": {"font": {"color": theme.ink_secondary, "size": 13}},
        "ticks": "",
        "automargin": True,
    }
    fig.update_layout(
        height=height,
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        font={"family": FONT_FAMILY, "color": theme.ink_secondary, "size": 13},
        margin={"l": 56, "r": 24, "t": 24, "b": 48},
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": theme.ink_secondary},
        },
        hoverlabel={
            "bgcolor": theme.surface,
            "bordercolor": theme.baseline,
            "font": {"color": theme.ink, "family": FONT_FAMILY},
        },
        separators=".,",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, **axis)
    fig.update_yaxes(
        showgrid=True, gridcolor=theme.grid, gridwidth=1, zeroline=True,
        zerolinecolor=theme.baseline, zerolinewidth=1, **axis,
    )  # fmt: skip


def _ring(theme: Theme) -> dict[str, Any]:
    return {"color": theme.surface, "width": 2}


def _line_trace(series: _Series, theme: Theme, value_format: str, label_x: bool) -> go.Scatter:
    count = len(series.y)
    sizes = [8] * count if count <= MAX_FULL_MARKERS else [0] * (count - 1) + [8]
    return go.Scatter(
        x=series.x,
        y=series.y,
        name=series.name,
        mode="lines+markers",
        line={"color": series.color, "width": 2, "shape": "linear"},
        marker={"color": series.color, "size": sizes, "line": _ring(theme)},
        connectgaps=False,
        hovertemplate=f"{series.name}: %{{y:{value_format}}}<extra></extra>",
        showlegend=True,
    )


def _end_labels_fit(series: Sequence[_Series]) -> bool:
    if not 2 <= len(series) <= 4:
        return False
    if any(len(s.name) > MAX_END_LABEL_CHARS for s in series):
        return False
    ends = []
    for s in series:
        last = next((v for v in reversed(s.y) if v is not None), None)
        if last is None:
            return False
        ends.append(last)
    everything = [v for s in series for v in s.y if v is not None]
    spread = max(everything) - min(everything)
    if spread == 0:
        return False
    ends.sort()
    return all((b - a) / spread >= 0.1 for a, b in itertools.pairwise(ends))


def _add_end_labels(fig: go.Figure, series: Sequence[_Series], theme: Theme) -> None:
    for s in series:
        index = max(i for i, v in enumerate(s.y) if v is not None)
        fig.add_annotation(
            x=s.x[index],
            y=s.y[index],
            text=s.name,
            showarrow=False,
            xanchor="left",
            xshift=10,
            font={"color": theme.ink_secondary, "size": 12},
        )
    fig.update_layout(margin_r=120)


def _bar_gap(categories: int, groups: int) -> float:
    """Gap between categories that keeps bars near ``BAR_PIXELS`` thick, between 0.4 and 0.92."""
    slot = ASSUMED_PLOT_WIDTH / max(categories, 1)
    share = min(0.6, max(0.08, BAR_PIXELS * groups / slot))
    return round(1 - share, 3)


def _bar_traces(series: Sequence[_Series], theme: Theme, texture: bool) -> list[go.Bar]:
    single = len(series) == 1
    traces = []
    for i, s in enumerate(series):
        fmt = _value_format(s.y)
        marker: dict[str, Any] = {
            "color": s.color,
            "line": _ring(theme),
            "cornerradius": 4,
        }
        if texture:
            marker["pattern"] = {
                "shape": _PATTERNS[i % 2],
                "fgcolor": s.color,
                "bgcolor": theme.surface,
                "size": 8,
                "solidity": 0.35,
            }
        labelled = single and len(s.x) <= MAX_VALUE_LABELS
        traces.append(
            go.Bar(
                x=s.x,
                y=s.y,
                name=s.name,
                marker=marker,
                text=[_label(v, fmt) for v in s.y] if labelled else None,
                textposition="outside" if labelled else None,
                textfont={"color": theme.ink_secondary, "size": 12},
                cliponaxis=False,
                hovertemplate=f"%{{x}}<br>{s.name}: %{{y:{fmt}}}<extra></extra>",
            )
        )
    return traces


def build_figure(
    result: QueryResult,
    spec: ChartSpec,
    *,
    dark: bool = False,
    texture: bool = False,
    height: int | None = None,
) -> go.Figure:
    """Draw ``result`` as ``spec`` says.

    Args:
        result: The executed result.
        spec: A chart that fits it, as returned by ``recommend_chart``.
        dark: Use the dark-mode colours.
        texture: Hatch bars and histograms (print, or readers who cannot tell the colours apart).
        height: Figure height in pixels; defaults to 380, or 230 per panel when the value
            columns need separate panels.

    Raises:
        ValueError: ``spec`` is ``none`` or does not fit ``result``.
    """
    if spec.type == "none":
        raise ValueError("there is no chart to draw")
    problem = fit_problem(result, spec)
    if problem is not None:
        raise ValueError(f"the chart does not fit the result: {problem}")
    theme = DARK if dark else LIGHT
    frame = result.dataframe
    assert spec.x is not None  # noqa: S101 - fit_problem checked it
    if spec.type == "histogram":
        return _histogram(frame, spec, theme, texture, height)
    if spec.type == "scatter":
        return _scatter(frame, spec, theme, height)
    separate = len(spec.y) > 1 and spec.color is None and not _compatible(frame, spec.y)
    series = _series_for(frame, spec, theme)
    if separate:
        return _panels(frame, spec, series, theme, texture, height)
    fig = go.Figure()
    values = [s.y for s in series]
    if spec.type == "line":
        fmt = _value_format([v for s in series for v in s.y])
        for s in series:
            fig.add_trace(_line_trace(s, theme, fmt, True))
        fig.update_layout(hovermode="x unified")
    else:
        for trace in _bar_traces(series, theme, texture):
            fig.add_trace(trace)
        categories = len(series[0].x)
        fig.update_layout(
            barmode="group",
            bargap=_bar_gap(categories, len(series)),
            bargroupgap=0.08,
            hovermode="closest",
        )
    _base_layout(fig, theme, height or DEFAULT_HEIGHT, legend=len(series) > 1)
    _titles(fig, spec, series)
    fig.update_yaxes(tickformat=_tick_format(values))
    _x_axis(fig, frame, spec)
    if spec.type == "line" and _end_labels_fit(series):
        _add_end_labels(fig, series, theme)
    return fig


def _titles(fig: go.Figure, spec: ChartSpec, series: Sequence[_Series]) -> None:
    assert spec.x is not None  # noqa: S101
    fig.update_xaxes(title_text=plot_text(spec.x))
    if len(spec.y) == 1:
        fig.update_yaxes(title_text=plot_text(spec.y[0]))


def _x_axis(fig: go.Figure, frame: pd.DataFrame, spec: ChartSpec) -> None:
    assert spec.x is not None  # noqa: S101
    kind = column_kind(frame[spec.x])
    if kind == "date" and spec.type == "line":
        fig.update_xaxes(type="date")
    elif spec.type == "bar":
        order = list(dict.fromkeys(c for c in _axis_values(frame[spec.x]) if c is not None))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
    elif kind == "numeric":
        fig.update_xaxes(type="linear", tickformat=_tick_format([_numbers(frame[spec.x])]))


def _panels(
    frame: pd.DataFrame,
    spec: ChartSpec,
    series: Sequence[_Series],
    theme: Theme,
    texture: bool,
    height: int | None,
) -> go.Figure:
    """One panel per value column, sharing the x axis (never two scales on one plot)."""
    rows = len(series)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for row, s in enumerate(series, start=1):
        fmt = _value_format(s.y)
        if spec.type == "line":
            fig.add_trace(_line_trace(s, theme, fmt, False), row=row, col=1)
        else:
            (trace,) = _bar_traces([s], theme, texture)
            fig.add_trace(trace, row=row, col=1)
        fig.update_yaxes(title_text=s.name, tickformat=_tick_format([s.y]), row=row, col=1)
    total = height or (PANEL_HEIGHT * rows + 80)
    _base_layout(fig, theme, total, legend=False)
    if spec.type == "bar":
        fig.update_layout(bargap=_bar_gap(len(series[0].x), 1), hovermode="closest")
    else:
        fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_font={"color": theme.ink_secondary, "size": 13})
    fig.update_xaxes(title_text=plot_text(spec.x), row=rows, col=1)
    kind = column_kind(frame[spec.x])
    if kind == "date" and spec.type == "line":
        fig.update_xaxes(type="date")
    elif spec.type == "bar":
        order = list(dict.fromkeys(c for c in _axis_values(frame[spec.x]) if c is not None))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
    return fig


def _scatter(frame: pd.DataFrame, spec: ChartSpec, theme: Theme, height: int | None) -> go.Figure:
    assert spec.x is not None  # noqa: S101
    y_name = spec.y[0]
    fig = go.Figure()
    if spec.color is None:
        groups = [("", frame)]
    else:
        names = sorted(frame[spec.color].dropna().unique(), key=str)
        groups = [(plot_text(g), frame[frame[spec.color] == g]) for g in names]
    all_x, all_y = [], []
    for i, (name, part) in enumerate(groups):
        x, y = _numbers(part[spec.x]), _numbers(part[y_name])
        all_x.append(x)
        all_y.append(y)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name or plot_text(y_name),
                mode="markers",
                marker={"color": theme.series[i], "size": 8, "line": _ring(theme)},
                hovertemplate=(
                    (f"{name}<br>" if name else "")
                    + f"{plot_text(spec.x)}: %{{x:{_value_format(x)}}}<br>"
                    + f"{plot_text(y_name)}: %{{y:{_value_format(y)}}}<extra></extra>"
                ),
            )
        )
    _base_layout(fig, theme, height or DEFAULT_HEIGHT, legend=len(groups) > 1)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text=plot_text(spec.x), tickformat=_tick_format(all_x), type="linear")
    fig.update_yaxes(title_text=plot_text(y_name), tickformat=_tick_format(all_y))
    return fig


def _histogram(
    frame: pd.DataFrame, spec: ChartSpec, theme: Theme, texture: bool, height: int | None
) -> go.Figure:
    assert spec.x is not None  # noqa: S101
    values = [v for v in _numbers(frame[spec.x]) if v is not None]
    marker: dict[str, Any] = {"color": theme.series[0], "line": _ring(theme)}
    if texture:
        marker["pattern"] = {
            "shape": "/",
            "fgcolor": theme.series[0],
            "bgcolor": theme.surface,
            "size": 8,
            "solidity": 0.35,
        }
    fig = go.Figure(
        go.Histogram(
            x=values,
            marker=marker,
            hovertemplate=f"{plot_text(spec.x)}: %{{x}}<br>Rows: %{{y:,d}}<extra></extra>",
        )
    )
    _base_layout(fig, theme, height or DEFAULT_HEIGHT, legend=False)
    fig.update_layout(bargap=0.02, hovermode="closest")
    fig.update_xaxes(title_text=plot_text(spec.x), tickformat=_tick_format([values]), type="linear")
    fig.update_yaxes(title_text="Rows", tickformat=",d")
    return fig
