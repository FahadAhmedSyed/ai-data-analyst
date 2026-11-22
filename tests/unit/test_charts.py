"""Tests for the chart builder (T1.15): what ends up in the figure, in both modes."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from analyst.core.models import ChartSpec, ChartType, QueryResult
from analyst.visualization import build_figure
from analyst.visualization.charts import (
    ASSUMED_PLOT_WIDTH,
    BAR_PIXELS,
    MAX_LABEL_CHARS,
    plot_text,
)
from analyst.visualization.theme import DARK, LIGHT, Theme

MONTHS = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])


def result(**columns: object) -> QueryResult:
    frame = pd.DataFrame(columns)
    return QueryResult(dataframe=frame, row_count=len(frame))


def spec(
    kind: ChartType, x: str | None = None, y: list[str] | None = None, color: str | None = None
) -> ChartSpec:
    return ChartSpec(type=kind, x=x, y=y or [], color=color)


def regions(count: int, **columns: object) -> QueryResult:
    return result(region=[f"region {i:02d}" for i in range(count)], **columns)


# --- refusing what cannot be drawn ---------------------------------------------------------


def test_a_none_spec_raises() -> None:
    with pytest.raises(ValueError, match="no chart to draw"):
        build_figure(result(a=[1, 2]), spec("none"))


def test_a_spec_that_does_not_fit_raises_and_says_why() -> None:
    frame = result(region=["n", "s"], revenue=[1, 2])
    with pytest.raises(ValueError, match="no such column: sales"):
        build_figure(frame, spec("bar", "region", ["sales"]))


def test_building_does_not_change_the_result() -> None:
    frame = result(month=MONTHS[::-1], revenue=[4.0, None, 2.0, 1.0])
    before = frame.dataframe.copy()
    build_figure(frame, spec("line", "month", ["revenue"]))
    pd.testing.assert_frame_equal(frame.dataframe, before)


# --- line charts ---------------------------------------------------------------------------


def test_a_line_is_drawn_two_pixels_wide_with_the_first_slot_colour() -> None:
    fig = build_figure(
        result(month=MONTHS, revenue=[1, 2, 3, 4]), spec("line", "month", ["revenue"])
    )
    (trace,) = fig.data
    assert trace.type == "scatter"
    assert trace.mode == "lines+markers"
    assert trace.line.width == 2
    assert trace.line.color == LIGHT.series[0]
    assert list(trace.x) == ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    assert list(trace.y) == [1, 2, 3, 4]
    assert fig.layout.xaxis.type == "date"


def test_a_single_series_has_no_legend_and_no_end_label() -> None:
    fig = build_figure(
        result(month=MONTHS, revenue=[1, 2, 3, 4]), spec("line", "month", ["revenue"])
    )
    assert fig.layout.showlegend is False
    assert fig.layout.annotations == ()


def test_rows_are_drawn_in_x_order() -> None:
    frame = result(month=MONTHS[[2, 0, 3, 1]], revenue=[3, 1, 4, 2])
    (trace,) = build_figure(frame, spec("line", "month", ["revenue"])).data
    assert list(trace.y) == [1, 2, 3, 4]


def test_timestamps_keep_their_time_and_midnights_drop_it() -> None:
    hours = pd.to_datetime(["2024-01-01 10:30:00", "2024-01-01 11:30:00"])
    (trace,) = build_figure(result(t=hours, v=[1, 2]), spec("line", "t", ["v"])).data
    assert list(trace.x) == ["2024-01-01 10:30:00", "2024-01-01 11:30:00"]


def test_missing_values_leave_a_gap_rather_than_a_zero() -> None:
    frame = result(month=MONTHS, revenue=[1.0, None, 3.0, float("inf")])
    (trace,) = build_figure(frame, spec("line", "month", ["revenue"])).data
    assert list(trace.y) == [1.0, None, 3.0, None]
    assert trace.connectgaps is False
    assert "NaN" not in build_figure(frame, spec("line", "month", ["revenue"])).to_json()


def test_a_numeric_x_axis_is_linear() -> None:
    fig = build_figure(result(day=[1, 2, 3], visits=[5, 6, 7]), spec("line", "day", ["visits"]))
    assert fig.layout.xaxis.type == "linear"


def test_a_long_line_marks_only_its_last_point() -> None:
    n = 30
    frame = result(day=range(n), visits=range(n))
    (trace,) = build_figure(frame, spec("line", "day", ["visits"])).data
    assert list(trace.marker.size) == [0] * (n - 1) + [8]


def test_a_short_line_marks_every_point() -> None:
    (trace,) = build_figure(
        result(day=[1, 2, 3], visits=[5, 6, 7]), spec("line", "day", ["visits"])
    ).data
    assert list(trace.marker.size) == [8, 8, 8]
    assert trace.marker.line.width == 2
    assert trace.marker.line.color == LIGHT.surface


def test_two_lines_get_slots_in_order_a_legend_and_end_labels() -> None:
    frame = result(month=MONTHS, a=[10, 20, 30, 40], b=[12, 14, 16, 18])
    fig = build_figure(frame, spec("line", "month", ["a", "b"]))
    assert [t.line.color for t in fig.data] == [LIGHT.series[0], LIGHT.series[1]]
    assert fig.layout.showlegend is True
    assert sorted(a.text for a in fig.layout.annotations) == ["a", "b"]
    assert fig.layout.margin.r == 120


def test_end_labels_are_dropped_when_the_line_ends_would_collide() -> None:
    frame = result(month=MONTHS, a=[10, 20, 30, 40], b=[12, 14, 16, 40.5])
    fig = build_figure(frame, spec("line", "month", ["a", "b"]))
    assert fig.layout.annotations == ()
    assert fig.layout.showlegend is True


def test_end_labels_are_dropped_when_a_name_is_long() -> None:
    long_name = "a" * 25
    frame = result(month=MONTHS, **{long_name: [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    fig = build_figure(frame, spec("line", "month", [long_name, "b"]))
    assert fig.layout.annotations == ()


def test_end_labels_are_dropped_beyond_four_lines() -> None:
    columns = {name: [1.0, 2.0, 3.0, 4.0 + i * 10] for i, name in enumerate("abcde")}
    fig = build_figure(result(month=MONTHS, **columns), spec("line", "month", list("abcde")))
    assert fig.layout.annotations == ()
    assert len(fig.data) == 5
    assert fig.layout.showlegend is True


def test_end_labels_are_dropped_when_every_value_is_the_same() -> None:
    frame = result(month=MONTHS, a=[5, 5, 5, 5], b=[5, 5, 5, 5])
    assert build_figure(frame, spec("line", "month", ["a", "b"])).layout.annotations == ()


def test_an_end_label_sits_on_the_last_value_that_exists() -> None:
    frame = result(month=MONTHS, a=[1.0, 2.0, 3.0, None], b=[9.0, 8.0, 7.0, 6.0])
    fig = build_figure(frame, spec("line", "month", ["a", "b"]))
    by_text = {a.text: a for a in fig.layout.annotations}
    assert by_text["a"].x == "2024-03-01"
    assert by_text["a"].y == 3.0
    assert by_text["b"].x == "2024-04-01"


def test_a_group_with_no_values_gets_no_end_labels() -> None:
    frame = result(
        month=[*MONTHS[:2], *MONTHS[:2]],
        team=["a", "a", "b", "b"],
        score=[1.0, 2.0, None, None],
    )
    fig = build_figure(frame, spec("line", "month", ["score"], "team"))
    assert [t.name for t in fig.data] == ["a", "b"]
    assert fig.layout.annotations == ()


# --- measures of different sizes -----------------------------------------------------------


def test_measures_of_very_different_size_get_their_own_panels_not_a_second_axis() -> None:
    frame = result(month=MONTHS, orders=[10, 12, 11, 13], revenue=[50_000, 61_000, 58_000, 70_000])
    fig = build_figure(frame, spec("line", "month", ["orders", "revenue"]))
    assert len(fig.data) == 2
    assert fig.data[0].yaxis != fig.data[1].yaxis
    assert fig.layout.yaxis.overlaying is None
    assert fig.layout.yaxis2.overlaying is None
    assert fig.layout.yaxis.title.text == "orders"
    assert fig.layout.yaxis2.title.text == "revenue"
    assert fig.layout.showlegend is False
    assert fig.layout.height == 540


def test_separate_panels_share_the_x_axis_and_keep_slot_colours() -> None:
    frame = result(month=MONTHS, orders=[10, 12, 11, 13], revenue=[50_000, 61_000, 58_000, 70_000])
    fig = build_figure(frame, spec("line", "month", ["orders", "revenue"]))
    assert fig.layout.xaxis.matches == "x2"  # the top panel follows the bottom one
    assert fig.layout.xaxis2.title.text == "month"
    assert fig.layout.xaxis.title.text is None
    assert [t.line.color for t in fig.data] == [LIGHT.series[0], LIGHT.series[1]]


def test_measures_of_a_similar_size_share_one_axis() -> None:
    frame = result(month=MONTHS, a=[10, 12, 11, 13], b=[50, 61, 58, 70])
    fig = build_figure(frame, spec("line", "month", ["a", "b"]))
    assert fig.data[0].yaxis == fig.data[1].yaxis
    assert fig.layout.height == 380


def test_the_size_limit_is_a_factor_of_ten_between_medians() -> None:
    def axes(big: int) -> tuple[str, str]:
        frame = result(month=MONTHS, a=[1, 1, 1, 1], b=[big] * 4)
        fig = build_figure(frame, spec("line", "month", ["a", "b"]))
        return fig.data[0].yaxis, fig.data[1].yaxis

    same = axes(10)
    assert same[0] == same[1]
    split = axes(11)
    assert split[0] != split[1]


def test_zero_and_missing_cells_do_not_count_as_a_size() -> None:
    frame = result(month=MONTHS, a=[0, 0, 0, 0], b=[5, 6, 7, 8])
    fig = build_figure(frame, spec("line", "month", ["a", "b"]))
    assert fig.data[0].yaxis == fig.data[1].yaxis


def test_bars_of_very_different_size_get_panels_too() -> None:
    frame = regions(3, small=[1, 2, 3], big=[1000, 2000, 3000])
    fig = build_figure(frame, spec("bar", "region", ["small", "big"]))
    assert fig.data[0].yaxis != fig.data[1].yaxis
    assert fig.layout.showlegend is False
    assert fig.layout.xaxis2.type == "category"


def test_an_explicit_height_wins_over_the_panel_height() -> None:
    frame = result(month=MONTHS, a=[1, 2, 3, 4], b=[1000, 2000, 3000, 4000])
    assert build_figure(frame, spec("line", "month", ["a", "b"]), height=700).layout.height == 700
    single = result(month=MONTHS, a=[1, 2, 3, 4])
    assert build_figure(single, spec("line", "month", ["a"]), height=500).layout.height == 500


@pytest.mark.parametrize(
    ("top", "tick_format"),
    [(9_999, ",~g"), (10_000, "~s"), (2_500_000, "~s"), (12, ",~g")],
)
def test_big_numbers_get_short_ticks_and_small_ones_stay_in_full(
    top: int, tick_format: str
) -> None:
    frame = result(month=MONTHS, v=[1, 2, 3, top])
    fig = build_figure(frame, spec("line", "month", ["v"]))
    assert fig.layout.yaxis.tickformat == tick_format


def test_every_panel_gets_the_tick_format_for_its_own_size() -> None:
    frame = result(month=MONTHS, small=[1, 2, 3, 4], big=[20_000, 30_000, 40_000, 50_000])
    fig = build_figure(frame, spec("line", "month", ["small", "big"]))
    assert fig.layout.yaxis.tickformat == ",~g"
    assert fig.layout.yaxis2.tickformat == "~s"


# --- colour by group -----------------------------------------------------------------------


def grouped_lines(order: list[str]) -> QueryResult:
    rows = [(m, g, i + 10 * k) for k, g in enumerate(order) for i, m in enumerate(MONTHS[:3])]
    frame = pd.DataFrame(rows, columns=["month", "team", "score"])
    return QueryResult(dataframe=frame, row_count=len(frame))


def test_groups_are_coloured_by_name_order_not_by_row_order() -> None:
    forward = build_figure(
        grouped_lines(["red", "blue", "green"]), spec("line", "month", ["score"], "team")
    )
    backward = build_figure(
        grouped_lines(["green", "blue", "red"]), spec("line", "month", ["score"], "team")
    )
    expected = {"blue": LIGHT.series[0], "green": LIGHT.series[1], "red": LIGHT.series[2]}
    for fig in (forward, backward):
        assert {t.name: t.line.color for t in fig.data} == expected
        assert [t.name for t in fig.data] == ["blue", "green", "red"]


def test_eight_groups_use_all_eight_slots_in_order() -> None:
    names = [f"g{i}" for i in range(8)]
    fig = build_figure(grouped_lines(names), spec("line", "month", ["score"], "team"))
    assert [t.line.color for t in fig.data] == list(LIGHT.series)
    assert fig.layout.showlegend is True
    assert fig.layout.annotations == ()  # more than four lines: the legend names them


def test_nine_groups_are_refused_rather_than_given_a_repeated_colour() -> None:
    names = [f"g{i}" for i in range(9)]
    with pytest.raises(ValueError, match="more than 8 groups"):
        build_figure(grouped_lines(names), spec("line", "month", ["score"], "team"))


def test_a_group_with_no_name_is_not_drawn() -> None:
    frame = result(
        month=[*MONTHS[:2], *MONTHS[:2]], team=["a", "a", None, None], score=[1, 2, 3, 4]
    )
    fig = build_figure(frame, spec("line", "month", ["score"], "team"))
    assert [t.name for t in fig.data] == ["a"]


def test_grouped_bars_line_up_on_the_same_categories() -> None:
    frame = result(region=["n", "s", "n"], team=["a", "a", "b"], revenue=[1.0, 2.0, 3.0])
    fig = build_figure(frame, spec("bar", "region", ["revenue"], "team"))
    a, b = fig.data
    assert list(a.x) == list(b.x) == ["n", "s"]
    assert list(a.y) == [1.0, 2.0]
    assert list(b.y) == [3.0, None]


# --- bar charts ----------------------------------------------------------------------------


def test_one_series_of_bars_is_one_colour_and_names_its_axis() -> None:
    fig = build_figure(regions(3, revenue=[5, 3, 4]), spec("bar", "region", ["revenue"]))
    (trace,) = fig.data
    assert trace.type == "bar"
    assert trace.marker.color == LIGHT.series[0]
    assert fig.layout.showlegend is False
    assert fig.layout.xaxis.title.text == "region"
    assert fig.layout.yaxis.title.text == "revenue"
    assert fig.layout.xaxis.type == "category"


def test_bars_keep_the_order_of_the_result() -> None:
    frame = result(k=["z", "a", "m"], v=[1, 3, 2])
    fig = build_figure(frame, spec("bar", "k", ["v"]))
    assert list(fig.data[0].x) == ["z", "a", "m"]
    assert list(fig.layout.xaxis.categoryarray) == ["z", "a", "m"]


def test_bars_are_rounded_at_the_end_and_separated_by_a_surface_gap() -> None:
    (trace,) = build_figure(regions(3, revenue=[5, 3, 4]), spec("bar", "region", ["revenue"])).data
    assert trace.marker.cornerradius == 4
    assert trace.marker.line.color == LIGHT.surface
    assert trace.marker.line.width == 2


def test_up_to_twelve_single_series_bars_show_their_values() -> None:
    fig = build_figure(regions(12, n=list(range(12))), spec("bar", "region", ["n"]))
    assert list(fig.data[0].text) == [str(i) for i in range(12)]
    assert fig.data[0].textposition == "outside"


def test_thirteen_bars_do_not_show_values() -> None:
    fig = build_figure(regions(13, n=list(range(13))), spec("bar", "region", ["n"]))
    assert fig.data[0].text is None


def test_bars_of_several_series_do_not_show_values() -> None:
    fig = build_figure(regions(3, a=[1, 2, 3], b=[2, 3, 4]), spec("bar", "region", ["a", "b"]))
    assert all(t.text is None for t in fig.data)
    assert fig.layout.showlegend is True
    assert fig.layout.barmode == "group"
    assert fig.layout.bargroupgap == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("values", "labels"),
    [
        ([1234, 5, 0], ["1,234", "5", "0"]),
        ([1234.5, 5.25, 0.0], ["1,234.50", "5.25", "0.00"]),
        ([0.1234, 0.5], ["0.1234", "0.5"]),
        ([1.0, None, 3.0], ["1", None, "3"]),
    ],
)
def test_bar_value_labels_use_the_same_format_as_the_hover_label(
    values: list[float | None], labels: list[str | None]
) -> None:
    frame = regions(len(values), n=values)
    assert list(build_figure(frame, spec("bar", "region", ["n"])).data[0].text) == labels


def test_bar_thickness_aims_for_24_pixels_and_stays_within_bounds() -> None:
    def gap(categories: int) -> float:
        frame = regions(categories, n=list(range(categories)))
        return build_figure(frame, spec("bar", "region", ["n"])).layout.bargap

    assert (BAR_PIXELS, ASSUMED_PLOT_WIDTH) == (24, 640)
    assert gap(10) == pytest.approx(0.625)  # 64px slot, 24px bar
    assert gap(2) == pytest.approx(0.92)  # a few bars stay slim instead of becoming slabs
    assert gap(30) == pytest.approx(0.4)  # many bars never touch


def test_a_missing_category_cell_is_left_out() -> None:
    frame = result(k=["a", None, "c"], v=[1, 2, 3])
    fig = build_figure(frame, spec("bar", "k", ["v"]))
    assert "None" not in json.dumps(list(fig.layout.xaxis.categoryarray))
    assert list(fig.layout.xaxis.categoryarray) == ["a", "c"]


def test_bars_over_dates_label_the_categories_as_dates() -> None:
    fig = build_figure(result(month=MONTHS, n=[1, 2, 3, 4]), spec("bar", "month", ["n"]))
    assert next(iter(fig.data[0].x)) == "2024-01-01"


# --- text from the file ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain", "plain"),
        ("R&D", "R&amp;D"),
        ("a < b > c", "a &lt; b &gt; c"),
        ('<a href="javascript:alert(1)">x</a>', '&lt;a href="javascript:alert(1)"&gt;x&lt;/a&gt;'),
        ("<b>bold</b>", "&lt;b&gt;bold&lt;/b&gt;"),
        ("two\nlines\tand tab", "two lines and tab"),
        ("100%{x}", "100% {x}"),
        ("&lt;", "&amp;lt;"),
    ],
)
def test_plot_text_cannot_become_markup(raw: str, expected: str) -> None:
    assert plot_text(raw) == expected


def test_plot_text_cuts_long_text() -> None:
    cut = plot_text("x" * 500)
    assert len(cut) <= MAX_LABEL_CHARS
    assert cut.startswith("xxx")


def test_no_markup_from_the_data_reaches_the_figure() -> None:
    nasty = '<img src=x onerror="alert(1)">'
    frame = result(**{nasty: ["<b>one</b>", "two", "R&D"], "total %{y}": [1, 2, 3]})
    fig = build_figure(frame, spec("bar", nasty, ["total %{y}"]))
    blob = fig.to_json()
    assert "<img" not in blob
    assert "<b>" not in blob
    assert "&lt;img" in blob
    assert "R&amp;D" in blob
    assert "%{y}" not in fig.data[0].name
    assert fig.data[0].hovertemplate.count("%{y") == 1  # only the template's own placeholder


def test_group_names_are_made_safe_too() -> None:
    frame = result(
        month=[*MONTHS[:2], *MONTHS[:2]],
        team=["<i>a</i>", "<i>a</i>", "b&c", "b&c"],
        v=[1, 2, 3, 4],
    )
    fig = build_figure(frame, spec("line", "month", ["v"], "team"))
    assert sorted(t.name for t in fig.data) == ["&lt;i&gt;a&lt;/i&gt;", "b&amp;c"]


# --- scatter and histogram -----------------------------------------------------------------


def test_a_scatter_plot_draws_points_only() -> None:
    frame = result(price=[1.0, 2.0, 3.0, 4.0], units=[9, 7, 4, 1])
    fig = build_figure(frame, spec("scatter", "price", ["units"]))
    (trace,) = fig.data
    assert trace.mode == "markers"
    assert trace.marker.size == 8
    assert trace.marker.color == LIGHT.series[0]
    assert fig.layout.showlegend is False
    assert fig.layout.xaxis.title.text == "price"
    assert fig.layout.yaxis.title.text == "units"
    assert list(trace.x) == [1.0, 2.0, 3.0, 4.0]


def test_a_scatter_plot_by_group_uses_one_trace_per_group() -> None:
    frame = result(a=[1, 2, 3, 4], b=[4, 3, 2, 1], team=["y", "y", "x", "x"])
    fig = build_figure(frame, spec("scatter", "a", ["b"], "team"))
    assert [t.name for t in fig.data] == ["x", "y"]
    assert [t.marker.color for t in fig.data] == list(LIGHT.series[:2])
    assert fig.layout.showlegend is True


def test_a_histogram_counts_rows_and_labels_the_axis() -> None:
    fig = build_figure(result(age=[float(i) for i in range(20)]), spec("histogram", "age"))
    (trace,) = fig.data
    assert trace.type == "histogram"
    assert len(trace.x) == 20
    assert fig.layout.yaxis.title.text == "Rows"
    assert fig.layout.xaxis.title.text == "age"
    assert fig.layout.showlegend is False
    assert fig.layout.bargap == 0.02


def test_a_histogram_ignores_missing_values() -> None:
    values = [1.0, None, 3.0, *[float(i) for i in range(9)]]
    (trace,) = build_figure(result(age=values), spec("histogram", "age")).data
    assert len(trace.x) == 11
    assert None not in list(trace.x)


# --- texture --------------------------------------------------------------------------------


def test_bars_are_solid_unless_texture_is_asked_for() -> None:
    fig = build_figure(regions(3, a=[1, 2, 3], b=[2, 3, 4]), spec("bar", "region", ["a", "b"]))
    assert all(t.marker.pattern.shape is None for t in fig.data)


def test_texture_hatches_neighbouring_series_in_opposite_directions() -> None:
    fig = build_figure(
        regions(3, a=[1, 2, 3], b=[2, 3, 4], c=[3, 4, 5]),
        spec("bar", "region", ["a", "b", "c"]),
        texture=True,
    )
    assert [t.marker.pattern.shape for t in fig.data] == ["/", "\\", "/"]
    assert fig.data[0].marker.pattern.fgcolor == LIGHT.series[0]
    assert fig.data[0].marker.pattern.bgcolor == LIGHT.surface
    assert 0.2 <= fig.data[0].marker.pattern.solidity <= 0.6


def test_texture_applies_to_panels_and_histograms() -> None:
    panels = build_figure(
        regions(3, a=[1, 2, 3], b=[1000, 2000, 3000]),
        spec("bar", "region", ["a", "b"]),
        texture=True,
    )
    assert all(t.marker.pattern.shape == "/" for t in panels.data)
    hist = build_figure(result(v=range(12)), spec("histogram", "v"), texture=True)
    assert hist.data[0].marker.pattern.shape == "/"
    assert (
        build_figure(result(v=range(12)), spec("histogram", "v")).data[0].marker.pattern.shape
        is None
    )


# --- dark mode ------------------------------------------------------------------------------


def test_dark_mode_uses_the_dark_surface_ink_and_slots() -> None:
    frame = result(month=MONTHS, a=[10, 20, 30, 40], b=[12, 14, 16, 18])
    fig = build_figure(frame, spec("line", "month", ["a", "b"]), dark=True)
    assert fig.layout.paper_bgcolor == DARK.surface
    assert fig.layout.plot_bgcolor == DARK.surface
    assert fig.layout.yaxis.gridcolor == DARK.grid
    assert [t.line.color for t in fig.data] == [DARK.series[0], DARK.series[1]]
    assert fig.data[0].marker.line.color == DARK.surface


def test_light_mode_is_the_default() -> None:
    fig = build_figure(regions(2, n=[1, 2]), spec("bar", "region", ["n"]))
    assert fig.layout.paper_bgcolor == LIGHT.surface


def test_dark_mode_applies_to_every_chart_type() -> None:
    cases = [
        (regions(3, n=[1, 2, 3]), spec("bar", "region", ["n"])),
        (result(a=[1, 2, 3], b=[3, 2, 1]), spec("scatter", "a", ["b"])),
        (result(v=range(12)), spec("histogram", "v")),
        (
            result(month=MONTHS, a=[1, 2, 3, 4], b=[1000, 2000, 3000, 4000]),
            spec("line", "month", ["a", "b"]),
        ),
    ]
    for frame, chart in cases:
        assert build_figure(frame, chart, dark=True).layout.paper_bgcolor == DARK.surface


# --- the theme ------------------------------------------------------------------------------


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_each_theme_has_eight_distinct_slots(theme: Theme) -> None:
    assert len(theme.series) == 8
    assert len(set(theme.series)) == 8
    for colour in (*theme.series, theme.surface, theme.ink, theme.grid, theme.baseline):
        assert len(colour) == 7
        assert colour.startswith("#")
        int(colour[1:], 16)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_text_is_readable_on_the_surface(theme: Theme) -> None:
    assert contrast(theme.ink, theme.surface) >= 7
    assert contrast(theme.ink_secondary, theme.surface) >= 4.5
    assert contrast(theme.muted, theme.surface) >= 3


def test_dark_series_all_have_enough_contrast_on_the_dark_surface() -> None:
    assert all(contrast(c, DARK.surface) >= 3 for c in DARK.series)


def test_the_grid_is_quieter_than_the_baseline() -> None:
    for theme in (LIGHT, DARK):
        assert contrast(theme.grid, theme.surface) < contrast(theme.baseline, theme.surface)
