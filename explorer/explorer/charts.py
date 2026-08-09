"""
The chart pieces, built once here and reused by every tab.

Colours come from CSS custom properties so light and dark are one
definition in assets/theme.css rather than a branch in every chart. The
two series hues are the first two slots of the validated palette; the
diverging pair for "which way is it wrong" is blue against red with a
neutral grey where the answer is nothing.

Marks are thin, grids are hairlines one shade off the surface, and every
chart carries a hover tooltip, since a chart on a page is something people
point at.
"""

import reflex as rx

SERIES_1 = "var(--series-1)"
SERIES_2 = "var(--series-2)"
GRID = "var(--grid)"
AXIS = "var(--axis)"
MUTED = "var(--muted)"
SURFACE = "var(--surface)"

POLE_LOW = "var(--pole-low)"
POLE_HIGH = "var(--pole-high)"
POLE_MID = "var(--pole-mid)"

TICK = {"fill": MUTED, "fontSize": 11}

# The fit works on a stretched accuracy scale, where equal steps near 100%
# mean equal steps in how hard they are to reach. Nobody reads a score
# that way, so the axis is turned back into the percentage it came from.
AS_ACCURACY = rx.Var(
    "((v) => (100 * (1 - Math.pow(10, -v))).toFixed(v >= 2 ? 2 : 1) + '%')"
)


def frame(*children, height=300, data=None, kind=None):
    """A chart body with the chrome every chart here shares."""
    maker = kind or rx.recharts.composed_chart

    return maker(
        rx.recharts.cartesian_grid(
            stroke=GRID, vertical=False, stroke_dasharray="0"
        ),
        *children,
        rx.recharts.graphing_tooltip(
            content_style={
                "background": SURFACE,
                "border": "1px solid var(--hairline)",
                "borderRadius": "8px",
                "fontSize": "12px",
                "color": "var(--ink)",
                "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            },
            item_style={"color": "var(--ink-2)"},
            label_style={"color": "var(--ink)", "fontWeight": 600},
            cursor={"stroke": AXIS, "strokeWidth": 1},
        ),
        data=data,
        height=height,
        width="100%",
        margin={"top": 8, "right": 12, "bottom": 4, "left": 0},
    )


def legend(*entries):
    """
    Identity spelled out in text beside a colour chip, so the reading
    never rests on hue alone.
    """
    return rx.hstack(
        *[
            rx.hstack(
                rx.box(
                    width="10px",
                    height="10px",
                    border_radius="2px",
                    background=colour,
                    flex_shrink="0",
                ),
                rx.text(label, class_name="text-xs"),
                spacing="2",
                align="center",
            )
            for label, colour in entries
        ],
        spacing="4",
        wrap="wrap",
        class_name="text-[var(--ink-2)] gap-y-1",
    )


def curve_chart(rows, marker, scale):
    """
    A map's expected performance across the skill range, the band one
    spread either side, and the scores actually set on it.

    The scores ride in the same table as the curve and are drawn as a line
    with no stroke, since a chart that mixes areas and lines will not also
    take a scatter.
    """
    return rx.vstack(
        legend(("expected, one spread either side", SERIES_1),
               ("scores set", SERIES_2)),
        frame(
            rx.recharts.x_axis(
                data_key="x",
                type_="number",
                domain=[-3, 3],
                ticks=[-3, -2, -1, 0, 1, 2, 3],
                allow_data_overflow=True,
                tick_line=False,
                axis_line=True,
                tick=TICK,
                label={
                    "value": "skill",
                    "position": "insideBottom",
                    "offset": -2,
                    "fill": MUTED,
                    "fontSize": 11,
                },
                height=38,
            ),
            rx.recharts.y_axis(
                type_="number",
                tick_line=False,
                axis_line=False,
                tick=TICK,
                width=58,
                domain=scale["domain"],
                ticks=scale["ticks"],
                custom_attrs={"tickFormatter": AS_ACCURACY},
                label={
                    "value": "accuracy",
                    "angle": -90,
                    "position": "insideLeft",
                    "fill": MUTED,
                    "fontSize": 11,
                },
            ),
            rx.recharts.area(
                data_key="band",
                stroke="none",
                fill=SERIES_1,
                fill_opacity=0.14,
                connect_nulls=True,
                is_animation_active=False,
                name="spread",
            ),
            rx.recharts.line(
                data_key="mean",
                stroke=SERIES_1,
                stroke_width=2,
                dot=False,
                connect_nulls=True,
                is_animation_active=False,
                name="expected",
            ),
            rx.recharts.line(
                data_key="score",
                stroke="none",
                dot=marker,
                active_dot=False,
                connect_nulls=False,
                is_animation_active=False,
                name="score set",
            ),
            data=rows,
            height=320,
        ),
        spacing="2",
        width="100%",
    )


def falls_chart(bars, height=260):
    """
    Where scores landed inside the distributions predicted for them.

    An even height across every bin is what a model with the spread right
    looks like, so the flat rule is the thing to read the bars against.
    """
    return rx.vstack(
        legend(("share of scores", SERIES_1), ("even, if the fit is right", MUTED)),
        frame(
            rx.recharts.x_axis(
                data_key="bin",
                tick_line=False,
                axis_line=True,
                tick=TICK,
                interval=3,
                height=34,
                label={
                    "value": "where the score fell, 0 below to 1 above",
                    "position": "insideBottom",
                    "offset": -2,
                    "fill": MUTED,
                    "fontSize": 11,
                },
            ),
            rx.recharts.y_axis(
                tick_line=False,
                axis_line=False,
                tick=TICK,
                width=44,
            ),
            rx.recharts.bar(
                data_key="share",
                fill=SERIES_1,
                radius=[4, 4, 0, 0],
                is_animation_active=False,
                name="share",
            ),
            rx.recharts.line(
                data_key="even",
                stroke=MUTED,
                stroke_width=2,
                dot=False,
                is_animation_active=False,
                name="even",
            ),
            data=bars,
            height=height,
        ),
        spacing="2",
        width="100%",
    )


def drift_chart(bars, caption, domain, height=260):
    """
    How far the middle of those landings sits from the middle of the
    predicted distribution, band by band.

    Zero is agreement, so the sign is the reading and the colour carries
    it: one hue for scores landing lower than predicted, the opposite hue
    for higher, and near-nothing in the middle.
    """
    return rx.vstack(
        legend(("lands below prediction", POLE_LOW),
               ("lands above", POLE_HIGH)),
        frame(
            rx.recharts.x_axis(
                data_key="band",
                tick_line=False,
                axis_line=True,
                tick=TICK,
                height=34,
                label={
                    "value": caption,
                    "position": "insideBottom",
                    "offset": -2,
                    "fill": MUTED,
                    "fontSize": 11,
                },
            ),
            rx.recharts.y_axis(
                tick_line=False,
                axis_line=False,
                tick=TICK,
                width=52,
                domain=domain,
            ),
            rx.recharts.reference_line(y=0, stroke=AXIS, stroke_width=1),
            rx.recharts.bar(
                rx.foreach(
                    bars,
                    lambda row: rx.recharts.cell(fill=row["colour"]),
                ),
                data_key="offBy",
                radius=[4, 4, 0, 0],
                is_animation_active=False,
                name="off by",
            ),
            data=bars,
            height=height,
        ),
        spacing="2",
        width="100%",
    )


def belief_chart(curve):
    """One player's belief about their skill, against the population."""
    return rx.vstack(
        legend(("everyone", MUTED), ("this player", SERIES_1)),
        frame(
            rx.recharts.x_axis(
                data_key="x",
                type_="number",
                domain=[-3, 3],
                tick_line=False,
                axis_line=True,
                tick=TICK,
                height=34,
                label={
                    "value": "skill",
                    "position": "insideBottom",
                    "offset": -2,
                    "fill": MUTED,
                    "fontSize": 11,
                },
            ),
            rx.recharts.y_axis(hide=True),
            rx.recharts.area(
                data_key="population",
                stroke=MUTED,
                stroke_width=1,
                fill=MUTED,
                fill_opacity=0.12,
                is_animation_active=False,
                name="everyone",
            ),
            rx.recharts.area(
                data_key="belief",
                stroke=SERIES_1,
                stroke_width=2,
                fill=SERIES_1,
                fill_opacity=0.18,
                is_animation_active=False,
                name="this player",
            ),
            data=curve,
            height=220,
        ),
        spacing="2",
        width="100%",
    )
