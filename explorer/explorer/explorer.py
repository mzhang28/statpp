"""
A page for reading the fit: the maps, the players, and where the scores
landed inside the distributions the model predicted for them.

Two columns on a wide screen, one on a narrow one. The list on the left is
sortable and filterable; picking a row draws it on the right.
"""

import reflex as rx

from . import charts
from .state import Explorer

CARD = (
    "rounded-xl border border-[var(--hairline)] bg-[var(--surface)] "
    "p-4 sm:p-5"
)
FIELD_LABEL = "text-xs text-[var(--muted)] uppercase tracking-wide"


def stat(label, value):
    return rx.vstack(
        rx.text(label, class_name=FIELD_LABEL),
        rx.text(value, class_name="text-lg figures text-[var(--ink)]"),
        spacing="0",
        align="start",
    )


def picker(label, value, choices, on_change):
    return rx.vstack(
        rx.text(label, class_name=FIELD_LABEL),
        rx.select(
            choices,
            value=value,
            on_change=on_change,
            width="100%",
        ),
        spacing="1",
        align="start",
        class_name="min-w-[9rem] flex-1",
    )


def search(label, value, on_change, placeholder):
    return rx.vstack(
        rx.text(label, class_name=FIELD_LABEL),
        rx.input(
            value=value,
            on_change=on_change,
            placeholder=placeholder,
            width="100%",
        ),
        spacing="1",
        align="start",
        class_name="min-w-[10rem] flex-1",
    )


def header():
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading(
                        "statpp", size="6", class_name="text-[var(--ink)]"
                    ),
                    rx.text(
                        "skill and map curves fitted to the scores "
                        "themselves, not to pp",
                        class_name="text-sm text-[var(--ink-2)]",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                rx.hstack(
                    rx.color_mode.button(),
                    spacing="2",
                    align="center",
                ),
                width="100%",
                align="start",
            ),
            rx.cond(
                Explorer.ready,
                rx.vstack(
                    stat("panel", Explorer.panel_line),
                    stat("population", Explorer.population_line),
                    spacing="3",
                    align="start",
                ),
            ),
            rx.hstack(
                rx.checkbox(
                    "every cell, including the pp-truncated top-100 lists",
                    checked=Explorer.every_cell,
                    on_change=Explorer.set_every_cell,
                    class_name="text-sm",
                ),
                rx.button(
                    "refit from the database",
                    on_click=Explorer.refit,
                    disabled=~Explorer.ready,
                    variant="soft",
                    size="2",
                ),
                spacing="4",
                align="center",
                wrap="wrap",
                class_name="gap-y-2",
            ),
            spacing="4",
            width="100%",
            align="start",
        ),
        class_name=CARD + " w-full",
    )


def table(head, body):
    return rx.box(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    *[
                        rx.table.column_header_cell(
                            name,
                            class_name="text-xs text-[var(--muted)] "
                            "uppercase tracking-wide whitespace-nowrap",
                        )
                        for name in head
                    ]
                )
            ),
            rx.table.body(body),
            variant="ghost",
            size="1",
            width="100%",
        ),
        class_name="overflow-x-auto max-h-[26rem] overflow-y-auto w-full",
    )


def map_row(row):
    return rx.table.row(
        rx.table.cell(
            rx.text(row["name"], class_name="truncate max-w-[16rem]"),
            rx.text(
                row["key"],
                class_name="text-xs text-[var(--muted)] figures",
            ),
        ),
        rx.table.cell(row["players"], class_name="figures"),
        rx.table.cell(row["slopeText"], class_name="figures"),
        rx.table.cell(row["atMedianText"], class_name="figures"),
        rx.table.cell(row["starsText"], class_name="figures"),
        rx.table.cell(row["gapCurveText"], class_name="figures"),
        on_click=Explorer.choose_map(row["index"]),
        class_name="cursor-pointer row-button",
    )


def maps_tab():
    return rx.grid(
        rx.box(
            rx.vstack(
                rx.hstack(
                    search(
                        "search",
                        Explorer.map_query,
                        Explorer.set_map_query,
                        "song, difficulty, or id:mods",
                    ),
                    picker(
                        "mods",
                        Explorer.map_mods,
                        Explorer.mod_choices,
                        Explorer.set_map_mods,
                    ),
                    picker(
                        "sort by",
                        Explorer.map_order,
                        Explorer.map_order_choices,
                        Explorer.set_map_order,
                    ),
                    spacing="3",
                    width="100%",
                    wrap="wrap",
                    class_name="gap-y-3",
                ),
                rx.text(
                    Explorer.showing_maps,
                    class_name="text-xs text-[var(--muted)]",
                ),
                table(
                    ["map", "players", "slope", "nines at 50th", "stars",
                     "pp gap"],
                    rx.foreach(Explorer.map_rows, map_row),
                ),
                spacing="3",
                width="100%",
            ),
            class_name=CARD,
        ),
        rx.box(
            rx.cond(
                Explorer.map_detail,
                rx.vstack(
                    rx.vstack(
                        rx.heading(
                            Explorer.map_title,
                            size="4",
                            class_name="text-[var(--ink)]",
                        ),
                        rx.text(
                            Explorer.map_caption,
                            class_name="text-sm text-[var(--ink-2)] figures",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    charts.curve_chart(
                        Explorer.map_curve, Explorer.map_marker
                    ),
                    rx.text(
                        "A flat curve means everyone scores about the same "
                        "here, so a score on this map says little about who "
                        "set it.",
                        class_name="text-sm text-[var(--ink-2)]",
                    ),
                    charts.falls_chart(Explorer.map_falls, height=220),
                    spacing="4",
                    width="100%",
                ),
                rx.text(
                    "Pick a map on the left.",
                    class_name="text-sm text-[var(--muted)]",
                ),
            ),
            class_name=CARD,
        ),
        columns=rx.breakpoints(initial="1", lg="2"),
        spacing="4",
        width="100%",
    )


def player_row(row):
    return rx.table.row(
        rx.table.cell(
            rx.text(row["name"], class_name="truncate max-w-[10rem]"),
        ),
        rx.table.cell(row["stratum"], class_name="text-xs figures"),
        rx.table.cell(row["skillText"], class_name="figures"),
        rx.table.cell(row["percentileText"], class_name="figures"),
        rx.table.cell(row["sdText"], class_name="figures"),
        rx.table.cell(row["scores"], class_name="figures"),
        on_click=Explorer.choose_player(row["index"]),
        class_name="cursor-pointer row-button",
    )


def score_row(row):
    return rx.table.row(
        rx.table.cell(
            rx.text(row["name"], class_name="truncate max-w-[14rem]"),
            rx.text(
                row["mods"], class_name="text-xs text-[var(--muted)] figures"
            ),
        ),
        rx.table.cell(row["accuracyText"], class_name="figures"),
        rx.table.cell(row["expectedText"], class_name="figures"),
        rx.table.cell(
            rx.hstack(
                rx.box(
                    rx.box(
                        width=row["barWidth"],
                        height="100%",
                        class_name="rounded-full",
                        background=row["barColour"],
                    ),
                    class_name="h-1.5 w-16 rounded-full "
                    "bg-[var(--pole-mid)] overflow-hidden",
                ),
                rx.text(row["fellAtText"], class_name="figures text-xs"),
                spacing="2",
                align="center",
            )
        ),
    )


def players_tab():
    return rx.grid(
        rx.box(
            rx.vstack(
                rx.hstack(
                    search(
                        "search",
                        Explorer.player_query,
                        Explorer.set_player_query,
                        "username",
                    ),
                    picker(
                        "stratum",
                        Explorer.player_stratum,
                        Explorer.stratum_choices,
                        Explorer.set_player_stratum,
                    ),
                    picker(
                        "sort by",
                        Explorer.player_order,
                        Explorer.player_order_choices,
                        Explorer.set_player_order,
                    ),
                    spacing="3",
                    width="100%",
                    wrap="wrap",
                    class_name="gap-y-3",
                ),
                table(
                    ["player", "stratum", "skill", "below them",
                     "uncertainty", "scores"],
                    rx.foreach(Explorer.player_rows, player_row),
                ),
                spacing="3",
                width="100%",
            ),
            class_name=CARD,
        ),
        rx.box(
            rx.cond(
                Explorer.player_detail,
                rx.vstack(
                    rx.vstack(
                        rx.heading(
                            Explorer.player_title,
                            size="4",
                            class_name="text-[var(--ink)]",
                        ),
                        rx.text(
                            Explorer.player_caption,
                            class_name="text-sm text-[var(--ink-2)] figures",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    charts.belief_chart(Explorer.player_belief),
                    rx.text(
                        "Their scores, the best above prediction first.",
                        class_name="text-sm text-[var(--ink-2)]",
                    ),
                    table(
                        ["map", "accuracy", "expected nines", "where it fell"],
                        rx.foreach(Explorer.player_scores, score_row),
                    ),
                    spacing="4",
                    width="100%",
                ),
                rx.text(
                    "Pick a player on the left.",
                    class_name="text-sm text-[var(--muted)]",
                ),
            ),
            class_name=CARD,
        ),
        columns=rx.breakpoints(initial="1", lg="2"),
        spacing="4",
        width="100%",
    )


def falls_tab():
    return rx.grid(
        rx.box(
            rx.vstack(
                rx.heading(
                    "Where every score fell",
                    size="4",
                    class_name="text-[var(--ink)]",
                ),
                rx.text(
                    "For each score the model predicted a whole distribution "
                    "rather than one number. This is where the score that "
                    "actually happened landed inside it: 0 means far below "
                    "what the map and the player's skill led you to expect, "
                    "1 far above. If the fit has the spread right the bars "
                    "sit level with the line.",
                    class_name="text-sm text-[var(--ink-2)]",
                ),
                charts.falls_chart(Explorer.all_falls, height=300),
                spacing="3",
                width="100%",
            ),
            class_name=CARD,
        ),
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.heading(
                            "Split by",
                            size="4",
                            class_name="text-[var(--ink)]",
                        ),
                        rx.text(
                            "A bar away from zero means the model is wrong "
                            "in one direction for that band.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    rx.spacer(),
                    picker(
                        "band by",
                        Explorer.falls_split,
                        Explorer.split_choices,
                        Explorer.set_falls_split,
                    ),
                    width="100%",
                    align="start",
                    spacing="3",
                    wrap="wrap",
                    class_name="gap-y-3",
                ),
                charts.drift_chart(
                    Explorer.falls_drift,
                    Explorer.drift_caption,
                    Explorer.drift_domain,
                    height=300,
                ),
                spacing="3",
                width="100%",
            ),
            class_name=CARD,
        ),
        columns=rx.breakpoints(initial="1", lg="2"),
        spacing="4",
        width="100%",
    )


def waiting():
    return rx.center(
        rx.vstack(
            rx.spinner(size="3"),
            rx.text(Explorer.working, class_name="text-sm text-[var(--ink-2)]"),
            spacing="3",
            align="center",
        ),
        class_name=CARD + " w-full min-h-[18rem]",
    )


def index():
    return rx.box(
        rx.vstack(
            header(),
            rx.cond(
                Explorer.ready,
                rx.tabs.root(
                    rx.tabs.list(
                        rx.tabs.trigger("Maps", value="maps"),
                        rx.tabs.trigger("Players", value="players"),
                        rx.tabs.trigger("Where scores fall", value="falls"),
                    ),
                    rx.tabs.content(maps_tab(), value="maps", class_name="pt-4"),
                    rx.tabs.content(
                        players_tab(), value="players", class_name="pt-4"
                    ),
                    rx.tabs.content(
                        falls_tab(), value="falls", class_name="pt-4"
                    ),
                    default_value="maps",
                    width="100%",
                ),
                waiting(),
            ),
            spacing="4",
            width="100%",
            class_name="max-w-[1400px] mx-auto",
        ),
        on_mount=Explorer.open,
        class_name="min-h-screen p-4 sm:p-6 bg-[var(--plane)]",
    )


app = rx.App(stylesheets=["/theme.css"])
app.add_page(index, route="/", title="statpp explorer")
