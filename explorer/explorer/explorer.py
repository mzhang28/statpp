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


def explain(*lines):
    """
    Where a number on screen came from.

    A popover rather than a title attribute, because a phone has no hover
    to put one behind, and the same tap target answers a mouse.
    """
    return rx.popover.root(
        rx.popover.trigger(
            rx.icon_button(
                rx.icon("info", size=13),
                variant="ghost",
                color_scheme="gray",
                size="1",
                aria_label="how this is worked out",
                class_name="align-middle",
            ),
        ),
        rx.popover.content(
            rx.vstack(
                *[
                    rx.text(line, class_name="text-sm text-[var(--ink-2)]")
                    for line in lines
                ],
                spacing="2",
            ),
            side="top",
            align="end",
            max_width="22rem",
            class_name="bg-[var(--surface)]",
        ),
    )


def titled(text, *lines, size="4"):
    """A heading with the explanation of what sits under it beside it."""
    return rx.hstack(
        rx.heading(text, size=size, class_name="text-[var(--ink)]"),
        explain(*lines),
        spacing="1",
        align="center",
    )


def on_osu(url, what):
    """A way out to the osu! page for whatever is on screen."""
    return rx.link(
        rx.button(
            rx.icon("external-link", size=14),
            what,
            variant="soft",
            size="1",
            class_name="whitespace-nowrap",
        ),
        href=url,
        is_external=True,
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
                    rx.cond(
                        Explorer.cached_songs,
                        rx.text(
                            Explorer.cached_songs,
                            class_name="text-xs text-[var(--muted)] figures",
                        ),
                    ),
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
                rx.hstack(
                    rx.text(
                        Explorer.showing_maps,
                        class_name="text-xs text-[var(--muted)]",
                    ),
                    explain(
                        "One row per beatmap played under one set of mods, "
                        "since the same map with DT is not the same thing "
                        "to play.",
                        "Separates by: how much the expected accuracy "
                        "changes as skill changes, read where this map's "
                        "own players sit. Near zero means everyone scores "
                        "about the same and the map tells them apart on "
                        "nothing.",
                        "Average player: the accuracy the curve expects "
                        "from someone at the middle of the panel, which "
                        "for a map only strong players touch is the fit "
                        "guessing from similar maps.",
                        "pp gap: what osu! awards here, minus what it "
                        "awards on maps the fit rates equally hard. "
                        "Positive means more pp than the difficulty "
                        "measured here accounts for.",
                    ),
                    spacing="1",
                    align="center",
                ),
                table(
                    ["map", "players", "separates by", "average player",
                     "stars", "pp gap"],
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
                    rx.hstack(
                        rx.vstack(
                            rx.heading(
                                Explorer.map_title,
                                size="4",
                                class_name="text-[var(--ink)]",
                            ),
                            rx.text(
                                Explorer.map_caption,
                                class_name="text-sm text-[var(--ink-2)] "
                                "figures",
                            ),
                            spacing="1",
                            align="start",
                            class_name="min-w-0",
                        ),
                        rx.spacer(),
                        on_osu(Explorer.map_url, "osu!"),
                        width="100%",
                        align="start",
                        spacing="3",
                    ),
                    charts.curve_chart(
                        Explorer.map_curve,
                        Explorer.map_marker,
                        Explorer.map_range,
                    ),
                    rx.hstack(
                        rx.text(
                            "A flat curve means everyone scores about the "
                            "same here, so a score on this map says little "
                            "about who set it.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "The line is the accuracy the fit expects at "
                            "each skill, and the band is one spread either "
                            "side of it. Both are fitted to this map's own "
                            "scores, pulled towards the average map where "
                            "it has few.",
                            "Each dot is one score: the player's fitted "
                            "skill across, the accuracy they got up. A "
                            "map with more than 1200 of them is drawn from "
                            "an even sample.",
                            "The scale is stretched near the top, so the "
                            "gap from 99% to 99.9% takes as much room as "
                            "90% to 99%. Equal steps are equally hard to "
                            "make.",
                        ),
                        spacing="1",
                        align="start",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.text(
                            "Where the scores on this map landed against "
                            "what was predicted for each of them.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "For every score on this map, the fit had a "
                            "whole distribution in mind rather than one "
                            "number. This counts where the score that "
                            "happened sat inside it, from 0 far below to "
                            "1 far above.",
                            "Level bars mean the spread is about right "
                            "here. A pile at one end means the map's "
                            "players beat the prediction, or missed it, "
                            "more often than the fit allows for.",
                        ),
                        spacing="1",
                        align="start",
                        width="100%",
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
                rx.hstack(
                    rx.text(
                        "every player in the panel",
                        class_name="text-xs text-[var(--muted)]",
                    ),
                    explain(
                        "Skill is one number per player on a scale where "
                        "the whole panel averages zero and spreads by "
                        "one. It comes from their scores and the curves "
                        "of the maps they set them on, solved together.",
                        "Below them: the share of this panel sitting "
                        "lower. The panel is drawn from log-spaced slices "
                        "of the ranking rather than at random, so it is a "
                        "place among these players and not among "
                        "everyone.",
                        "Uncertainty: how wide the fit's belief about "
                        "them is. Few scores, or scores on maps that "
                        "separate nobody, leave it wide.",
                    ),
                    spacing="1",
                    align="center",
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
                    rx.hstack(
                        rx.vstack(
                            rx.heading(
                                Explorer.player_title,
                                size="4",
                                class_name="text-[var(--ink)]",
                            ),
                            rx.text(
                                Explorer.player_caption,
                                class_name="text-sm text-[var(--ink-2)] "
                                "figures",
                            ),
                            spacing="1",
                            align="start",
                            class_name="min-w-0",
                        ),
                        rx.spacer(),
                        on_osu(Explorer.player_url, "osu!"),
                        width="100%",
                        align="start",
                        spacing="3",
                    ),
                    rx.hstack(
                        rx.text(
                            "Where the fit thinks they sit, against "
                            "everyone.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "The grey curve is the population every skill "
                            "is measured against, held to average zero and "
                            "spread one while the fit runs.",
                            "The blue curve is this player: its middle is "
                            "their skill and its width is how sure the fit "
                            "is. A narrow one means their scores agree "
                            "with each other and were set on maps that "
                            "separate players.",
                        ),
                        spacing="1",
                        align="start",
                        width="100%",
                    ),
                    charts.belief_chart(Explorer.player_belief),
                    rx.hstack(
                        rx.text(
                            "Their scores, the best above prediction first.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "Expected of them: the accuracy this map's "
                            "curve predicts at this player's fitted "
                            "skill.",
                            "Where it fell: how the accuracy they got "
                            "compares, as a place inside the whole "
                            "predicted distribution. Past halfway means "
                            "they beat what was expected on that map.",
                        ),
                        spacing="1",
                        align="start",
                        width="100%",
                    ),
                    table(
                        ["map", "accuracy", "expected of them", "where it fell"],
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
                titled(
                    "Where every score fell",
                    "Each score's accuracy is turned into its place inside "
                    "the distribution the fit predicted for that exact "
                    "cell: this player, this map, these mods.",
                    "Twenty bins across, and the share of all scores "
                    "falling in each. A fit with the spread right scatters "
                    "them evenly, so the flat line is what agreement looks "
                    "like and needs no held-out data to check.",
                    "A hump in the middle means the predicted spread is "
                    "too wide; piles at both ends mean it is too narrow.",
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
                        titled(
                            "Split by",
                            "The same scores, cut into ten equal-sized "
                            "bands of whatever is chosen, with the middle "
                            "of each band's landings plotted against the "
                            "halfway mark.",
                            "Zero means the fit is right on average for "
                            "that band. A bar below zero means scores "
                            "there come in lower than predicted, above "
                            "means higher. A steady slope across the "
                            "bands is the fit being wrong in a way that "
                            "tracks whatever the bands are made of.",
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
