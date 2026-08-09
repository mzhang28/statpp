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
                aria_label="How this number is made",
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
                        "The fit gives each player a skill and each map a "
                        "curve. It reads the scores. It does not read pp.",
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
                    "Include the top-100 lists",
                    checked=Explorer.every_cell,
                    on_change=Explorer.set_every_cell,
                    class_name="text-sm",
                ),
                rx.button(
                    "Fit again from the database",
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
                        "Each row is one beatmap with one set of mods. "
                        "The same beatmap with DT is a different row, "
                        "because DT changes how hard it is.",
                        "Separates by: the change in expected accuracy "
                        "for one step of skill. The fit reads this number "
                        "where the players of this map sit. A value near "
                        "zero means that all players get almost the same "
                        "accuracy.",
                        "Average player: the accuracy that the curve "
                        "gives at the middle of the panel. If only strong "
                        "players play this map, the fit has no data "
                        "there. Then it uses similar maps.",
                        "pp gap: the pp of this map, less the pp of maps "
                        "that the fit rates equally hard. A positive "
                        "value means that osu! gives more pp than the "
                        "measured difficulty explains.",
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
                            "A flat curve means that all players get "
                            "almost the same accuracy. Then a score here "
                            "tells you little about the player.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "The line is the expected accuracy at each "
                            "skill. The band is one spread on each side "
                            "of the line. Both come from the scores of "
                            "this map. If the map has few scores, the fit "
                            "moves them toward the average map.",
                            "Each dot is one score. The horizontal "
                            "position is the skill of the player. The "
                            "vertical position is the accuracy. For a map "
                            "with more than 1200 scores, this chart shows "
                            "an even sample of them.",
                            "The vertical scale is not linear. The step "
                            "from 99% to 99.9% is as wide as the step "
                            "from 90% to 99%. Equal steps on this scale "
                            "are equally hard to reach.",
                        ),
                        spacing="1",
                        align="start",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.text(
                            "This chart shows where the scores of this "
                            "map sit inside their predictions.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "For each score the fit gives a full "
                            "distribution, not one number. This chart "
                            "counts the position of the real score inside "
                            "that distribution. A position of 0 is far "
                            "below the prediction. A position of 1 is far "
                            "above it.",
                            "Bars of equal height mean that the spread is "
                            "correct for this map. Tall bars at one end "
                            "mean that the players of this map miss the "
                            "prediction in one direction.",
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
                    "Select a map from the list.",
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
                        "This list shows every player in the panel.",
                        class_name="text-xs text-[var(--muted)]",
                    ),
                    explain(
                        "Skill is one number for each player. The panel "
                        "has an average skill of zero and a spread of "
                        "one. The fit finds the skills and the map curves "
                        "together, from the scores.",
                        "Below them: the part of the panel with a lower "
                        "skill. The panel comes from selected pages of "
                        "the ranking, not from a random sample. This "
                        "value is a place among these players only.",
                        "Uncertainty: the width of the belief that the "
                        "fit has about this player. Few scores make it "
                        "wide. Scores on maps that separate no players "
                        "also make it wide.",
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
                            "This chart shows the skill of this player "
                            "against the whole panel.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "The gray curve is the panel. The fit holds "
                            "it at an average of zero and a spread of "
                            "one.",
                            "The blue curve is this player. The center is "
                            "the skill. The width is the uncertainty. A "
                            "thin curve means that the scores agree with "
                            "each other. It also means that the maps "
                            "separate players well.",
                        ),
                        spacing="1",
                        align="start",
                        width="100%",
                    ),
                    charts.belief_chart(Explorer.player_belief),
                    rx.hstack(
                        rx.text(
                            "The scores of this player. The best score "
                            "against its prediction is first.",
                            class_name="text-sm text-[var(--ink-2)]",
                        ),
                        explain(
                            "Expected of them: the accuracy that the "
                            "curve of this map gives at the skill of this "
                            "player.",
                            "Where it fell: the position of the real "
                            "accuracy inside the prediction. A value of "
                            "more than 0.5 means that the player did "
                            "better than the prediction.",
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
                    "Select a player from the list.",
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
                    "The fit makes one prediction for each score, for this "
                    "player on this map with these mods. This chart turns "
                    "each accuracy into its position inside that "
                    "prediction.",
                    "The chart has twenty bins. Each bar is the part of "
                    "all scores in that bin. A correct spread puts an "
                    "equal part in each bin. The flat line shows that "
                    "height.",
                    "Tall bars in the middle mean that the predicted "
                    "spread is too wide. Tall bars at both ends mean that "
                    "it is too small.",
                ),
                rx.text(
                    "The fit gives each score a full distribution, not one "
                    "number. This chart shows where the real score sits "
                    "inside it. A value of 0 is far below the prediction. "
                    "A value of 1 is far above it. If the spread is "
                    "correct, the bars sit level with the line.",
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
                            "The chart puts the same scores into ten bands "
                            "of equal size. You select what makes the "
                            "bands. Each bar is the middle position for "
                            "one band, against the halfway mark.",
                            "Zero means that the fit is correct for that "
                            "band. A bar below zero means that the scores "
                            "are lower than the prediction. A bar above "
                            "zero means that they are higher. A steady "
                            "slope across the bands means that the error "
                            "follows the value of the band.",
                        ),
                        rx.text(
                            "A bar away from zero means that the fit is "
                            "wrong in one direction for that band.",
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
