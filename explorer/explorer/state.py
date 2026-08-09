"""
What the page is currently showing.

The fit itself lives in fitted.py and stays numpy. Everything here is a
plain list or dict, because a Reflex state var has to survive being sent
to the browser as JSON.

Sorting and filtering happen here rather than in the browser, so the same
table can be ordered by any column without shipping every row.
"""

import asyncio

import reflex as rx

from . import fitted, metadata

# The fit is expensive and shared by every visitor, so it is held once on
# the module rather than per session.
CURRENT = {"fit": None}

ROWS_SHOWN = 250

MAP_ORDERS = {
    "steepest curve": ("slope", True),
    "flattest curve": ("slope", False),
    "hardest at median skill": ("atMedian", False),
    "easiest at median skill": ("atMedian", True),
    "most players": ("players", True),
    "highest star rating": ("stars", True),
    "most pp for its difficulty": ("gapCurve", True),
    "least pp for its difficulty": ("gapCurve", False),
    "most played": ("playcount", True),
    "longest": ("length", True),
}

PLAYER_ORDERS = {
    "highest skill": ("skill", True),
    "lowest skill": ("skill", False),
    "least certain": ("sd", True),
    "most certain": ("sd", False),
    "most scores": ("scores", True),
}


def held():
    if CURRENT["fit"] is None:
        CURRENT["fit"] = fitted.load()

    return CURRENT["fit"]


def ordered(rows, orders, choice):
    field, descending = orders[choice]
    ranked = [r for r in rows if r.get(field) is not None]
    missing = [r for r in rows if r.get(field) is None]
    ranked.sort(key=lambda r: r[field], reverse=descending)

    return ranked + missing


class Explorer(rx.State):
    """One session's view of the fit."""

    ready: bool = False
    working: str = ""
    summary: dict = {}

    every_cell: bool = False

    map_query: str = ""
    map_mods: str = "any"
    map_order: str = "most players"
    chosen_map: int = -1

    player_query: str = ""
    player_stratum: str = "any"
    player_order: str = "highest skill"
    chosen_player: int = -1

    falls_split: str = "how much the map separates players"

    # Song titles arrive after the page does, so they live beside the maps
    # rather than inside them: the fit does not need them and should not
    # be redone when they land.
    song: dict[str, str] = {}
    cached_songs: str = ""

    # -- getting the fit in

    @rx.event(background=True)
    async def open(self):
        if self.ready:
            return

        async with self:
            self.working = "fitting the panel; the first one takes a while"

        # Fitting is numpy and holds the interpreter, so it goes to a
        # thread. Left on the event loop it stops the socket the page is
        # waiting on, and the page never hears that the fit is done.
        fit = await asyncio.to_thread(held)

        async with self:
            self.summary = fit.summary
            self.chosen_map = self.busiest_map(fit)
            self.chosen_player = fit.players[0]["index"] if fit.players else -1
            self.ready = True
            self.working = ""

        return Explorer.name_the_maps

    @rx.event(background=True)
    async def refit(self):
        async with self:
            self.working = "reading the database again and refitting"
            self.ready = False

        fit = await asyncio.to_thread(
            fitted.load, {"all_cells": self.every_cell}, True
        )
        CURRENT["fit"] = fit

        async with self:
            self.summary = fit.summary
            self.chosen_map = self.busiest_map(fit)
            self.chosen_player = fit.players[0]["index"] if fit.players else -1
            self.ready = True
            self.working = ""

        return Explorer.name_the_maps

    @rx.event
    def set_every_cell(self, value: bool):
        self.every_cell = value

    @staticmethod
    def busiest_map(fit):
        if not fit.maps:
            return -1

        return max(fit.maps, key=lambda m: m["players"])["index"]

    # -- maps

    @rx.var
    def mod_choices(self) -> list[str]:
        if not self.ready:
            return ["any"]

        return ["any"] + sorted({m["mods"] for m in held().maps})

    @rx.var
    def map_order_choices(self) -> list[str]:
        return list(MAP_ORDERS)

    def with_song(self, row):
        """The map, under its song title once one has been fetched."""
        title = self.song.get(str(row["beatmap"]))

        return dict(row, name=title) if title else row

    @rx.var
    def map_rows(self) -> list[dict]:
        if not self.ready:
            return []

        wanted = self.map_query.strip().lower()
        rows = [
            self.with_song(m) for m in held().maps
            if (self.map_mods == "any" or m["mods"] == self.map_mods)
        ]

        if wanted:
            rows = [
                m for m in rows
                if wanted in m["name"].lower() or wanted in m["key"]
            ]

        return ordered(rows, MAP_ORDERS, self.map_order)[:ROWS_SHOWN]

    @rx.var
    def map_total(self) -> int:
        if not self.ready:
            return 0

        wanted = self.map_query.strip().lower()

        return sum(
            1 for m in held().maps
            if (self.map_mods == "any" or m["mods"] == self.map_mods)
            and (not wanted or wanted in m["name"].lower()
                 or wanted in m["key"])
        )

    @rx.var
    def map_detail(self) -> dict:
        if not self.ready or self.chosen_map < 0:
            return {}

        return self.with_song(held().maps[self.chosen_map])

    @rx.var
    def map_title(self) -> str:
        chosen = self.map_detail

        return chosen.get("name", "") if chosen else ""

    @rx.var
    def map_caption(self) -> str:
        chosen = self.map_detail

        if not chosen:
            return ""

        return (
            f"{chosen['key']} · {chosen['players']} players · "
            f"typical skill {chosen['typical']} · "
            f"{chosen['starsText']} stars"
        )

    @rx.var
    def map_url(self) -> str:
        chosen = self.map_detail

        return chosen.get("url", "") if chosen else ""

    @rx.var
    def player_url(self) -> str:
        chosen = self.player_detail

        return chosen.get("url", "") if chosen else ""

    @rx.var
    def showing_maps(self) -> str:
        return f"showing {len(self.map_rows)} of {self.map_total} maps"

    @rx.var
    def map_curve(self) -> list[dict]:
        if not self.ready or self.chosen_map < 0:
            return []

        return held().curve_and_scores(self.chosen_map)

    @rx.var
    def map_range(self) -> dict:
        if not self.ready or self.chosen_map < 0:
            return {"domain": [0, 3.4], "ticks": [0, 1, 2, 3]}

        return held().accuracy_range(self.chosen_map)

    @rx.var
    def map_marker(self) -> dict:
        if not self.ready or self.chosen_map < 0:
            return {}

        return held().marker(self.chosen_map)

    @rx.var
    def map_falls(self) -> list[dict]:
        if not self.ready or self.chosen_map < 0:
            return []

        fit = held()

        return fit.falls(fit.cells_of_map.get(self.chosen_map, fitted.EMPTY))

    @rx.event
    def choose_map(self, index: int):
        self.chosen_map = index

        return Explorer.name_the_maps

    @rx.event
    def set_map_query(self, value: str):
        self.map_query = value

        return Explorer.name_the_maps

    @rx.event
    def set_map_mods(self, value: str):
        self.map_mods = value

        return Explorer.name_the_maps

    @rx.event
    def set_map_order(self, value: str):
        self.map_order = value

        return Explorer.name_the_maps

    @rx.event(background=True)
    async def name_the_maps(self):
        """
        Put song titles on the maps currently listed.

        Only the ones on screen, because that is a few hundred against
        tens of thousands in the database, and the fit needs none of them.
        Cached titles land at once; the rest cost a request per fifty maps
        and arrive a moment later.
        """
        async with self:
            if not self.ready:
                return

            shown = {
                row["beatmap"]: row.get("version", "")
                for row in self.map_rows + self.player_scores
            }

            if self.chosen_map >= 0:
                chosen = held().maps[self.chosen_map]
                shown[chosen["beatmap"]] = chosen.get("version", "")

            already = set(self.song)

        wanted = [b for b in shown if str(b) not in already]

        if not wanted:
            return

        found = await asyncio.to_thread(metadata.named, wanted)
        await self.show_songs(found, shown)

        missing = [b for b in wanted if b not in found]

        if missing:
            await self.show_songs(
                await asyncio.to_thread(metadata.fetch, missing), shown
            )

        rows, size = await asyncio.to_thread(metadata.held)

        async with self:
            self.cached_songs = f"{rows:,} songs cached, {size // 1024:,}kB"

    async def show_songs(self, found, versions):
        if not found:
            return

        async with self:
            for beatmap_id, entry in found.items():
                title = metadata.label(entry, versions.get(beatmap_id, ""))

                if title:
                    self.song[str(beatmap_id)] = title

    # -- players

    @rx.var
    def stratum_choices(self) -> list[str]:
        if not self.ready:
            return ["any"]

        present = {p["stratum"] for p in held().players}

        return ["any"] + [s for s in held().strata if s in present]

    @rx.var
    def player_order_choices(self) -> list[str]:
        return list(PLAYER_ORDERS)

    @rx.var
    def player_rows(self) -> list[dict]:
        if not self.ready:
            return []

        wanted = self.player_query.strip().lower()
        rows = [
            p for p in held().players
            if (self.player_stratum == "any"
                or p["stratum"] == self.player_stratum)
            and (not wanted or wanted in p["name"].lower())
        ]

        return ordered(rows, PLAYER_ORDERS, self.player_order)[:ROWS_SHOWN]

    @rx.var
    def player_detail(self) -> dict:
        if not self.ready or self.chosen_player < 0:
            return {}

        return held().players[self.chosen_player]

    @rx.var
    def player_title(self) -> str:
        chosen = self.player_detail

        return chosen.get("name", "") if chosen else ""

    @rx.var
    def player_caption(self) -> str:
        chosen = self.player_detail

        if not chosen:
            return ""

        return (
            f"{chosen['stratum']} · skill {chosen['skillText']} "
            f"{chosen['sdText']} · {chosen['percentileText']} of the panel "
            f"sits below · {chosen['scores']} scores"
        )

    @rx.var
    def panel_line(self) -> str:
        if not self.ready:
            return ""

        shown = self.summary

        return (
            f"{shown['cells']} · {shown['players']} players · "
            f"{shown['items']} maps · {shown['observations']:,} scores"
        )

    @rx.var
    def population_line(self) -> str:
        if not self.ready:
            return ""

        shown = self.summary

        return (
            f"skills average {shown['skillMean']} with spread "
            f"{shown['skillSd']}, belief width {shown['medianWidth']} · "
            f"fitted {shown['fittedAt']}"
        )

    @rx.var
    def player_belief(self) -> list[dict]:
        if not self.ready or self.chosen_player < 0:
            return []

        return held().belief(self.chosen_player)

    @rx.var
    def player_scores(self) -> list[dict]:
        if not self.ready or self.chosen_player < 0:
            return []

        return [
            self.with_song(row)
            for row in held().player_scores(self.chosen_player)
        ]

    @rx.event
    def choose_player(self, index: int):
        self.chosen_player = index

        return Explorer.name_the_maps

    @rx.event
    def set_player_query(self, value: str):
        self.player_query = value

    @rx.event
    def set_player_stratum(self, value: str):
        self.player_stratum = value

    @rx.event
    def set_player_order(self, value: str):
        self.player_order = value

    # -- the distribution over everything

    @rx.var
    def all_falls(self) -> list[dict]:
        if not self.ready:
            return []

        fit = held()

        return fit.falls(slice_of(fit))

    @rx.var
    def split_choices(self) -> list[str]:
        return [
            "how much the map separates players",
            "player skill",
            "star rating",
            "how hard the map is",
        ]

    @rx.var
    def falls_drift(self) -> list[dict]:
        if not self.ready:
            return []

        fit = held()
        cells = slice_of(fit)

        if self.falls_split == "player skill":
            values = [
                fit.players[int(i)]["skill"] for i in fit.score_player[cells]
            ]
            label = "skill"
        elif self.falls_split == "star rating":
            values = [fit.maps[int(j)]["stars"] for j in fit.score_map[cells]]
            label = "stars"
        elif self.falls_split == "how hard the map is":
            values = [
                fit.maps[int(j)]["atMedian"] for j in fit.score_map[cells]
            ]
            label = "accuracy an average player reaches"
        else:
            values = [fit.maps[int(j)]["slope"] for j in fit.score_map[cells]]
            label = "separates by"

        return keep_rated(fit, cells, values, label)

    @rx.var
    def drift_domain(self) -> list[float]:
        """
        A symmetric scale wide enough for the bars and no wider.

        Fixing it in advance either flattens every bar into a sliver or
        crops the one band that matters, and the sizes here differ by an
        order of magnitude between splits.
        """
        rows = self.falls_drift
        widest = max([abs(r["offBy"]) for r in rows], default=0.0)
        edge = max(0.02, round(widest * 1.25, 3))

        return [-edge, edge]

    @rx.var
    def drift_caption(self) -> str:
        return f"{self.falls_split}, low to high"

    @rx.event
    def set_falls_split(self, value: str):
        self.falls_split = value


def slice_of(fit):
    import numpy as np

    return np.arange(len(fit.score_outcome))


def keep_rated(fit, cells, values, label):
    """Drop the cells whose split value is missing, then band the rest."""
    import numpy as np

    usable = np.array([v is not None for v in values])

    if not usable.any():
        return []

    kept = np.asarray(values, dtype=object)[usable].astype(float)

    return fit.falls_by(cells[usable], kept, label)
