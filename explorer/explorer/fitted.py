"""
The fitted model, held in memory for the pages to read.

Fitting takes a couple of minutes, so it happens once per choice of
settings and the result stays here. Nothing in this module is a Reflex state var: numpy stays on this
side of the line and the pages are handed plain lists of dicts.

The database is read through the same read-only connection the command
line scripts use, so this runs while the sampler writes.
"""

import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

# star_ratings reads its cache directory at import time, and Reflex runs
# from the explorer directory rather than the repository root.
os.environ.setdefault("OSU_BEATMAP_CACHE", str(ROOT / "beatmap-files"))

from fit_skill_and_curves import (  # noqa: E402
    NotEnoughData,
    fit_panel,
    map_columns,
    map_facts,
    predictive_cdf,
    prepare,
    usernames,
)
from outcomes import INSIDE, logit, normal_cdf  # noqa: E402
from sample import connect_readonly  # noqa: E402

DATABASE = ROOT / "osu.sqlite"
CACHE = Path(__file__).resolve().parents[1] / ".cache"

DEFAULTS = {
    "family": "logit-normal",
    "min_players": 10,
    "min_items": 5,
    "knots": 6,
    "reach": 2.0,
    "quadrature": 7,
    "steps": 600,
    "rate": 0.05,
    "population": 1.0,
    "pool": {},
    "pool_level": 0.05,
    "star_window": 0.5,
    "skill_window": 0.25,
    "min_neighbours": 8,
}


@dataclass
class Fitted:
    """A finished fit, with the panel indexed the ways the pages ask for."""

    settings: dict
    summary: dict
    players: list
    maps: list
    strata: list

    params: object
    model: object

    # One entry per observed cell, all four aligned.
    score_player: np.ndarray
    score_map: np.ndarray
    score_outcome: np.ndarray
    score_fell_at: np.ndarray

    cells_of_map: dict = field(default_factory=dict)
    cells_of_player: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cells_of_map = index_by(self.score_map, len(self.maps))
        self.cells_of_player = index_by(self.score_player, len(self.players))

    # -- what the map view draws

    def curve(self, j, lo=-3.0, hi=3.0, steps=97):
        """
        The accuracy the map gives across the skill range: the middle of
        its distribution, and the stretch eight scores in ten fall inside.

        The model no longer has a mean and a spread to draw. It has a
        distribution per skill, so what is drawn are three of its
        quantiles, which say the same thing without assuming the shape is
        symmetric. It is not: near 100% there is far more room below the
        middle than above it.
        """
        skills = np.linspace(lo, hi, steps)
        values, _ = self.model.at(self.params, np.full(steps, j), skills)

        def at(share):
            return self.model.family.quantile(np.full(steps, share), values)

        return [
            {
                "x": round(float(x), 3),
                "mean": round(on_axis(m), 4),
                "band": [round(on_axis(a), 4), round(on_axis(b), 4)],
            }
            for x, m, a, b in zip(skills, at(0.5), at(0.1), at(0.9))
        ]

    def map_scores(self, j, cap=1200):
        """Every score on the map, as points to draw against the curve."""
        cells = self.cells_of_map.get(j, EMPTY)

        return [
            {
                "x": round(float(self.players[p]["skill"]), 3),
                "score": round(on_axis(self.score_outcome[c]), 4),
            }
            for c, p in zip(*thinned(cells, self.score_player, cap))
        ]

    def curve_and_scores(self, j):
        """
        The curve and the scores in one series of rows, ordered by skill.

        They share a chart, and the chart takes a single table: a row is
        either a point on the curve or a score that was set, and the other
        columns are left empty so nothing is drawn there.
        """
        rows = self.curve(j) + self.map_scores(j)
        rows.sort(key=lambda row: row["x"])

        return rows

    def accuracy_range(self, j):
        """
        The stretch of the axis this map needs, and round accuracies to
        mark it at.

        The axis is the logit of accuracy. That is a choice about drawing
        and not about the model: a linear axis puts every score worth
        comparing into the top twentieth of the chart, and the logit is
        already the coordinate every family writes its location channel
        in. The ticks carry percentages, since that is what a person
        reads.
        """
        rows = self.curve(j) + self.map_scores(j)

        low = min(
            [r["band"][0] for r in rows if "band" in r]
            + [r["score"] for r in rows if "score" in r]
        )
        high = max(
            [r["band"][1] for r in rows if "band" in r]
            + [r["score"] for r in rows if "score" in r]
        )

        low = low - 0.2
        high = high + 0.2

        # Marked at accuracies a person would name, which are not round
        # numbers once the scale is stretched.
        marks = [
            on_axis(a)
            for a in (0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995,
                      0.998, 0.999, 0.9995)
        ]
        inside = [round(t, 4) for t in marks if low <= t <= high]

        while len(inside) > 6:
            inside = inside[::2]

        return {
            "domain": [round(low, 3), round(high, 3)],
            "ticks": inside,
        }

    def marker(self, j):
        """
        How to draw one score.

        Big enough to aim at when a map has few of them, smaller once they
        start to pile up, and always ringed in the surface colour so
        overlapping scores stay countable.
        """
        crowd = len(self.cells_of_map.get(j, EMPTY))

        return {
            "r": 4 if crowd <= 150 else 2.5,
            "fill": "var(--series-2)",
            "stroke": "var(--surface)",
            "strokeWidth": 1.5,
        }

    # -- what the player view lists

    def player_scores(self, i, cap=400):
        cells = self.cells_of_player.get(i, EMPTY)

        skill = np.full(len(cells), self.players[i]["skill"])
        maps = self.score_map[cells]

        values, _ = self.model.at(self.params, maps, skill)
        middle = self.model.family.quantile(
            np.full(len(cells), 0.5), values
        )

        rows = [
            {
                "key": self.maps[int(j)]["key"],
                "name": self.maps[int(j)]["name"],
                "mods": self.maps[int(j)]["mods"],
                "beatmap": self.maps[int(j)]["beatmap"],
                "version": self.maps[int(j)]["version"],
                "url": self.maps[int(j)]["url"],
                "expectedText": f"{100.0 * float(m):.2f}%",
                "fellAt": round(float(self.score_fell_at[c]), 3),
                "fellAtText": f"{float(self.score_fell_at[c]):.2f}",
                "barWidth": f"{100 * float(self.score_fell_at[c]):.0f}%",
                "barColour": pole(float(self.score_fell_at[c]) - 0.5),
                "accuracyText":
                    f"{100.0 * float(self.score_outcome[c]):.2f}%",
            }
            for c, j, m in zip(cells, maps, middle)
        ]
        rows.sort(key=lambda r: -r["fellAt"])

        return rows[:cap]

    def belief(self, i, lo=-3.0, hi=3.0, steps=97):
        """
        The population against this one player's belief, each drawn to its
        own height.

        Both are densities and both cover an area of one, so the narrower
        of the two is also the taller. A player with two hundred scores is
        placed to within a twentieth of the population spread, which puts
        their peak twenty times above the population's and flattens the
        population into a line along the bottom. Scaling each to its own
        peak keeps the reading that matters, which is where the player
        sits and how wide the belief is against the spread behind it.
        """
        skills = np.linspace(lo, hi, steps)
        centre = self.players[i]["skill"]
        width = self.players[i]["sd"]

        population = np.exp(-0.5 * skills ** 2)
        belief = np.exp(-0.5 * ((skills - centre) / width) ** 2)

        return [
            {
                "x": round(float(x), 3),
                "population": round(float(a), 4),
                "belief": round(float(b), 4),
            }
            for x, a, b in zip(skills, population, belief)
        ]

    # -- what the distribution view draws

    def falls(self, cells, bins=20):
        """
        How the scores fall inside the distributions predicted for them.

        A model that has the spread right puts an equal share in every
        bin, so the flat line is what agreement looks like.
        """
        values = self.score_fell_at[cells]
        counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
        share = counts / max(len(values), 1)

        return [
            {
                "bin": f"{(k + 0.5) / bins:.2f}",
                "share": round(float(share[k]), 5),
                "even": round(1.0 / bins, 5),
            }
            for k in range(bins)
        ]

    def falls_by(self, cells, values, label, bands=10):
        """
        Where scores fall, split into bands of something else. Away from
        0.5 in a band means the model is wrong in one direction there.
        """
        if len(cells) == 0:
            return []

        edges = np.quantile(values, np.linspace(0.0, 1.0, bands + 1))
        edges[-1] += 1e-9

        rows = []

        for k in range(bands):
            inside = (values >= edges[k]) & (values < edges[k + 1])

            if not inside.any():
                continue

            middle = float(np.mean(self.score_fell_at[cells[inside]]))

            rows.append({
                "band": f"{edges[k]:.2f}",
                "label": f"{label} {edges[k]:.2f} to {edges[k + 1]:.2f}",
                "fellAt": round(middle, 4),
                "offBy": round(middle - 0.5, 4),
                "colour": pole(middle - 0.5),
                "scores": int(inside.sum()),
            })

        return rows


EMPTY = np.zeros(0, dtype=np.int64)


def pole(offset):
    """
    Which end of the diverging pair a signed number sits on.

    Chosen here rather than on the page, because the page only sees these
    numbers as opaque JSON and cannot compare them.
    """
    return "var(--pole-low)" if offset < 0 else "var(--pole-high)"


def index_by(keys, size):
    """Which cells belong to each row, in one pass rather than per lookup."""
    order = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[order], np.arange(size + 1))

    return {
        j: order[bounds[j]:bounds[j + 1]]
        for j in range(size)
        if bounds[j + 1] > bounds[j]
    }


def thinned(cells, lookup, cap):
    """Every cell, or an even sample of them when there are too many."""
    if len(cells) > cap:
        cells = cells[np.linspace(0, len(cells) - 1, cap).astype(int)]

    return cells, lookup[cells]


def on_axis(accuracy):
    """
    An accuracy placed on the chart's vertical scale.

    See accuracy_range for why the scale is the logit. Nothing outside
    the charts uses this, and no number the model produces is stored this
    way.
    """
    return float(logit(np.clip(np.asarray(accuracy, dtype=float), *INSIDE)))


def settings_from(overrides):
    chosen = dict(DEFAULTS)
    chosen.update(overrides or {})

    return chosen


def cache_path(settings):
    """
    Where this fit is kept.

    The name carries the settings and the modification time of this file,
    since a cached fit holds rows already turned into the strings the page
    shows. Editing how they are built has to reach the page, and only a
    rebuild does that.
    """
    stamp = json.dumps(settings, sort_keys=True) + str(
        Path(__file__).stat().st_mtime_ns
    )
    digest = hashlib.sha1(stamp.encode()).hexdigest()[:16]

    return CACHE / f"fit-{digest}.pickle"


def stratum_names(conn):
    """
    Readable names for the sampled ranking slices, in rank order.

    A stratum is stored under a label like US-r03951, which says where it
    came from but not what it is. The ranking it was drawn from and the
    ranks it spans are both on the row already.
    """
    named = {}
    order = []

    for label, country, low, high in conn.execute(
        "select label, country, rank_low, rank_high from Stratum"
    ):
        where = country if country else "global"
        named[label] = f"{where} #{low:,}–{high:,}"
        order.append((country or "", int(low), named[label]))

    order.sort()

    return named, [name for _, _, name in order]


def build(settings):
    """Read the database, fit, and index the result."""
    conn = connect_readonly(str(DATABASE))
    knobs = SimpleNamespace(**settings)

    study = prepare(conn, knobs)
    params, history = fit_panel(study, knobs)

    facts, stars = map_facts(conn, study)
    study.stars = stars
    columns = map_columns(study, params, facts, stars, knobs)

    # No rng, so a score of exactly 100% lands in the middle of the
    # stretch the point mass covers rather than somewhere random in it.
    # A page that moved its own bars between reloads would be unreadable.
    fell_at = predictive_cdf(params, study.panel, study.model)

    named = usernames(conn, study.roster)
    strata, strata_order = stratum_names(conn)
    sd = np.exp(params.skill_log_sd)
    place = normal_cdf(params.skill_mean)
    held = np.bincount(study.panel.rows, minlength=study.panel.n_players)

    players = [
        {
            "index": i,
            "id": int(player),
            "name": named.get(int(player), str(player)),
            "stratum": strata.get(
                study.stratum_of[player], study.stratum_of[player]
            ),
            "skill": round(float(params.skill_mean[i]), 4),
            "sd": round(float(sd[i]), 4),
            "percentile": round(100.0 * float(place[i]), 1),
            "scores": int(held[i]),
            "skillText": f"{float(params.skill_mean[i]):+.2f}",
            "sdText": f"± {float(sd[i]):.2f}",
            "percentileText": as_percentile(100.0 * float(place[i])),
            "url": f"https://osu.ppy.sh/users/{int(player)}",
        }
        for i, player in enumerate(study.roster)
    ]

    maps = []

    for j, key in enumerate(study.items):
        beatmap_id, mods = key.split(":", 1)

        row = {
            "index": j,
            "key": key,
            "beatmap": int(beatmap_id),
            "mods": mods,
            "name": facts.get(int(beatmap_id), {}).get("name", key),
            "players": int(columns["counts"][j]),
            "tells": clean(columns["information"][j], 2),
            "atMedian": clean(columns["at_median"][j], 4),
            "typical": clean(columns["typical"][j], 2),
            "stars": clean(columns["stars"][j], 2),
            "length": clean(columns["length"][j], 0),
            "playcount": clean(columns["playcount"][j], 0),
            "livePP": clean(columns["live"][j], 1),
            "gapStars": clean(columns["from_stars"][j], 1),
            "gapCurve": clean(columns["from_curves"][j], 1),
            "straddles": bool(columns["straddles"][j]),
            "version": facts.get(int(beatmap_id), {}).get("version", ""),
            "url": f"https://osu.ppy.sh/b/{int(beatmap_id)}",
        }

        # A missing number has to reach the page as text, since the page
        # cannot tell None from zero once it is JSON.
        for name in ("tells", "atMedian", "stars", "gapCurve", "length",
                     "playcount", "livePP"):
            row[name + "Text"] = shown(row[name])

        row["atMedianText"] = (
            "—" if row["atMedian"] is None
            else f"{100.0 * row['atMedian']:.2f}%"
        )

        maps.append(row)

    return Fitted(
        settings=settings,
        summary={
            "family": settings["family"],
            "players": study.panel.n_players,
            "items": study.panel.n_items,
            "observations": int(len(study.panel.outcome)),
            "objective": round(float(history[-1]), 1),
            "skillMean": round(float(params.skill_mean.mean()), 3),
            "skillSd": round(float(params.skill_mean.std()), 3),
            "medianWidth": round(float(np.median(sd)), 3),
            "fittedAt": time.strftime("%Y-%m-%d %H:%M"),
        },
        players=players,
        maps=maps,
        strata=strata_order,
        params=params,
        model=study.model,
        score_player=study.panel.rows,
        score_map=study.panel.cols,
        score_outcome=study.panel.outcome,
        score_fell_at=fell_at,
    )


def clean(value, digits):
    if value is None or not np.isfinite(value):
        return None

    return round(float(value), digits) if digits else int(value)


def as_percentile(value):
    """
    A place in the population.

    Whole numbers everywhere except the two ends, where rounding would
    print 100% for every strong player and hide the order among them.
    """
    if value > 99.5 or value < 0.5:
        return f"{value:.2f}%"

    return f"{value:.0f}%"


def shown(value, dash="—"):
    return dash if value is None else f"{value:,}".rstrip()


def load(overrides=None, refit=False):
    """
    The fit for these settings, from the cache when one is there.

    A cached fit is whatever the database held when it ran, so `refit`
    is how you pick up what the sampler has collected since.
    """
    settings = settings_from(overrides)
    path = cache_path(settings)

    if not refit and path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)

    fit = build(settings)

    CACHE.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as handle:
        pickle.dump(fit, handle)

    return fit
