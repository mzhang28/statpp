#!/usr/bin/env python3
"""
Fit one skill number per player and a performance curve per map.

The fit in fit_ability_and_difficulty.py takes pp as the observation and
solves pp = ability + difficulty, then reports where the answer departs
from pp. This one takes the score itself as the observation: accuracy,
turned into a number on the real line. pp is never fitted to, so a
departure from pp is two measurements of different things rather than one
measurement compared against itself.

A player is a Gaussian belief over a skill on the real line, held to a
standard normal population so that Phi(skill) reads as a percentile. A map
is an increasing curve giving expected performance at each skill, and a
separate positive spread giving how much performance varies there. A map
that everyone clears alike has a flat curve, and a flat curve is what makes
a score on it change nothing about the player.

The panel is the probed cells by default. A probe is requested whatever the
player scored, so nothing about the result decided whether the cell is
here; a top-100 list is cut at the player's hundredth-best pp, so it is
selected on the outcome.

Reads through connect_readonly(), so it runs while the sampler writes.
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from diagnose import spearman, trim
from fit_ability_and_difficulty import fit as additive_fit
from fit_ability_and_difficulty import observations
from find_overvalued_maps import expected_pp, rating_for
from sample import connect_readonly

SQRT2 = math.sqrt(2.0)
LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)

FIELDS = (
    "skill_mean",
    "skill_log_sd",
    "curve_floor",
    "curve_rise",
    "spread_level",
    "spread_shape",
)

# A perfect score has no finite number of nines, so the miss fraction is
# floored here. 5e-4 is about half of one 100 on a 700-object map, which
# is below anything a real score short of an SS reaches.
ACCURACY_FLOOR = 5e-4


# ---------------------------------------------------------------------------
# Small numeric pieces


def sigmoid(x):
    """Logistic function, written so large |x| cannot overflow."""
    decay = np.exp(-np.abs(x))

    return np.where(x >= 0.0, 1.0 / (1.0 + decay), decay / (1.0 + decay))


def softplus(x):
    return np.logaddexp(0.0, x)


def inverse_softplus(x):
    return float(np.log(np.expm1(x)))


def normal_cdf(x):
    """
    Phi. numpy has no erf, and math.erf is one value at a time, which is
    fine because this is only used for reporting and for building the
    quantile table below.
    """
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / SQRT2))


def normal_quantile(p, _table={}):
    """Inverse of normal_cdf, by interpolating the curve itself."""
    if not _table:
        grid = np.linspace(-6.0, 6.0, 24001)
        _table["x"] = grid
        _table["p"] = normal_cdf(grid)

    return np.interp(p, _table["p"], _table["x"])


def nines(accuracy):
    """
    Accuracy as a number on the real line: how many nines it has.

    0.9 is one nine, 0.99 is two, 0.999 is three. Raw accuracy piles up
    against 1 and every difference between strong players sits in the last
    fraction of a percent, which this spreads out.
    """
    return -np.log10(np.maximum(1.0 - accuracy, ACCURACY_FLOOR))


def gaussian_log_density(y, mean, sd):
    return -LOG_SQRT_TWO_PI - np.log(sd) - 0.5 * ((y - mean) / sd) ** 2


# ---------------------------------------------------------------------------
# The curve basis


class Basis:
    """
    Soft ramps at fixed knots, which every map curve is built out of.

    Expected performance is a positive combination of ramps, so it rises
    with skill by construction, which is the one thing the curve has to do.
    Its derivative is the matching sum of bumps, so a map can be steep in
    one band of skill and flat elsewhere: weight on a single knot separates
    players around that knot and nowhere else.

    Spread uses the same bumps in the log, so it stays positive and is
    otherwise free to rise and fall across the range.
    """

    def __init__(self, count, reach):
        self.count = count
        self.knots = np.linspace(-reach, reach, count)
        self.width = 2.0 * reach / (count - 1)

    def ramps(self, theta):
        """One column per knot: 0 well below it, 1 well above."""
        return sigmoid(
            (np.asarray(theta)[:, None] - self.knots[None, :]) / self.width
        )


def bumps_of(ramps):
    """Derivative of each ramp, scaled to peak at 1."""
    return 4.0 * ramps * (1.0 - ramps)


@dataclass
class Parameters:
    skill_mean: np.ndarray       # (players,)
    skill_log_sd: np.ndarray     # (players,)
    curve_floor: np.ndarray      # (items,)   curve value below the range
    curve_rise: np.ndarray       # (items, knots)  softplus of this is the rise
    spread_level: np.ndarray     # (items,)   log spread before the bumps
    spread_shape: np.ndarray     # (items, knots)


def curve_values(params, items, theta, basis):
    """Expected performance and spread for each (item, skill) pair."""
    ramps = basis.ramps(theta)

    mean = params.curve_floor[items] + (
        softplus(params.curve_rise[items]) * ramps
    ).sum(1)

    log_spread = params.spread_level[items] + (
        params.spread_shape[items] * bumps_of(ramps)
    ).sum(1)

    return mean, np.exp(log_spread)


def curve_slope(params, items, theta, basis):
    """d(expected performance)/d(skill): how much the map separates here."""
    ramps = basis.ramps(theta)

    return (
        softplus(params.curve_rise[items]) * ramps * (1.0 - ramps)
    ).sum(1) / basis.width


# ---------------------------------------------------------------------------
# The panel


@dataclass
class Panel:
    rows: np.ndarray
    cols: np.ndarray
    outcome: np.ndarray
    n_players: int
    n_items: int

    def take(self, mask):
        return Panel(
            self.rows[mask], self.cols[mask], self.outcome[mask],
            self.n_players, self.n_items,
        )


def load_panel(conn, probed_only):
    """
    Best accuracy per (player, beatmap, mod_key), as nines, plus the pp
    that score was awarded.

    `probed_only` keeps the cells we asked for by name. A probe goes out
    before anyone knows what is on the other end, so nothing about the
    result decided whether the cell is in the panel. Dropping it adds the
    top-100 lists, where a play is visible only if its pp beat the
    player's hundredth best, which is selection on the outcome.

    sqlite returns the other columns from the row that supplied max(), so
    the pp here belongs to the same score as the accuracy.
    """
    probe_join = (
        "join Probe pr on pr.player = s.player and pr.beatmap = s.beatmap"
        if probed_only else ""
    )

    rows = conn.execute(f"""
        select s.player, s.beatmap, s.mod_key, max(s.accuracy), s.pp,
               pl.stratum, pl.pp
        from Score s
        join Player pl on pl.id = s.player
        {probe_join}
        where s.mod_settings = 0 and pl.stratum is not null
        group by s.player, s.beatmap, s.mod_key
    """)

    players = defaultdict(dict)
    stratum_of = {}
    awarded = {}
    player_pp = {}

    for player, beatmap, mods, accuracy, pp, stratum, overall in rows:
        key = f"{beatmap}:{mods}"

        players[player][key] = float(nines(np.array(accuracy)))
        stratum_of[player] = stratum
        player_pp[player] = overall

        if pp is not None:
            awarded[(player, key)] = pp

    return dict(players), stratum_of, player_pp, awarded


def beatmap_facts(conn, beatmap_ids):
    """
    Rating, name, length and playcount for the maps in the panel.

    Only the maps fetched from the beatmap endpoint carry the song they
    belong to. The ones first seen inside a score response do not, and
    those fall back to the beatmapset number and the difficulty name.
    """
    if not beatmap_ids:
        return {}

    holes = ",".join("?" * len(beatmap_ids))

    facts = {}

    for row in conn.execute(
        f"select id, stars, version, length, playcount, raw "
        f"from Beatmap where id in ({holes})",
        list(beatmap_ids),
    ):
        beatmap_id, stars, version, length, playcount, raw = row
        parsed = json.loads(raw)
        listing = parsed.get("beatmapset")

        if listing:
            name = (
                f"{listing.get('artist', '?')} - "
                f"{listing.get('title', '?')} [{version}]"
            )
        else:
            name = f"set {parsed.get('beatmapset_id', '?')} [{version}]"

        facts[int(beatmap_id)] = {
            "stars": stars,
            "version": version,
            "length": length,
            "playcount": playcount,
            "name": name,
        }

    return facts


# ---------------------------------------------------------------------------
# The objective


def objective(params, panel, basis, nodes, weights, settings):
    """
    The variational objective and its gradient.

    Every observation contributes the expected negative log-likelihood
    under the player's own belief rather than at a point estimate, so a
    player the data barely places contributes a smeared-out likelihood
    instead of a confident wrong one. The expectation is a Gauss-Hermite
    sum over the belief, which is exact for the parts of the integrand a
    low-order polynomial covers and close enough for the rest.

    KL to the standard normal prior is what stops the beliefs from
    wandering, and the population term holds the first two moments of the
    fitted population in place: nothing in the likelihood alone prevents
    every skill drifting together while the map curves slide to match, and
    Phi only reads as a percentile if the population really is N(0, 1).

    The map penalty pools each map's curve towards the average curve, since
    no single map here has the players to fit its own shape.
    """
    rows, cols = panel.rows, panel.cols
    n_players, n_items = panel.n_players, panel.n_items

    sd = np.exp(params.skill_log_sd)
    rise = softplus(params.curve_rise)

    grad = {n: np.zeros_like(getattr(params, n)) for n in FIELDS}
    total = 0.0

    for node, weight in zip(nodes, weights):
        theta = params.skill_mean[rows] + SQRT2 * sd[rows] * node

        ramps = basis.ramps(theta)
        bumps = bumps_of(ramps)

        mean = params.curve_floor[cols] + (rise[cols] * ramps).sum(1)
        log_spread = (
            params.spread_level[cols]
            + (params.spread_shape[cols] * bumps).sum(1)
        )
        spread = np.exp(log_spread)

        z = (panel.outcome - mean) / spread
        total += weight * float(np.sum(log_spread + 0.5 * z * z))

        # How this node's loss moves with the two things the map supplies.
        d_mean = -z / spread
        d_log_spread = 1.0 - z * z

        grad["curve_floor"] += weight * np.bincount(cols, d_mean, n_items)
        grad["spread_level"] += weight * np.bincount(
            cols, d_log_spread, n_items
        )

        for k in range(basis.count):
            grad["curve_rise"][:, k] += weight * np.bincount(
                cols, d_mean * ramps[:, k], n_items
            )
            grad["spread_shape"][:, k] += weight * np.bincount(
                cols, d_log_spread * bumps[:, k], n_items
            )

        # Moving the skill moves both the curve and the spread under it.
        curve_gradient = (
            rise[cols] * ramps * (1.0 - ramps)
        ).sum(1) / basis.width
        spread_gradient = (
            params.spread_shape[cols] * bumps * (1.0 - 2.0 * ramps)
        ).sum(1) / basis.width

        d_theta = d_mean * curve_gradient + d_log_spread * spread_gradient

        grad["skill_mean"] += weight * np.bincount(rows, d_theta, n_players)
        grad["skill_log_sd"] += weight * np.bincount(
            rows, d_theta * SQRT2 * node, n_players
        )

    # Up to here the accumulators are in the units the likelihood sees.
    # These two carry them back to the parameters that are optimised.
    grad["curve_rise"] *= sigmoid(params.curve_rise)
    grad["skill_log_sd"] *= sd

    total += float(
        np.sum(
            0.5 * (sd ** 2 + params.skill_mean ** 2 - 1.0)
            - params.skill_log_sd
        )
    )
    grad["skill_mean"] += params.skill_mean
    grad["skill_log_sd"] += sd ** 2 - 1.0

    centre = float(params.skill_mean.mean())
    spread_of_population = float(
        ((params.skill_mean - centre) ** 2).mean() + (sd ** 2).mean()
    )

    total += settings.population * n_players * (
        centre ** 2 + (spread_of_population - 1.0) ** 2
    )
    grad["skill_mean"] += settings.population * (
        2.0 * centre
        + 4.0 * (spread_of_population - 1.0) * (params.skill_mean - centre)
    )
    grad["skill_log_sd"] += (
        settings.population * 4.0 * (spread_of_population - 1.0) * sd ** 2
    )

    for name, strength in (
        ("curve_floor", settings.pool_level),
        ("spread_level", settings.pool_level),
        ("curve_rise", settings.pool_curve),
        ("spread_shape", settings.pool_spread),
    ):
        value = getattr(params, name)
        deviation = value - value.mean(axis=0)

        total += strength * float(np.sum(deviation ** 2))
        grad[name] += 2.0 * strength * deviation

    return total, grad


def initialise(panel, basis):
    """
    A first guess good enough that the fit only has to refine it.

    A two-way additive fit orders players and maps cheaply. Turning each
    player's place in that order into a normal score starts the population
    already standard normal, and the map curves are then set to pass
    through their own mean outcome at the slope the additive fit implies.
    """
    level, item_level, _ = additive_fit(
        panel.rows, panel.cols, panel.outcome,
        panel.n_players, panel.n_items,
        np.arange(panel.n_items), 100, False,
    )

    place = np.argsort(np.argsort(level))
    skill_mean = normal_quantile((place + 0.5) / panel.n_players)

    residual = panel.outcome - level[panel.rows] - item_level[panel.cols]

    # How much a nine of accuracy moves per unit of skill, across all maps.
    skill_at_row = skill_mean[panel.rows]
    slope = float(
        np.dot(skill_at_row, panel.outcome - item_level[panel.cols])
        / np.dot(skill_at_row, skill_at_row)
    )

    at_centre = basis.ramps(np.zeros(1))
    unit_slope = float((at_centre * (1.0 - at_centre)).sum() / basis.width)

    height = max(slope / unit_slope, 1e-3)

    return Parameters(
        skill_mean=skill_mean,
        skill_log_sd=np.full(panel.n_players, math.log(0.5)),
        curve_floor=(
            item_level + level.mean() - height * float(at_centre.sum())
        ),
        curve_rise=np.full(
            (panel.n_items, basis.count), inverse_softplus(height)
        ),
        spread_level=np.full(panel.n_items, math.log(residual.std())),
        spread_shape=np.zeros((panel.n_items, basis.count)),
    )


def descend(params, loss_and_grad, steps, rate, floor, ceiling):
    """
    Adam, with the belief widths held inside a range.

    A width running to zero turns the expectation back into a point
    estimate and the gradient with it, so it is kept off the floor.
    """
    moment = {n: np.zeros_like(getattr(params, n)) for n in FIELDS}
    scale = {n: np.zeros_like(getattr(params, n)) for n in FIELDS}

    history = []

    for step in range(1, steps + 1):
        loss, grad = loss_and_grad(params)
        history.append(loss)

        for name in FIELDS:
            moment[name] = 0.9 * moment[name] + 0.1 * grad[name]
            scale[name] = 0.999 * scale[name] + 0.001 * grad[name] ** 2

            adjusted = moment[name] / (1.0 - 0.9 ** step)
            spread = scale[name] / (1.0 - 0.999 ** step)

            setattr(
                params, name,
                getattr(params, name)
                - rate * adjusted / (np.sqrt(spread) + 1e-8),
            )

        params.skill_log_sd = np.clip(
            params.skill_log_sd, math.log(floor), math.log(ceiling)
        )

    return params, history


# ---------------------------------------------------------------------------
# What one score does to a player


def score_derivatives(params, item, y, theta, basis):
    """
    Slope and curvature of one score's log-likelihood in the player's skill.

    g says which way the result disagrees with where the player sits and
    how strongly. h says how much the map and the result reveal about skill
    at all: a map whose curve is flat here gives both near zero.

    h is a curvature rather than a count of information, so it can come out
    negative where the curve bends the wrong way under the residual, and
    the caller has to handle that.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    item = np.atleast_1d(np.asarray(item))
    y = np.atleast_1d(np.asarray(y, dtype=float))

    ramps = basis.ramps(theta)
    bumps = bumps_of(ramps)

    rise = softplus(params.curve_rise[item])
    shape = params.spread_shape[item]
    width = basis.width

    mean = params.curve_floor[item] + (rise * ramps).sum(1)
    mean_slope = (rise * ramps * (1.0 - ramps)).sum(1) / width
    mean_bend = (
        rise * ramps * (1.0 - ramps) * (1.0 - 2.0 * ramps)
    ).sum(1) / width ** 2

    log_spread = params.spread_level[item] + (shape * bumps).sum(1)
    spread_slope = (shape * bumps * (1.0 - 2.0 * ramps)).sum(1) / width
    spread_bend = (
        shape * bumps * (1.0 - 6.0 * ramps + 6.0 * ramps ** 2)
    ).sum(1) / width ** 2

    residual = y - mean
    variance = np.exp(2.0 * log_spread)

    g = -spread_slope + (
        residual * mean_slope + residual ** 2 * spread_slope
    ) / variance

    second = -spread_bend + (
        -mean_slope ** 2
        + residual * mean_bend
        - 4.0 * residual * mean_slope * spread_slope
        + residual ** 2 * spread_bend
        - 2.0 * residual ** 2 * spread_slope ** 2
    ) / variance

    return g, -second


def laplace_step(mean, sd, g, h, keep=0.05):
    """
    One score folded into a player's belief without refitting.

    Returns the new mean and width, and which steps were taken. A negative
    curvature can drive the new precision to zero or below, which is not a
    belief at all, and even short of that a single score widening the
    belief severalfold is the approximation leaving the range it holds in.
    Anything left with less than `keep` of the precision it started with is
    refused, and the belief stands unchanged.
    """
    precision = 1.0 / sd ** 2 + h
    taken = precision > keep / sd ** 2

    safe = np.maximum(precision, 1e-12)

    return (
        np.where(taken, mean + g / safe, mean),
        np.where(taken, 1.0 / np.sqrt(safe), sd),
        taken,
    )


# ---------------------------------------------------------------------------
# Prediction


def predictive(params, panel, basis, nodes, weights):
    """
    Density the model puts on each outcome, and its expected value.

    The player's skill is a belief rather than a number, so this averages
    the map's density over that belief instead of reading it at a point.
    """
    sd = np.exp(params.skill_log_sd)

    density = np.zeros(len(panel.outcome))
    expected = np.zeros(len(panel.outcome))

    for node, weight in zip(nodes, weights):
        theta = params.skill_mean[panel.rows] + SQRT2 * sd[panel.rows] * node
        mean, spread = curve_values(params, panel.cols, theta, basis)

        density += weight * np.exp(
            gaussian_log_density(panel.outcome, mean, spread)
        )
        expected += weight * mean

    return np.log(np.maximum(density, 1e-300)), expected


def predictive_cdf(params, panel, basis, nodes, weights):
    """
    Where each outcome falls inside the distribution predicted for it: 0
    when it is far below what the map and the player's skill lead you to
    expect, 1 when it is far above, 0.5 when it lands exactly there.

    If the model is right these come out uniform on (0, 1), whatever the
    map and whoever the player, so their histogram is a calibration check
    that needs no held-out data.
    """
    sd = np.exp(params.skill_log_sd)
    total = np.zeros(len(panel.outcome))

    for node, weight in zip(nodes, weights):
        theta = params.skill_mean[panel.rows] + SQRT2 * sd[panel.rows] * node
        mean, spread = curve_values(params, panel.cols, theta, basis)

        total += weight * normal_cdf((panel.outcome - mean) / spread)

    return total


def score_predictions(name, y, log_density, prediction):
    return (
        name,
        float(np.mean(log_density)),
        float(np.sqrt(np.mean((y - prediction) ** 2))),
    )


# ---------------------------------------------------------------------------


def check_gradient(params, loss_and_grad, rng, samples=40):
    """Analytic gradient against a central difference at random entries."""
    _, grad = loss_and_grad(params)

    worst = 0.0
    step = 1e-5

    for _ in range(samples):
        name = FIELDS[rng.integers(len(FIELDS))]
        value = getattr(params, name)
        spot = tuple(rng.integers(n) for n in value.shape)

        keep = value[spot]

        value[spot] = keep + step
        up, _ = loss_and_grad(params)

        value[spot] = keep - step
        down, _ = loss_and_grad(params)

        value[spot] = keep

        numeric = (up - down) / (2.0 * step)
        analytic = grad[name][spot]

        scale = max(abs(numeric), abs(analytic), 1.0)
        worst = max(worst, abs(numeric - analytic) / scale)

    return worst


def check_score_derivatives(params, basis, rng, samples=200):
    """
    The slope and curvature one score reports, against differences of the
    log-likelihood it is the slope and curvature of.

    The incremental update rests entirely on these two numbers and nothing
    else in the fit exercises them, so they are checked separately.
    """
    n_items = params.curve_floor.shape[0]

    def log_likelihood(item, y, theta):
        mean, spread = curve_values(
            params, np.array([item]), np.array([theta]), basis
        )

        return float(gaussian_log_density(np.array([y]), mean, spread)[0])

    step = 1e-4
    worst_slope = worst_curvature = 0.0

    for _ in range(samples):
        item = int(rng.integers(n_items))
        y = float(rng.normal(1.4, 0.4))
        theta = float(rng.normal(0.0, 1.2))

        g, h = score_derivatives(
            params, np.array([item]), np.array([y]), np.array([theta]), basis
        )

        here = log_likelihood(item, y, theta)
        above = log_likelihood(item, y, theta + step)
        below = log_likelihood(item, y, theta - step)

        numeric_slope = (above - below) / (2.0 * step)
        numeric_curvature = -(above - 2.0 * here + below) / step ** 2

        worst_slope = max(
            worst_slope,
            abs(numeric_slope - g[0]) / max(abs(numeric_slope), 1.0),
        )
        worst_curvature = max(
            worst_curvature,
            abs(numeric_curvature - h[0]) / max(abs(numeric_curvature), 1.0),
        )

    return worst_slope, worst_curvature


class NotEnoughData(Exception):
    """The database does not hold a panel worth fitting yet."""


@dataclass
class Study:
    """The panel to fit, and everything that indexes it."""

    panel: Panel
    roster: list
    items: list
    stratum_of: dict
    player_pp: dict
    awarded: dict
    basis: Basis
    nodes: np.ndarray
    weights: np.ndarray


def prepare(conn, settings):
    """
    Read the panel, drop the thin edges of it, and set up the quadrature
    the player beliefs are integrated with.
    """
    held, stratum_of, player_pp, awarded = load_panel(
        conn, not settings.all_cells
    )

    if not held:
        raise NotEnoughData("No scores in the panel yet.")

    core = trim(held, settings.min_players, settings.min_items)

    if len(core) < 2:
        raise NotEnoughData(
            f"Nothing survives {settings.min_players} players x "
            f"{settings.min_items} items. Keep filling."
        )

    roster, items, rows, cols, outcome = observations(core)
    nodes, weights = np.polynomial.hermite.hermgauss(settings.quadrature)

    return Study(
        panel=Panel(rows, cols, outcome, len(roster), len(items)),
        roster=roster,
        items=items,
        stratum_of=stratum_of,
        player_pp=player_pp,
        awarded=awarded,
        basis=Basis(settings.knots, settings.reach),
        nodes=nodes,
        weights=weights / math.sqrt(math.pi),
    )


def fit_panel(study, settings, panel=None):
    """Fit skill beliefs and map curves to a panel, or a subset of one."""
    panel = study.panel if panel is None else panel

    def loss_and_grad(params):
        return objective(
            params, panel, study.basis, study.nodes, study.weights, settings
        )

    return descend(
        initialise(panel, study.basis), loss_and_grad,
        settings.steps, settings.rate, 0.02, 3.0,
    )


def map_facts(conn, study):
    """
    What the official numbers say about each map, indexed like the panel.

    The stored rating is the unmodded one and DT moves a map by two stars
    or more, so each item is rated as it was actually played.
    """
    facts = beatmap_facts(
        conn, sorted({int(k.split(":", 1)[0]) for k in study.items})
    )

    stars = np.full(study.panel.n_items, np.nan)

    for j, key in enumerate(study.items):
        beatmap_id, mods = key.split(":", 1)
        beatmap_id = int(beatmap_id)

        if beatmap_id not in facts:
            continue

        # rating_for wants the stored unmodded rating to fall back on when
        # the map file for a modded calculation is missing.
        rating = rating_for(
            beatmap_id, mods, {beatmap_id: (facts[beatmap_id]["stars"],)}
        )

        if rating is not None:
            stars[j] = rating

    return facts, stars


def map_columns(study, params, facts, stars, settings):
    """
    One row per map: how steeply its curve rises, how hard it comes out,
    what pp it pays, and how far that pp sits from what maps of the same
    difficulty pay.
    """
    panel, basis = study.panel, study.basis
    index = np.arange(panel.n_items)

    skills = defaultdict(list)

    for row, col in zip(panel.rows, panel.cols):
        skills[col].append(params.skill_mean[row])

    typical = np.array([np.median(skills[j]) for j in index])
    below = np.array(
        [float(np.mean(np.array(skills[j]) < 0.0)) for j in index]
    )

    # What each map pays a player of average level. The raw mean pp on a
    # map would instead say who happened to be probed on it, which is the
    # thing the length and playcount check is testing for.
    paid = np.array(
        [study.awarded.get((study.roster[i], study.items[j]), np.nan)
         for i, j in zip(panel.rows, panel.cols)]
    )
    priced = np.isfinite(paid)

    _, live, _ = additive_fit(
        panel.rows[priced], panel.cols[priced], paid[priced],
        panel.n_players, panel.n_items, index, 200, False,
    )
    live[np.bincount(panel.cols[priced], minlength=panel.n_items) == 0] = np.nan

    at_median = curve_values(
        params, index, np.zeros(panel.n_items), basis
    )[0]

    def about(field):
        return np.array([
            facts.get(int(k.split(":", 1)[0]), {}).get(field) or np.nan
            for k in study.items
        ])

    def discrepancy(axis, window):
        """
        How much more pp a map pays than maps of the same difficulty do.

        Each map is compared against its own neighbours on the axis rather
        than against a curve fitted across the whole range, since the ends
        of the range are where the sparsest maps sit.
        """
        ok = np.isfinite(axis) & np.isfinite(live)

        gap = np.full(panel.n_items, np.nan)
        expected, _ = expected_pp(
            axis[ok], live[ok], window, settings.min_neighbours
        )
        gap[ok] = live[ok] - expected

        return gap

    return {
        "counts": np.array([len(skills[j]) for j in index]),
        "typical": typical,
        "straddles": (below > 0.2) & (below < 0.8),
        "slope": curve_slope(params, index, typical, basis),
        "slope_at_centre": curve_slope(
            params, index, np.zeros(panel.n_items), basis
        ),
        "at_median": at_median,
        "raw_mean": np.bincount(
            panel.cols, panel.outcome, panel.n_items
        ) / np.maximum(np.bincount(panel.cols, minlength=panel.n_items), 1),
        "stars": stars,
        "length": about("length"),
        "playcount": about("playcount"),
        "live": live,
        "from_stars": discrepancy(stars, settings.star_window),
        "from_curves": discrepancy(-at_median, settings.skill_window),
    }


def usernames(conn, player_ids):
    holes = ",".join("?" * len(player_ids))

    return {
        int(row[0]): row[1]
        for row in conn.execute(
            f"select id, username from Player where id in ({holes})",
            list(player_ids),
        )
    }


def population_report(skill):
    """Moments and the largest gap between the fitted skills and N(0, 1)."""
    centred = (skill - skill.mean()) / skill.std()

    order = np.sort(skill)
    place = (np.arange(len(order)) + 0.5) / len(order)
    gap = float(np.max(np.abs(normal_cdf(order) - place)))

    return {
        "mean": float(skill.mean()),
        "sd": float(skill.std()),
        "skew": float((centred ** 3).mean()),
        "kurtosis": float((centred ** 4).mean() - 3.0),
        "gap": gap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--db", default=None)

    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="add the top-100 lists, which are cut on pp, to the probed panel",
    )

    parser.add_argument("--min-players", type=int, default=10)
    parser.add_argument("--min-items", type=int, default=5)

    parser.add_argument("--knots", type=int, default=6)
    parser.add_argument(
        "--reach",
        type=float,
        default=2.0,
        help="skill the outermost knots sit at",
    )
    parser.add_argument("--quadrature", type=int, default=7)

    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--rate", type=float, default=0.05)

    parser.add_argument(
        "--population",
        type=float,
        default=1.0,
        help="strength of the hold on the fitted population's moments",
    )
    parser.add_argument(
        "--pool-curve",
        type=float,
        default=2.0,
        help="how hard each map's curve shape is pulled towards the average "
             "shape, which is what decides whether maps are allowed to "
             "differ in how steeply they separate players",
    )
    parser.add_argument(
        "--pool-spread",
        type=float,
        default=50.0,
        help="the same for the spread. Weak values let a map with few "
             "players shrink its spread onto its own scores and claim a "
             "density no held-out score can match",
    )
    parser.add_argument("--pool-level", type=float, default=0.05)

    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=12)

    parser.add_argument(
        "--star-window",
        type=float,
        default=0.5,
        help="star window for the map each map is compared against",
    )
    parser.add_argument(
        "--skill-window",
        type=float,
        default=0.25,
        help="the same window in fitted difficulty",
    )
    parser.add_argument("--min-neighbours", type=int, default=8)

    parser.add_argument("--check-gradient", action="store_true")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    conn = connect_readonly(args.db)

    try:
        study = prepare(conn, args)
    except NotEnoughData as problem:
        print(problem)
        return

    panel, roster, items = study.panel, study.roster, study.items
    stratum_of, player_pp = study.stratum_of, study.player_pp
    basis, nodes, weights = study.basis, study.nodes, study.weights
    outcome = panel.outcome

    print(
        f"panel: {'probed cells only' if not args.all_cells else 'every cell'}"
        f", {panel.n_players} players, {panel.n_items} items, "
        f"{len(outcome)} observations"
    )
    print(
        f"outcome: nines of accuracy, "
        f"{np.quantile(outcome, 0.1):.2f} to {np.quantile(outcome, 0.9):.2f} "
        f"over the middle four fifths, "
        f"{100 * np.mean(outcome >= nines(np.array(1.0))):.1f}% at the ceiling"
    )
    print(
        f"curve: {args.knots} knots from {-args.reach:+.1f} to "
        f"{args.reach:+.1f} skill, {args.quadrature}-node quadrature"
    )

    holdout = rng.random(len(outcome)) < args.holdout
    training = ~holdout

    def loss_and_grad_on(subset):
        return lambda p: objective(
            p, subset, basis, nodes, weights, args
        )

    if args.check_gradient:
        start = initialise(panel, basis)
        start.skill_mean += rng.normal(0.0, 0.1, panel.n_players)
        start.spread_shape += rng.normal(0.0, 0.1, start.spread_shape.shape)

        worst = check_gradient(start, loss_and_grad_on(panel), rng)
        slope_error, curvature_error = check_score_derivatives(
            start, basis, rng
        )

        print(
            f"against central differences, worst relative error: "
            f"objective gradient {worst:.1e}, "
            f"one score's slope {slope_error:.1e}, "
            f"its curvature {curvature_error:.1e}"
        )

    print()
    print("fitting on the training cells")

    train_panel = panel.take(training)
    params = initialise(train_panel, basis)
    params, history = descend(
        params, loss_and_grad_on(train_panel),
        args.steps, args.rate, 0.02, 3.0,
    )

    settled = history[max(0, len(history) - 50)] - history[-1]

    print(
        f"  objective {history[0]:.0f} -> {history[-1]:.0f}, "
        f"still moving {settled:.2f} over the last 50 steps"
    )

    # ---- holdout, against the models that read pp instead

    test = panel.take(holdout)

    facts, modded_stars = map_facts(conn, study)

    overall = np.array(
        [player_pp[p] if player_pp[p] else np.nan for p in roster]
    )

    # Every model is scored on the same cells, so the comparison is not
    # between different subsets of the holdout.
    usable = (
        np.isfinite(modded_stars[test.cols])
        & np.isfinite(overall[test.rows])
        & (overall[test.rows] > 0)
    )
    test = test.take(usable)

    train_scored = train_panel.take(
        np.isfinite(modded_stars[train_panel.cols])
        & np.isfinite(overall[train_panel.rows])
        & (overall[train_panel.rows] > 0)
    )

    rankings = []

    mean_y = float(train_panel.outcome.mean())
    sd_y = float(train_panel.outcome.std())
    rankings.append(score_predictions(
        "one number for everything",
        test.outcome,
        gaussian_log_density(test.outcome, mean_y, sd_y),
        np.full(len(test.outcome), mean_y),
    ))

    ability, difficulty, _ = additive_fit(
        train_panel.rows, train_panel.cols, train_panel.outcome,
        panel.n_players, panel.n_items, np.arange(panel.n_items), 200, False,
    )
    additive_spread = float(
        (
            train_panel.outcome
            - ability[train_panel.rows]
            - difficulty[train_panel.cols]
        ).std()
    )
    additive_prediction = ability[test.rows] + difficulty[test.cols]
    rankings.append(score_predictions(
        "player offset + map offset",
        test.outcome,
        gaussian_log_density(
            test.outcome, additive_prediction, additive_spread
        ),
        additive_prediction,
    ))

    # The same offsets with a width per map, which is the part of the curve
    # model that is not the curve. It separates what the shape earns from
    # what merely noticing that maps differ in spread earns.
    left = (
        train_panel.outcome
        - ability[train_panel.rows]
        - difficulty[train_panel.cols]
    )
    seen = np.bincount(train_panel.cols, minlength=panel.n_items)
    scatter = np.bincount(train_panel.cols, left ** 2, panel.n_items)
    per_map_spread = np.sqrt(
        (scatter + 5.0 * additive_spread ** 2) / (seen + 5.0)
    )
    rankings.append(score_predictions(
        "the same, spread per map",
        test.outcome,
        gaussian_log_density(
            test.outcome, additive_prediction, per_map_spread[test.cols]
        ),
        additive_prediction,
    ))

    def pp_design(subset):
        strength = np.log10(overall[subset.rows])
        stars = modded_stars[subset.cols]

        return np.column_stack(
            [np.ones(len(stars)), strength, stars, strength * stars]
        )

    design = pp_design(train_scored)
    coefficients, *_ = np.linalg.lstsq(
        design, train_scored.outcome, rcond=None
    )
    pp_spread = float((train_scored.outcome - design @ coefficients).std())
    pp_prediction = pp_design(test) @ coefficients
    rankings.append(score_predictions(
        "player pp + map star rating",
        test.outcome,
        gaussian_log_density(test.outcome, pp_prediction, pp_spread),
        pp_prediction,
    ))

    # Reading the curve at the player's mean rather than averaging over
    # the belief, so the cost of carrying the belief is visible on its own.
    point_mean, point_spread = curve_values(
        params, test.cols, params.skill_mean[test.rows], basis
    )
    rankings.append(score_predictions(
        "skill point + map curve",
        test.outcome,
        gaussian_log_density(test.outcome, point_mean, point_spread),
        point_mean,
    ))

    log_density, prediction = predictive(params, test, basis, nodes, weights)
    rankings.append(score_predictions(
        "skill belief + map curve",
        test.outcome, log_density, prediction,
    ))

    print()
    print(
        f"predicting {len(test.outcome)} held-out cells "
        f"({100 * args.holdout:.0f}% of the panel, in nines of accuracy)"
    )
    print()
    print(f"{'model':<30}{'log density':>13}{'rmse':>8}")
    print("-" * 51)

    for name, density, error in rankings:
        print(f"{name:<30}{density:>13.3f}{error:>8.3f}")

    print()
    print()
    print(
        "log density is per cell and higher is better. The pp row is what "
        "the\nofficial numbers know about a cell without seeing the score, "
        "which is\nthe comparison the model has to win."
    )

    # ---- the fit on everything, which is what the maps are read off

    print()
    print("refitting on the whole panel")

    params = initialise(panel, basis)
    params, history = descend(
        params, loss_and_grad_on(panel), args.steps, args.rate, 0.02, 3.0
    )

    print(f"  objective {history[0]:.0f} -> {history[-1]:.0f}")

    stats = population_report(params.skill_mean)

    print()
    print("the fitted population of skills, which Phi reads as a percentile")
    print()
    print(
        f"  mean {stats['mean']:+.3f}, sd {stats['sd']:.3f}, "
        f"skew {stats['skew']:+.2f}, excess kurtosis {stats['kurtosis']:+.2f}"
    )
    print(
        f"  largest gap from the standard normal: {stats['gap']:.3f} "
        f"of the population"
    )
    print(
        f"  belief widths: median "
        f"{np.median(np.exp(params.skill_log_sd)):.2f}, "
        f"widest {np.exp(params.skill_log_sd).max():.2f}"
    )

    by_stratum = defaultdict(list)

    for i, player in enumerate(roster):
        by_stratum[stratum_of[player]].append(params.skill_mean[i])

    print()
    print(
        f"{'stratum':<14}{'players':>9}{'median skill':>14}"
        f"{'percentile':>12}"
    )
    print("-" * 49)

    for label in sorted(by_stratum, key=lambda s: -np.median(by_stratum[s])):
        middle = float(np.median(by_stratum[label]))
        print(
            f"{label:<14}{len(by_stratum[label]):>9}{middle:>14.2f}"
            f"{100 * float(normal_cdf(middle)):>11.0f}%"
        )

    # ---- how steep each map is

    columns = map_columns(study, params, facts, modded_stars, args)

    slope_where_played = columns["slope"]
    slope_at_centre = columns["slope_at_centre"]
    typical_skill = columns["typical"]
    counts = columns["counts"]

    def show_maps(order, title):
        print()
        print(title)
        print()
        print(
            f"{'item':<16}{'players':>8}{'slope':>8}{'at 50th':>9}"
            f"{'skill played':>14}  map"
        )
        print("-" * 96)

        for j in order:
            beatmap_id = int(items[j].split(":", 1)[0])
            name = facts.get(beatmap_id, {}).get("name", "?")

            print(
                f"{items[j]:<16}{counts[j]:>8}"
                f"{slope_where_played[j]:>8.2f}{slope_at_centre[j]:>9.2f}"
                f"{typical_skill[j]:>14.2f}  {name[:40]}"
            )

    ranked = np.argsort(slope_where_played)

    print()
    print(
        f"how steeply a map separates players, where its own players sit: "
        f"{slope_where_played.min():.2f} to {slope_where_played.max():.2f} "
        f"nines per\nunit of skill, middle four fifths "
        f"{np.quantile(slope_where_played, 0.1):.2f} to "
        f"{np.quantile(slope_where_played, 0.9):.2f}. That range is what "
        f"the two\ntables below sort on."
    )

    show_maps(
        ranked[:args.top],
        "maps whose curve is flattest where their players sit, so a score "
        "there\nsays least about skill",
    )
    show_maps(
        ranked[::-1][:args.top],
        "and the steepest, which separate the players who set them",
    )

    reol = [j for j, key in enumerate(items) if key.startswith("713818:")]

    if reol:
        print()
        print("Reol - No title [byfaR's Hard], the map pp cannot tell anyone")
        print("apart on:")
        print()

        for j in reol:
            print(
                f"  {items[j]:<16} {counts[j]:>4} players, "
                f"slope {slope_where_played[j]:.2f} where they sit, "
                f"{slope_at_centre[j]:.2f} at the 50th percentile"
            )
    else:
        print()
        print("Reol - No title [byfaR's Hard] is not in the panel.")

    # ---- what one score does, and what it does on a flat map

    print()
    print("folding one held-out score into a player's belief")

    step_panel = panel.take(holdout)

    g, h = score_derivatives(
        params, step_panel.cols, step_panel.outcome,
        params.skill_mean[step_panel.rows], basis,
    )
    prior_sd = np.exp(params.skill_log_sd[step_panel.rows])
    moved, tightened, taken = laplace_step(
        params.skill_mean[step_panel.rows], prior_sd, g, h
    )

    shift = np.abs(moved - params.skill_mean[step_panel.rows])
    cut = 1.0 - tightened / prior_sd

    flat = slope_where_played[step_panel.cols]
    thirds = np.quantile(flat, [1 / 3, 2 / 3])

    print()
    print(
        f"  {int((~taken).sum())} of {len(taken)} steps were refused for "
        f"leaving the belief\n  wider than the approximation supports; "
        f"{int((h < 0).sum())} had a negative curvature"
    )
    print(
        f"  median skill move {np.median(shift[taken]):.3f}, "
        f"median width cut {100 * np.median(cut[taken]):.1f}%"
    )
    print()
    print(
        f"{'map steepness':<20}{'scores':>8}{'median h':>11}"
        f"{'median move':>13}{'width cut':>11}"
    )
    print("-" * 63)

    for name, mask in (
        ("flattest third", flat <= thirds[0]),
        ("middle third", (flat > thirds[0]) & (flat <= thirds[1])),
        ("steepest third", flat > thirds[1]),
    ):
        both = mask & taken
        print(
            f"{name:<20}{int(mask.sum()):>8}{np.median(h[mask]):>11.2f}"
            f"{np.median(shift[both]):>13.3f}"
            f"{100 * np.median(cut[both]):>10.1f}%"
        )

    print()
    print(
        f"  across all held-out scores, how much a score reveals about "
        f"skill tracks\n  how steep the map is at "
        f"{spearman(list(flat), list(h)):+.2f} rank correlation, which "
        f"is the\n  claim that a flat map stops counting"
    )

    # ---- what the discrepancy against pp still tracks

    # Expected performance at the median player. Lower means harder, and
    # it is the fitted stand-in for a star rating.
    at_median = columns["at_median"]
    live = columns["live"]
    length, playcount = columns["length"], columns["playcount"]
    raw_mean = columns["raw_mean"]

    print()
    print("what the fitted difficulty axis is made of")
    print()

    def against(name, values):
        ok = (
            np.isfinite(values)
            & np.isfinite(modded_stars)
            & np.isfinite(playcount)
        )
        here = list(values[ok])

        by_stars = spearman(list(modded_stars[ok]), here)
        by_plays = spearman(list(np.log10(playcount[ok])), here)

        print(f"{name:<34}{by_stars:>+10.2f}{by_plays:>+14.2f}")

    print(f"{'per-map quantity':<34}{'vs stars':>10}{'vs playcount':>14}")
    print("-" * 58)

    against("expected nines at median skill", at_median)
    against("mean nines observed", raw_mean)
    against("pp the map pays average level", live)
    against("star rating", modded_stars)

    print()
    print(
        f"  expected nines at median skill runs "
        f"{np.quantile(at_median, 0.1):.2f} to "
        f"{np.quantile(at_median, 0.9):.2f} over the middle four fifths, "
        f"against\n  {modded_stars[np.isfinite(modded_stars)].min():.1f} to "
        f"{modded_stars[np.isfinite(modded_stars)].max():.1f} stars"
    )

    from_stars = columns["from_stars"]
    from_curves = columns["from_curves"]
    straddles = columns["straddles"]

    print()
    print(
        f"  {int(np.isfinite(from_stars).sum())} maps have enough neighbours "
        f"by star rating, "
        f"{int(np.isfinite(from_curves).sum())} by fitted difficulty"
    )
    print(
        f"  {int((np.isfinite(from_curves) & straddles).sum())} of those have "
        f"players on both sides of the median,\n  where the curve is read "
        f"rather than extrapolated"
    )

    print()
    print("what the discrepancy against pp still tracks")
    print()
    print(
        f"{'difficulty measured by':<26}{'maps':>6}{'length':>10}"
        f"{'playcount':>12}"
    )
    print("-" * 54)

    for name, gap, restrict in (
        ("star rating", from_stars, False),
        ("fitted curve", from_curves, False),
        ("fitted curve, straddling", from_curves, True),
    ):
        ok = np.isfinite(gap) & np.isfinite(length) & np.isfinite(playcount)

        if restrict:
            ok = ok & straddles

        if ok.sum() < 10:
            print(f"{name:<26}{int(ok.sum()):>6}{'too few':>22}")
            continue

        print(
            f"{name:<26}{int(ok.sum()):>6}"
            f"{spearman(list(np.log10(length[ok])), list(gap[ok])):>+10.2f}"
            f"{spearman(list(np.log10(playcount[ok])), list(gap[ok])):>+12.2f}"
        )

    print()
    print(
        "rank correlation of the pp discrepancy with map length and with "
        "how\noften the map is played. The first row is the pp-based fit's "
        "measure\nand the rest are the curve fit's. Towards zero is the "
        "correction\nworking."
    )


if __name__ == "__main__":
    main()
