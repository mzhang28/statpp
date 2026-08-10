#!/usr/bin/env python3
"""
Fit one skill number per player and a set of outcome curves per map.

The fit in fit_ability_and_difficulty.py takes pp as the observation and
solves pp = ability + difficulty, then reports where the answer departs
from pp. This one takes the score itself: the accuracy, exactly as it was
set. pp is never fitted to, so a departure from pp is two measurements of
different things rather than one measurement compared against itself.

A player is a Gaussian belief over a skill on the real line, held to a
standard normal population so that Phi(skill) reads as a percentile. A map
is a conditional distribution over accuracy at each skill, from one of the
families in outcomes.py. Which family it is does not reach this file's
skill side: the objective asks for a log density and how it moves with
each channel, and nothing else.

The map's difficulty is that whole distribution. How much a map separates
players is the Fisher information a single score there carries about
skill, which is a number the map produces rather than the slope of a
scale someone chose.

The panel is every score in the database: the probed cells, and the
top-100 lists the sampler expanded. Those two are not observed the same
way, and which cells appear at all is a question this file does not
answer. It fits what is there.

Reads through connect_readonly(), so it runs while the sampler writes.
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numba import get_num_threads, njit, prange

from diagnose import spearman, trim
from fit_ability_and_difficulty import fit as additive_fit
from fit_ability_and_difficulty import observations
from find_overvalued_maps import expected_pp, rating_for
from outcomes import (
    FAMILIES,
    INSIDE,
    LINK_CODES,
    inverse_softplus,
    logit,
    normal_cdf,
    normal_quantile,
    one_sigmoid,
    sigmoid,
    softplus,
)
from sample import connect_readonly

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# The curve basis


class Basis:
    """
    Soft ramps at fixed knots, which every channel curve is built out of.

    A channel that must rise with skill is a positive combination of
    ramps, so it rises by construction. Its derivative is the matching sum
    of bumps, so a map can move in one band of skill and sit still
    elsewhere: weight on a single knot separates players around that knot
    and nowhere else.

    A channel free to move either way is a combination of the same bumps,
    with coefficients of any sign, added to a level. Outside the knots it
    settles back to that level, which is the right behaviour for something
    with no reason to trend.
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


# ---------------------------------------------------------------------------
# Channels as functions of skill


@dataclass
class Parameters:
    """
    Everything the fit moves.

    The first two arrays are one number per player. The other two are the
    map side, indexed by channel first, and the channel axis is in the
    order the family declares its channels. That is the same order its
    compiled kernel reads, so a channel is an index here and a name only
    at the edges.

    `level` is what a channel is worth below the whole skill range for a
    rising channel, and what it settles back to for a free one. `shape` is
    what the knots add on top: pre-softplus rise weights for a rising or
    falling channel, and bump coefficients of any sign for a free one.
    """

    skill_mean: np.ndarray       # (players,)
    skill_log_sd: np.ndarray     # (players,)
    level: np.ndarray            # (channels, items)
    shape: np.ndarray            # (channels, items, knots)

    @classmethod
    def zeros_like(cls, other):
        return cls(*(np.zeros_like(getattr(other, name)) for name in FIELDS))


FIELDS = ("skill_mean", "skill_log_sd", "level", "shape")


def make_objective(kernel):
    """
    Compile the fit's inner loop around one family's kernel.

    Everything the loop needs is a plain array, so the whole calculation
    for one observation stays in registers: the ramps at its player's
    skill, every channel of its map, the log density, and the gradient
    pushed back onto both sides. numpy would build an array of each of
    those intermediates across every observation at once, and the traffic
    that costs is most of the run time.

    Observations are split into one block per thread, each with its own
    gradient buffer, because two threads adding to the same map would
    otherwise race. The caller sums the blocks.

    `rise` is the shape already put through its link, so the sign of a
    falling channel and the softplus of a rising one are applied once per
    (channel, item, knot) rather than once per observation. The matching
    factor on the way back out is applied by the caller for the same
    reason.
    """
    @njit(cache=False, parallel=True)
    def run(rows, cols, x, level, rise, free, mean, sd, nodes, weights,
            knots, width, bounds,
            g_level, g_shape, g_mean, g_log_sd):
        channels = level.shape[0]
        count = knots.shape[0]
        loss = 0.0

        for block in prange(bounds.shape[0] - 1):
            ramp = np.empty(count)
            bump = np.empty(count)
            value = np.empty(channels)
            slope = np.empty(channels)
            grad = np.empty(channels)

            here = 0.0

            for q in range(nodes.shape[0]):
                node = nodes[q]
                weight = weights[q]

                for i in range(bounds[block], bounds[block + 1]):
                    player = rows[i]
                    item = cols[i]

                    theta = mean[player] + SQRT2 * sd[player] * node

                    for k in range(count):
                        climb = one_sigmoid((theta - knots[k]) / width)
                        ramp[k] = climb
                        bump[k] = 4.0 * climb * (1.0 - climb)

                    for c in range(channels):
                        total = level[c, item]
                        moved = 0.0

                        if free[c]:
                            for k in range(count):
                                total += rise[c, item, k] * bump[k]
                                moved += rise[c, item, k] * bump[k] * (
                                    1.0 - 2.0 * ramp[k]
                                )
                        else:
                            for k in range(count):
                                total += rise[c, item, k] * ramp[k]
                                moved += rise[c, item, k] * ramp[k] * (
                                    1.0 - ramp[k]
                                )

                        value[c] = total
                        slope[c] = moved / width

                    here -= weight * kernel(x[i], value, grad)

                    d_theta = 0.0

                    for c in range(channels):
                        # The loss is the negative log density, so every
                        # push below is the family's gradient turned around.
                        push = -weight * grad[c]

                        g_level[block, c, item] += push
                        d_theta += push * slope[c]

                        if free[c]:
                            for k in range(count):
                                g_shape[block, c, item, k] += push * bump[k]
                        else:
                            for k in range(count):
                                g_shape[block, c, item, k] += push * ramp[k]

                    g_mean[block, player] += d_theta
                    g_log_sd[block, player] += d_theta * SQRT2 * node

            loss += here

        return loss

    return run


def make_scorer(kernel):
    """
    Compile a sweep that reads a log density and its channel gradients for
    every observation at once, with the channel values handed in.

    The objective builds its channel values from a skill and a spline. The
    models that use neither still need the same arithmetic over the same
    number of scores, so they share the kernel and do their own reduction
    afterwards, which is a bincount or one matrix product.
    """
    @njit(cache=False, parallel=True)
    def run(x, values, grad, bounds):
        channels = values.shape[0]
        total = 0.0

        for block in prange(bounds.shape[0] - 1):
            value = np.empty(channels)
            here = np.empty(channels)
            mine = 0.0

            for i in range(bounds[block], bounds[block + 1]):
                for c in range(channels):
                    value[c] = values[c, i]

                mine += kernel(x[i], value, here)

                for c in range(channels):
                    grad[c, i] = here[c]

            total += mine

        return total

    return run


COMPILED = {}
SCORERS = {}


def compiled_for(family):
    """One compiled loop per family, kept because compiling costs seconds."""
    if family.name not in COMPILED:
        COMPILED[family.name] = make_objective(family.kernel)

    return COMPILED[family.name]


def scorer_for(family):
    if family.name not in SCORERS:
        SCORERS[family.name] = make_scorer(family.kernel)

    return SCORERS[family.name]


def blocks_over(count):
    """Where to cut a run of observations so each thread owns a stretch."""
    return np.linspace(0, count, get_num_threads() + 1).astype(np.int64)


@dataclass
class Model:
    """
    The family, the basis its channels are drawn on, and the quadrature the
    player beliefs are integrated with.

    Everything that turns parameters into a distribution goes through here,
    so a caller never has to know which channels the family declared.
    """

    family: object
    basis: Basis
    nodes: np.ndarray
    weights: np.ndarray

    room: dict = field(default_factory=dict)

    @property
    def links(self):
        return np.array(
            [LINK_CODES[c.link] for c in self.family.channels], dtype=np.int64
        )

    @property
    def free(self):
        """Which channels are built from bumps rather than from ramps."""
        return self.links == 2

    @property
    def signs(self):
        """Which way a ramp channel is allowed to move. Free channels: 1."""
        return np.where(self.links == 1, -1.0, 1.0)

    def rise_of(self, params):
        """
        The shape put through its link, which is what the curve is built
        from: bump coefficients as they stand, and a signed softplus of the
        rise weights.
        """
        free = self.free[:, None, None]

        return np.where(
            free,
            params.shape,
            self.signs[:, None, None] * softplus(params.shape),
        )

    def at(self, params, items, theta):
        """
        Channel values at these (item, skill) pairs, and how each one moves
        with skill.

        This is the numpy side, for prediction and for drawing. The fit
        runs the compiled loop instead.
        """
        theta = np.asarray(theta, dtype=float)
        ramps = self.basis.ramps(theta)
        bumps = bumps_of(ramps)

        rise = self.rise_of(params)
        free = self.free

        values = {}
        slopes = {}

        for c, channel in enumerate(self.family.channels):
            carrier = bumps if free[c] else ramps
            moving = (
                bumps * (1.0 - 2.0 * ramps) if free[c]
                else ramps * (1.0 - ramps)
            )

            here = rise[c][items]

            values[channel.name] = params.level[c][items] + (
                here * carrier
            ).sum(1)
            slopes[channel.name] = (
                here * moving
            ).sum(1) / self.basis.width

        return values, slopes

    def blocks(self, panel, threads):
        """
        Where to cut the observations so each thread owns a stretch, and
        the gradient buffers they add into.

        Kept per panel size, since the fit calls the objective hundreds of
        times against the same one and these are megabytes.
        """
        key = (len(panel.outcome), panel.n_players, panel.n_items)

        if key not in self.room:
            channels = len(self.family.channels)

            self.room[key] = (
                np.linspace(
                    0, len(panel.outcome), threads + 1
                ).astype(np.int64),
                np.zeros((threads, channels, panel.n_items)),
                np.zeros(
                    (threads, channels, panel.n_items, self.basis.count)
                ),
                np.zeros((threads, panel.n_players)),
                np.zeros((threads, panel.n_players)),
            )

        return self.room[key]

    def log_density(self, params, items, x, theta):
        values, _ = self.at(params, items, theta)

        return self.family.log_density(x, values)

    def skill_slope(self, params, items, x, theta):
        """d log p(x) / d skill, by the chain rule through the channels."""
        values, slopes = self.at(params, items, theta)
        gradient = self.family.gradient(x, values)

        return sum(
            gradient[name] * slopes[name] for name in self.family.names
        )


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


def load_panel(conn):
    """
    Best accuracy per (player, beatmap, mod_key), untouched, plus the pp
    that score was awarded.

    sqlite returns the other columns from the row that supplied max(), so
    the pp here belongs to the same score as the accuracy.
    """
    rows = conn.execute("""
        select s.player, s.beatmap, s.mod_key, max(s.accuracy), s.pp,
               pl.stratum, pl.pp
        from Score s
        join Player pl on pl.id = s.player
        where s.mod_settings = 0 and pl.stratum is not null
        group by s.player, s.beatmap, s.mod_key
    """)

    players = defaultdict(dict)
    stratum_of = {}
    awarded = {}
    player_pp = {}

    for player, beatmap, mods, accuracy, pp, stratum, overall in rows:
        key = f"{beatmap}:{mods}"

        players[player][key] = float(accuracy)
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


def pooling(settings, channel):
    """How hard this channel's per-map shape is pulled towards the average."""
    return getattr(settings, "pool", {}).get(channel.name, channel.pool)


def objective(params, panel, model, settings):
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

    The map penalty pools each channel towards its average across maps,
    since no single map here has the players to fit its own shape. The
    strength differs by channel and outcomes.py declares the default.

    The first term is the compiled loop; everything after it is one number
    per player or per map and stays in numpy.
    """
    sd = np.exp(params.skill_log_sd)

    bounds, room_level, room_shape, room_mean, room_log_sd = model.blocks(
        panel, get_num_threads()
    )

    for room in (room_level, room_shape, room_mean, room_log_sd):
        room.fill(0.0)

    total = compiled_for(model.family)(
        panel.rows, panel.cols, panel.outcome,
        params.level, model.rise_of(params), model.free,
        params.skill_mean, sd,
        model.nodes, model.weights, model.basis.knots, model.basis.width,
        bounds, room_level, room_shape, room_mean, room_log_sd,
    )

    grad = Parameters(
        skill_mean=room_mean.sum(0),
        skill_log_sd=room_log_sd.sum(0),
        level=room_level.sum(0),
        shape=room_shape.sum(0),
    )

    # Carry the shape gradient back through the link the forward pass
    # applied, which the compiled loop left off because it belongs to the
    # channel and the knot rather than to the observation.
    through = np.where(
        model.free[:, None, None],
        1.0,
        model.signs[:, None, None] * sigmoid(params.shape),
    )
    grad.shape *= through
    grad.skill_log_sd *= sd

    total += float(
        np.sum(
            0.5 * (sd ** 2 + params.skill_mean ** 2 - 1.0)
            - params.skill_log_sd
        )
    )
    grad.skill_mean += params.skill_mean
    grad.skill_log_sd += sd ** 2 - 1.0

    centre = float(params.skill_mean.mean())
    spread_of_population = float(
        ((params.skill_mean - centre) ** 2).mean() + (sd ** 2).mean()
    )

    total += settings.population * panel.n_players * (
        centre ** 2 + (spread_of_population - 1.0) ** 2
    )
    grad.skill_mean += settings.population * (
        2.0 * centre
        + 4.0 * (spread_of_population - 1.0) * (params.skill_mean - centre)
    )
    grad.skill_log_sd += (
        settings.population * 4.0 * (spread_of_population - 1.0) * sd ** 2
    )

    strengths = np.array([
        pooling(settings, channel) for channel in model.family.channels
    ])

    for name, strength in (
        ("level", np.full(len(strengths), settings.pool_level)),
        ("shape", strengths),
    ):
        value = getattr(params, name)
        deviation = value - value.mean(axis=1, keepdims=True)
        weight = strength.reshape((-1,) + (1,) * (value.ndim - 1))

        total += float(np.sum(weight * deviation ** 2))
        getattr(grad, name)[...] += 2.0 * weight * deviation

    return total, grad


def initialise(panel, model):
    """
    A first guess good enough that the fit only has to refine it.

    Two cheap things are known before any distribution is fitted: roughly
    who is better, and roughly which maps are harder. A two-way additive
    fit on the logit of accuracy supplies both. The logit is a working
    scale for that ordering and nothing else; the likelihood never sees
    it, and every channel below is a channel of the family itself.

    Turning each player's place in that order into a normal score starts
    the population already standard normal. Every channel then starts flat
    at whatever value describes the whole panel, except the location, which
    starts at the map's own level and rises at the rate the additive fit
    implies.
    """
    work = logit(np.clip(panel.outcome, *INSIDE))

    level, item_level, _ = additive_fit(
        panel.rows, panel.cols, work,
        panel.n_players, panel.n_items,
        np.arange(panel.n_items), 100, False,
    )

    place = np.argsort(np.argsort(level))
    skill_mean = normal_quantile((place + 0.5) / panel.n_players)

    skill_at_row = skill_mean[panel.rows]
    slope = float(
        np.dot(skill_at_row, work - item_level[panel.cols])
        / np.dot(skill_at_row, skill_at_row)
    )

    at_centre = model.basis.ramps(np.zeros(1))
    reach = float(at_centre.sum())
    unit_slope = float(
        (at_centre * (1.0 - at_centre)).sum() / model.basis.width
    )

    height = max(slope / unit_slope, 1e-3)
    flat = 1e-3

    start = model.family.start(panel.outcome)

    channels = len(model.family.channels)
    free = model.free
    signs = model.signs

    params = Parameters(
        skill_mean=skill_mean,
        skill_log_sd=np.full(panel.n_players, math.log(0.5)),
        level=np.zeros((channels, panel.n_items)),
        shape=np.zeros((channels, panel.n_items, model.basis.count)),
    )

    for c, channel in enumerate(model.family.channels):
        if free[c]:
            params.level[c] = start[channel.name]
            continue

        if channel.name == "location":
            rise = height
            centre = item_level + level.mean()
        else:
            rise = flat
            centre = start[channel.name]

        params.level[c] = centre - signs[c] * rise * reach
        params.shape[c] = float(inverse_softplus(np.array(rise)))

    return params


def descend(params, loss_and_grad, steps, rate, floor, ceiling):
    """
    Adam, with the belief widths held inside a range.

    A width running to zero turns the expectation back into a point
    estimate and the gradient with it, so it is kept off the floor.
    """
    moment = Parameters.zeros_like(params)
    scale = Parameters.zeros_like(params)

    history = []

    for step in range(1, steps + 1):
        loss, grad = loss_and_grad(params)
        history.append(loss)

        for name in FIELDS:
            here = getattr(grad, name)

            moved = 0.9 * getattr(moment, name) + 0.1 * here
            spread = 0.999 * getattr(scale, name) + 0.001 * here ** 2

            setattr(moment, name, moved)
            setattr(scale, name, spread)

            setattr(
                params, name,
                getattr(params, name)
                - rate * (moved / (1.0 - 0.9 ** step))
                / (np.sqrt(spread / (1.0 - 0.999 ** step)) + 1e-8),
            )

        params.skill_log_sd = np.clip(
            params.skill_log_sd, math.log(floor), math.log(ceiling)
        )

    return params, history


# ---------------------------------------------------------------------------
# What one score does to a player


def score_derivatives(params, model, items, x, theta, step=1e-4):
    """
    Slope and curvature of one score's log-likelihood in the player's skill.

    g says which way the result disagrees with where the player sits and
    how strongly. h says how much the map and the result reveal about skill
    at all: a map whose channels barely move here gives both near zero.

    The slope is the family's own gradient carried through the channels.
    The curvature is a central difference of that slope, which is accurate
    to about ten digits and is asked for once per score rather than once
    per step of the fit, so writing every family's second derivatives by
    hand would buy nothing.

    h is a curvature rather than a count of information, so it can come out
    negative where the density bends the wrong way under the outcome, and
    the caller has to handle that.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    items = np.atleast_1d(np.asarray(items))
    x = np.atleast_1d(np.asarray(x, dtype=float))

    g = model.skill_slope(params, items, x, theta)

    above = model.skill_slope(params, items, x, theta + step)
    below = model.skill_slope(params, items, x, theta - step)

    return g, -(above - below) / (2.0 * step)


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


def over_belief(params, model, rows, cols, callback):
    """
    Average something over each player's belief about their own skill.

    `callback` is handed the channel values at one quadrature node and
    returns one number per cell.
    """
    sd = np.exp(params.skill_log_sd)
    total = None

    for node, weight in zip(model.nodes, model.weights):
        theta = params.skill_mean[rows] + SQRT2 * sd[rows] * node
        values, _ = model.at(params, cols, theta)

        piece = weight * callback(values)
        total = piece if total is None else total + piece

    return total


def predictive(params, panel, model):
    """
    Density the model puts on each outcome, and the accuracy it expected.

    The player's skill is a belief rather than a number, so this averages
    the map's density over that belief instead of reading it at a point.
    """
    density = over_belief(
        params, model, panel.rows, panel.cols,
        lambda values: np.exp(
            model.family.log_density(panel.outcome, values)
        ),
    )
    expected = over_belief(
        params, model, panel.rows, panel.cols, model.family.mean
    )

    return np.log(np.maximum(density, 1e-300)), expected


def predictive_cdf(params, panel, model, rng=None):
    """
    Where each outcome falls inside the distribution predicted for it: 0
    when it is far below what the map and the player's skill lead you to
    expect, 1 when it is far above, 0.5 when it lands exactly there.

    If the model is right these come out uniform on (0, 1), whatever the
    map and whoever the player, so their histogram is a calibration check
    that needs no held-out data.

    A score of exactly 100% has no single place inside the distribution.
    It sits somewhere in the stretch the point mass at 1 covers, and only
    picking a point in that stretch at random keeps the uniformity that
    makes the histogram readable. Without an `rng` the middle of the
    stretch is used instead, which is steady enough to show on a page but
    piles every such score into one bin.
    """
    ceiling = panel.outcome >= 1.0

    total = over_belief(
        params, model, panel.rows, panel.cols,
        lambda values: model.family.cdf(panel.outcome, values),
    )
    below = over_belief(
        params, model, panel.rows, panel.cols, model.family.below
    )

    share = rng.random(len(panel.outcome)) if rng is not None else 0.5
    inside_mass = below + share * (1.0 - below)

    return np.where(ceiling, inside_mass, total)


def calibration(u):
    """
    How far the positions inside the predicted distributions are from flat.

    A correct model spreads them evenly over (0, 1). The largest gap
    between their running share and a straight line is the whole departure
    in one number, and where the middle sits says which way it leans.
    """
    order = np.sort(u)
    place = (np.arange(len(order)) + 0.5) / len(order)

    return {
        "mean": float(u.mean()),
        "gap": float(np.max(np.abs(order - place))),
    }


# ---------------------------------------------------------------------------
# Models that do not use a fitted skill, on the same measure


def descend_channels(family, x, params, spread, gather, steps, rate):
    """
    Adam on channel parameters that do not depend on skill.

    `spread` puts the parameters onto the observations and `gather` takes
    a per-observation gradient back to them, which is all that separates
    the two models below. `params` is indexed by channel first, like the
    map side of a full fit.
    """
    names = family.names
    params = params.copy()

    values = np.empty((len(names), len(x)))
    grad = np.empty_like(values)

    score = scorer_for(family)
    bounds = blocks_over(len(x))

    moment = np.zeros_like(params)
    scale = np.zeros_like(params)

    for step in range(1, steps + 1):
        spread(params, values)
        score(x, values, grad, bounds)

        for c in range(len(names)):
            here = gather(params[c], grad[c])

            moment[c] = 0.9 * moment[c] + 0.1 * here
            scale[c] = 0.999 * scale[c] + 0.001 * here ** 2

            params[c] -= rate * (moment[c] / (1.0 - 0.9 ** step)) / (
                np.sqrt(scale[c] / (1.0 - 0.999 ** step)) + 1e-8
            )

    return {name: params[c] for c, name in enumerate(names)}


def fit_grouped(family, x, groups, n_groups, steps, rate, pool):
    """
    One constant per channel per group, with no skill in it.

    With every group the same this is one distribution for the whole
    panel; with one group per map it is what the map says about a score
    before anyone asks who set it.
    """
    start = family.start(x)

    params = np.array([
        np.full(n_groups, start[name]) for name in family.names
    ])

    def spread(params, values):
        for c in range(params.shape[0]):
            values[c] = params[c][groups]

    def gather(params, gradient):
        return (
            -np.bincount(groups, gradient, n_groups)
            + 2.0 * pool * (params - params.mean())
        )

    return descend_channels(family, x, params, spread, gather, steps, rate)


def official_columns(subset, stars, overall):
    """
    What the official numbers know about a cell.

    A map with no star rating is not dropped. Its rating and interaction go
    to zero and a column marks it, so this baseline learns what it is worth
    when the official numbers say nothing about the map, and every held-out
    cell stays in the comparison. 2.2% of cells are in that position.

    Star rating is an input to this row alone. Nothing in the model reads
    it, and no held-out set is cut to where it exists.
    """
    strength = np.log10(np.maximum(overall[subset.rows], 1.0))

    rating = stars[subset.cols]
    rated = np.isfinite(rating)
    rating = np.where(rated, rating, 0.0)

    return np.column_stack([
        np.ones(len(rating)), strength, rating, strength * rating,
        rated.astype(float),
    ])


def official_design(train, stars, overall):
    """
    The same columns centred and scaled, and a function that puts any
    other subset on the same footing.

    Raw, the interaction of pp with star rating runs an order of magnitude
    above the intercept, and the fit either crawls or steps far enough to
    ask for a concentration no density can take. Centring is a change of
    coordinate on the same straight lines, so it costs the model nothing.
    """
    raw = official_columns(train, stars, overall)

    centre = raw.mean(0)
    spread = raw.std(0)

    # The intercept is a column of ones and has to stay one.
    centre[0] = 0.0
    spread = np.where(spread > 0.0, spread, 1.0)

    def design(subset):
        return (
            official_columns(subset, stars, overall) - centre
        ) / spread

    return design


def fit_linear(family, x, design, steps, rate):
    """
    Every channel a straight line in the columns of `design`.

    This is how the official numbers are given their turn: hand it a
    player's pp and a map's star rating and it predicts a distribution
    over the accuracy without ever seeing the score.
    """
    start = family.start(x)

    # The level that describes the whole pile belongs in the intercept and
    # nowhere else, so every other column starts at nothing.
    params = np.zeros((len(family.names), design.shape[1]))
    params[:, 0] = [start[name] for name in family.names]

    def spread(params, values):
        for c in range(params.shape[0]):
            values[c] = design @ params[c]

    def gather(_, gradient):
        return -(design.T @ gradient)

    return descend_channels(family, x, params, spread, gather, steps, rate)


# ---------------------------------------------------------------------------


def check_gradient(params, loss_and_grad, rng, samples=60):
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
        analytic = getattr(grad, name)[spot]

        scale = max(abs(numeric), abs(analytic), 1.0)
        worst = max(worst, abs(numeric - analytic) / scale)

    return worst


def check_kernels(family, rng, samples=4000):
    """
    The compiled kernel against the numpy class it was written from.

    The fit runs only the kernel and everything reported afterwards runs
    only the class, so nothing else would notice the two drifting apart.
    """
    names = family.names
    channels = len(names)

    spread = rng.normal(0.0, 1.5, (samples, channels))
    spread[:, 0] += 3.0
    spread[:, -1] -= 4.0

    x = rng.uniform(0.2, 0.9999, samples)
    x[:samples // 20] = 1.0

    values = {name: spread[:, c] for c, name in enumerate(names)}

    density = family.log_density(x, values)
    gradient = family.gradient(x, values)

    grad = np.empty(channels)
    worst_density = worst_gradient = 0.0

    for i in range(samples):
        here = family.kernel(x[i], spread[i], grad)

        worst_density = max(
            worst_density,
            abs(here - density[i]) / max(abs(density[i]), 1.0),
        )

        for c, name in enumerate(names):
            worst_gradient = max(
                worst_gradient,
                abs(grad[c] - gradient[name][i])
                / max(abs(gradient[name][i]), 1.0),
            )

    return worst_density, worst_gradient


def check_score_derivatives(params, model, rng, samples=200):
    """
    The slope and curvature one score reports, against differences of the
    log-likelihood they are the slope and curvature of.

    The incremental update rests entirely on these two numbers and nothing
    else in the fit exercises them, so they are checked separately.
    """
    n_items = params.level.shape[1]

    step = 1e-3
    worst_slope = worst_curvature = 0.0

    for _ in range(samples):
        item = np.array([int(rng.integers(n_items))])
        x = np.array([float(rng.uniform(0.6, 0.999))])
        theta = float(rng.normal(0.0, 1.2))

        g, h = score_derivatives(params, model, item, x, np.array([theta]))

        def density(at):
            return float(
                model.log_density(params, item, x, np.array([at]))[0]
            )

        here = density(theta)
        above = density(theta + step)
        below = density(theta - step)

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
    model: Model

    # Filled in by map_facts, since it needs the item list this holds.
    stars: np.ndarray = None


def build_model(settings):
    family = FAMILIES[getattr(settings, "family", "logit-normal")]
    nodes, weights = np.polynomial.hermite.hermgauss(settings.quadrature)

    return Model(
        family=family,
        basis=Basis(settings.knots, settings.reach),
        nodes=nodes,
        weights=weights / math.sqrt(math.pi),
    )


def prepare(conn, settings):
    """Read the panel, drop the thin edges of it, and build the model."""
    held, stratum_of, player_pp, awarded = load_panel(conn)

    if not held:
        raise NotEnoughData("No scores in the panel yet.")

    core = trim(held, settings.min_players, settings.min_items)

    if len(core) < 2:
        raise NotEnoughData(
            f"Nothing survives {settings.min_players} players x "
            f"{settings.min_items} items. Keep filling."
        )

    roster, items, rows, cols, outcome = observations(core)

    return Study(
        panel=Panel(rows, cols, outcome, len(roster), len(items)),
        roster=roster,
        items=items,
        stratum_of=stratum_of,
        player_pp=player_pp,
        awarded=awarded,
        model=build_model(settings),
    )


def fit_panel(study, settings, panel=None):
    """Fit skill beliefs and map curves to a panel, or a subset of one."""
    panel = study.panel if panel is None else panel

    def loss_and_grad(params):
        return objective(params, panel, study.model, settings)

    return descend(
        initialise(panel, study.model), loss_and_grad,
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


def map_columns(study, params, facts, stars, settings, seed=0):
    """
    One row per map: how much a score there tells you about skill, how hard
    the map comes out, what pp it pays, and how far that pp sits from what
    maps of the same difficulty pay.
    """
    panel, model = study.panel, study.model
    index = np.arange(panel.n_items)
    rng = np.random.default_rng(seed)

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

    middle = np.zeros(panel.n_items)

    at_typical = model.at(params, index, typical)
    at_middle = model.at(params, index, middle)

    at_median = model.family.mean(at_middle[0])

    return {
        "counts": np.array([len(skills[j]) for j in index]),
        "typical": typical,
        "straddles": (below > 0.2) & (below < 0.8),
        "information": model.family.information(*at_typical, rng),
        "information_at_centre": model.family.information(*at_middle, rng),
        "at_median": at_median,
        "raw_mean": np.bincount(
            panel.cols, panel.outcome, panel.n_items
        ) / np.maximum(np.bincount(panel.cols, minlength=panel.n_items), 1),
        "stars": stars,
        "length": about(study, facts, "length"),
        "playcount": about(study, facts, "playcount"),
        "live": live,
        "from_stars": discrepancy(
            panel, live, stars, settings.star_window, settings
        ),
        "from_curves": discrepancy(
            panel, live, -logit(np.clip(at_median, *INSIDE)),
            settings.skill_window, settings,
        ),
    }


def about(study, facts, field):
    return np.array([
        facts.get(int(k.split(":", 1)[0]), {}).get(field) or np.nan
        for k in study.items
    ])


def discrepancy(panel, live, axis, window, settings):
    """
    How much more pp a map pays than maps of the same difficulty do.

    Each map is compared against its own neighbours on the axis rather
    than against a curve fitted across the whole range, since the ends of
    the range are where the sparsest maps sit.
    """
    ok = np.isfinite(axis) & np.isfinite(live)

    gap = np.full(panel.n_items, np.nan)
    expected, _ = expected_pp(
        axis[ok], live[ok], window, settings.min_neighbours
    )
    gap[ok] = live[ok] - expected

    return gap


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


# ---------------------------------------------------------------------------


def pool_override(text):
    name, _, value = text.partition("=")

    if not value:
        raise argparse.ArgumentTypeError("write it as channel=strength")

    return name, float(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--db", default=None)

    parser.add_argument(
        "--family",
        default="logit-normal",
        choices=sorted(FAMILIES),
        help="which conditional distribution over accuracy to fit. The "
             "default is the one --compare-families ranks first",
    )
    parser.add_argument(
        "--compare-families",
        action="store_true",
        help="fit every family and rank them on the same held-out cells",
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
        "--pool",
        type=pool_override,
        action="append",
        default=[],
        metavar="CHANNEL=STRENGTH",
        help="how hard one channel's per-map shape is pulled towards the "
             "average map's, overriding what outcomes.py declares for it. "
             "Weak values let a map with few players fit its own scores and "
             "claim a density no held-out score can match",
    )
    parser.add_argument(
        "--pool-level",
        type=float,
        default=5.0,
        help="the same for each channel's level, which is one number per "
             "map rather than a shape. Swept over 0.05 to 40, where the "
             "held-out density is flat from about 2 to 40 and falls off "
             "sharply below 1",
    )

    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=12)

    parser.add_argument(
        "--star-window",
        type=float,
        default=0.5,
        help="star window for the maps each map is compared against",
    )
    parser.add_argument(
        "--skill-window",
        type=float,
        default=0.5,
        help="the same window in fitted difficulty",
    )
    parser.add_argument("--min-neighbours", type=int, default=8)

    parser.add_argument("--check-gradient", action="store_true")

    args = parser.parse_args(argv)
    args.pool = dict(args.pool)

    return args


def rank_families(study, args, holdout, rng):
    """
    Every family on the same held-out cells, against models that do not
    use a fitted skill.

    All of these put their mass on the accuracy itself, mixing a density
    on (0, 1) with a point mass at 1, so their log densities are on one
    measure and can be read against each other. Anything that measures the
    outcome differently belongs in its own table, since a log density
    carries the units of whatever it was taken against.
    """
    panel = study.panel
    train = panel.take(~holdout)
    test = panel.take(holdout)

    overall = np.array(
        [study.player_pp[p] if study.player_pp[p] else np.nan
         for p in study.roster]
    )
    stars = study.stars

    design = official_design(train, stars, overall)

    rows = []

    for name in sorted(FAMILIES):
        family = FAMILIES[name]

        one = fit_grouped(
            family, train.outcome, np.zeros(len(train.outcome), dtype=int),
            1, 150, 0.15, 0.0,
        )
        rows.append((
            name, "one distribution for everything",
            family.log_density(
                test.outcome,
                {k: np.full(len(test.outcome), v[0]) for k, v in one.items()},
            ),
            None,
        ))

        per_map = fit_grouped(
            family, train.outcome, train.cols, panel.n_items,
            150, 0.15, args.pool_level,
        )
        rows.append((
            name, "one per map, no skill",
            family.log_density(
                test.outcome,
                {k: v[test.cols] for k, v in per_map.items()},
            ),
            None,
        ))

        official = fit_linear(
            family, train.outcome, design(train), 300, 0.05
        )
        rows.append((
            name, "player pp + map star rating",
            family.log_density(
                test.outcome,
                {k: design(test) @ v for k, v in official.items()},
            ),
            None,
        ))

        settings = argparse.Namespace(**vars(args))
        settings.family = name

        study.model = build_model(settings)
        params, _ = fit_panel(study, settings, panel=train)

        density, _ = predictive(params, test, study.model)
        u = predictive_cdf(params, test, study.model, rng)

        rows.append((name, "skill belief + map curves", density, u))

    return rows


def report_families(rows):
    print()
    print(
        f"{'family':<14}{'model':<30}{'log density':>13}"
        f"{'centre':>9}{'gap':>8}"
    )
    print("-" * 74)

    last = None

    for family, model, density, u in rows:
        if last is not None and family != last:
            print()

        last = family

        if u is None:
            print(
                f"{family:<14}{model:<30}{float(np.mean(density)):>13.3f}"
                f"{'':>9}{'':>8}"
            )
        else:
            fit = calibration(u)
            print(
                f"{family:<14}{model:<30}{float(np.mean(density)):>13.3f}"
                f"{fit['mean']:>9.3f}{fit['gap']:>8.3f}"
            )

    print()
    print(
        "log density is per held-out cell and higher is better. Every row "
        "puts\nits mass on the accuracy itself, so the numbers compare. "
        "`centre` is the\naverage place a score took inside its own "
        "predicted distribution and\nshould be 0.5; `gap` is the largest "
        "departure of those places from flat\nand should be near 0."
    )


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    conn = connect_readonly(args.db)

    try:
        study = prepare(conn, args)
    except NotEnoughData as problem:
        print(problem)
        return

    panel, roster, items = study.panel, study.roster, study.items
    stratum_of, player_pp = study.stratum_of, study.player_pp
    model = study.model
    outcome = panel.outcome

    facts, modded_stars = map_facts(conn, study)
    study.stars = modded_stars

    print(
        f"panel: {panel.n_players} players, {panel.n_items} items, "
        f"{len(outcome)} observations"
    )
    print(
        f"outcome: accuracy itself, "
        f"{100 * np.quantile(outcome, 0.1):.2f}% to "
        f"{100 * np.quantile(outcome, 0.9):.2f}% over the middle four "
        f"fifths, {100 * np.mean(outcome >= 1.0):.1f}% at exactly 100%"
    )
    print(
        f"family: {model.family.name}, channels "
        f"{', '.join(model.family.names)}"
    )
    print(
        f"curves: {args.knots} knots from {-args.reach:+.1f} to "
        f"{args.reach:+.1f} skill, {args.quadrature}-node quadrature"
    )

    holdout = rng.random(len(outcome)) < args.holdout
    training = ~holdout

    if args.compare_families:
        print()
        print(
            f"fitting every family on the same {int(training.sum())} "
            f"training cells"
        )

        report_families(rank_families(study, args, holdout, rng))

        study.model = model
        return

    def loss_and_grad_on(subset):
        return lambda p: objective(p, subset, model, args)

    if args.check_gradient:
        start = initialise(panel, model)
        start.skill_mean += rng.normal(0.0, 0.1, panel.n_players)
        start.shape += rng.normal(0.0, 0.1, start.shape.shape)

        worst = check_gradient(start, loss_and_grad_on(panel), rng)
        slope_error, curvature_error = check_score_derivatives(
            start, model, rng
        )

        print()
        print(
            f"against central differences, worst relative error: "
            f"objective gradient {worst:.1e}, "
            f"one score's slope {slope_error:.1e}, "
            f"its curvature {curvature_error:.1e}"
        )

        print()
        print(f"{'family':<14}{'log density':>13}{'gradient':>11}")
        print("-" * 38)

        for name in sorted(FAMILIES):
            print(
                f"{name:<14}"
                + "{:>13.1e}{:>11.1e}".format(
                    *check_kernels(FAMILIES[name], rng)
                )
            )

        print()
        print(
            "worst relative difference between each family's compiled "
            "kernel,\nwhich the fit runs, and the numpy class it was "
            "written from, which\neverything reported afterwards runs"
        )

    print()
    print("fitting on the training cells")

    train_panel = panel.take(training)
    params, history = descend(
        initialise(train_panel, model), loss_and_grad_on(train_panel),
        args.steps, args.rate, 0.02, 3.0,
    )

    settled = history[max(0, len(history) - 50)] - history[-1]

    print(
        f"  objective {history[0]:.0f} -> {history[-1]:.0f}, "
        f"still moving {settled:.2f} over the last 50 steps"
    )

    # ---- holdout, against the models that read the official numbers

    test = panel.take(holdout)

    overall = np.array(
        [player_pp[p] if player_pp[p] else np.nan for p in roster]
    )

    family = model.family
    rankings = []

    one = fit_grouped(
        family, train_panel.outcome,
        np.zeros(len(train_panel.outcome), dtype=int), 1, 150, 0.15, 0.0,
    )
    rankings.append((
        "one distribution for everything",
        family.log_density(
            test.outcome,
            {k: np.full(len(test.outcome), v[0]) for k, v in one.items()},
        ),
        None,
    ))

    per_map = fit_grouped(
        family, train_panel.outcome, train_panel.cols, panel.n_items,
        150, 0.15, args.pool_level,
    )
    rankings.append((
        "one per map, no skill",
        family.log_density(
            test.outcome, {k: v[test.cols] for k, v in per_map.items()}
        ),
        None,
    ))

    pp_design = official_design(train_panel, modded_stars, overall)

    official = fit_linear(
        family, train_panel.outcome, pp_design(train_panel), 300, 0.05
    )
    rankings.append((
        "player pp + map star rating",
        family.log_density(
            test.outcome,
            {k: pp_design(test) @ v for k, v in official.items()},
        ),
        None,
    ))

    # Reading the curves at the player's mean rather than averaging over
    # the belief, so the cost of carrying the belief is visible on its own.
    point, _ = model.at(params, test.cols, params.skill_mean[test.rows])
    rankings.append((
        "skill point + map curves",
        family.log_density(test.outcome, point),
        None,
    ))

    log_density, _ = predictive(params, test, model)
    rankings.append((
        "skill belief + map curves",
        log_density,
        predictive_cdf(params, test, model, rng),
    ))

    print()
    print(
        f"predicting {len(test.outcome)} held-out cells "
        f"({100 * args.holdout:.0f}% of the panel), as a distribution over "
        f"the\naccuracy itself"
    )
    print()
    print(f"{'model':<32}{'log density':>13}{'centre':>9}{'gap':>8}")
    print("-" * 62)

    for name, density, u in rankings:
        if u is None:
            print(f"{name:<32}{float(np.mean(density)):>13.3f}")
            continue

        fit = calibration(u)
        print(
            f"{name:<32}{float(np.mean(density)):>13.3f}"
            f"{fit['mean']:>9.3f}{fit['gap']:>8.3f}"
        )

    print()
    print(
        "log density is per cell and higher is better. The pp row is what "
        "the\nofficial numbers know about a cell without seeing the score, "
        "which is\nthe comparison the model has to win. `centre` is the "
        "average place a\nscore took inside its own predicted distribution "
        "and should be 0.5;\n`gap` is the largest departure of those places "
        "from flat."
    )

    # ---- the fit on everything, which is what the maps are read off

    print()
    print("refitting on the whole panel")

    params, history = descend(
        initialise(panel, model), loss_and_grad_on(panel),
        args.steps, args.rate, 0.02, 3.0,
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

    # ---- how much each map tells you

    columns = map_columns(study, params, facts, modded_stars, args, args.seed)

    tells = columns["information"]
    at_centre = columns["information_at_centre"]
    typical_skill = columns["typical"]
    counts = columns["counts"]

    def show_maps(order, title):
        print()
        print(title)
        print()
        print(
            f"{'item':<16}{'players':>8}{'tells':>8}{'at 50th':>9}"
            f"{'skill played':>14}  map"
        )
        print("-" * 96)

        for j in order:
            beatmap_id = int(items[j].split(":", 1)[0])
            name = facts.get(beatmap_id, {}).get("name", "?")

            print(
                f"{items[j]:<16}{counts[j]:>8}"
                f"{tells[j]:>8.2f}{at_centre[j]:>9.2f}"
                f"{typical_skill[j]:>14.2f}  {name[:40]}"
            )

    ranked = np.argsort(tells)

    print()
    print(
        f"how much one score tells you about skill, where the map's own "
        f"players sit:\n{tells.min():.2f} to {tells.max():.2f} in units of "
        f"precision, middle four fifths "
        f"{np.quantile(tells, 0.1):.2f} to {np.quantile(tells, 0.9):.2f}. "
        f"A map at 1.00\nis worth as much as the whole population prior. "
        f"That range is what the two\ntables below sort on."
    )

    show_maps(
        ranked[:args.top],
        "maps that tell you least about the player who set the score",
    )
    show_maps(
        ranked[::-1][:args.top],
        "and the ones that tell you most",
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
                f"tells {tells[j]:.2f} where they sit, "
                f"{at_centre[j]:.2f} at the 50th percentile"
            )
    else:
        print()
        print("Reol - No title [byfaR's Hard] is not in the panel.")

    # ---- what one score does, and what it does on a map that says little

    print()
    print("folding one held-out score into a player's belief")

    step_panel = panel.take(holdout)

    g, h = score_derivatives(
        params, model, step_panel.cols, step_panel.outcome,
        params.skill_mean[step_panel.rows],
    )
    prior_sd = np.exp(params.skill_log_sd[step_panel.rows])
    moved, tightened, taken = laplace_step(
        params.skill_mean[step_panel.rows], prior_sd, g, h
    )

    shift = np.abs(moved - params.skill_mean[step_panel.rows])
    cut = 1.0 - tightened / prior_sd

    says = tells[step_panel.cols]
    thirds = np.quantile(says, [1 / 3, 2 / 3])

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
        f"{'what the map tells':<20}{'scores':>8}{'median h':>11}"
        f"{'median move':>13}{'width cut':>11}"
    )
    print("-" * 63)

    for name, mask in (
        ("least third", says <= thirds[0]),
        ("middle third", (says > thirds[0]) & (says <= thirds[1])),
        ("most third", says > thirds[1]),
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
        f"skill tracks\n  the map's information at "
        f"{spearman(list(says), list(h)):+.2f} rank correlation, which "
        f"is the\n  claim that a map nobody is separated on stops counting"
    )

    # ---- what the discrepancy against pp still tracks

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

    against("expected accuracy at median skill", at_median)
    against("mean accuracy observed", raw_mean)
    against("pp the map pays average level", live)
    against("star rating", modded_stars)

    print()
    print(
        f"  expected accuracy at median skill runs "
        f"{100 * np.quantile(at_median, 0.1):.2f}% to "
        f"{100 * np.quantile(at_median, 0.9):.2f}% over the middle four "
        f"fifths,\n  against "
        f"{modded_stars[np.isfinite(modded_stars)].min():.1f} to "
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
        f"players on both sides of the median,\n  where the curves are read "
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
        ("fitted curves", from_curves, False),
        ("fitted curves, straddling", from_curves, True),
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
