#!/usr/bin/env python3
"""
The distribution a score is drawn from, once a player's skill is known.

Everything above this module sees one thing:

    D = family.at(theta)        a distribution over the outcome

and asks it for a log density, a CDF, a quantile, a sample, and how the
log density moves with skill. Whether D is a beta, a logit-normal or a
mixture is not visible from there.

The outcome here is the accuracy itself, a number in (0, 1]. It is not
transformed first. A transform followed by a normal distribution is two
assumptions where one will do, and the one that matters is which family
fits, which is answered by held-out likelihood in fit_skill_and_curves.py
rather than by picking a coordinate that looks symmetric.

Accuracy piles up against 1, and 1.4% of the panel sits exactly on it, so
every family here is a continuous density on (0, 1) mixed with a point
mass at 1. The mass makes the families comparable only against each other:
they share a dominating measure, and a model on any other scale does not.

A family is described by its channels. A channel is one number that the
distribution needs, written as a function of skill, and each one says
whether it must rise with skill or is free to move either way:

    location      where the map puts a player of this skill
    steadiness    how tightly their results cluster there
    perfect       how often they take the map to 100%

The map-side code turns each channel into a spline over skill. This module
only ever sees the channel values at some skill, and reports how the log
density moves with each of them.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
SQRT2 = math.sqrt(2.0)

# The continuous part of every family is a density on the open interval,
# so an accuracy is held off both ends before it is read. Only six scores
# in the panel fall between 0.9995 and 1, and none is below 6e-4, so
# neither clamp touches real data.
INSIDE = (1e-6, 1.0 - 1e-9)


# ---------------------------------------------------------------------------
# Numeric pieces numpy does not carry


def sigmoid(x):
    """Logistic function, written so large |x| cannot overflow."""
    decay = np.exp(-np.abs(x))

    return np.where(x >= 0.0, 1.0 / (1.0 + decay), decay / (1.0 + decay))


def logit(p):
    return np.log(p) - np.log1p(-p)


def softplus(x):
    return np.logaddexp(0.0, x)


def inverse_softplus(x):
    return np.log(np.expm1(x))


def gaussian_log_density(y, mean, sd):
    return -LOG_SQRT_TWO_PI - np.log(sd) - 0.5 * ((y - mean) / sd) ** 2


def normal_cdf(x):
    """
    Phi. numpy has no erf and math.erf takes one value at a time, which is
    affordable because nothing in the fit itself calls this: it is for
    reporting, for the calibration check, and for the quantile table below.
    """
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / SQRT2))


def normal_quantile(p, _table={}):
    """Inverse of normal_cdf, by interpolating the curve itself."""
    if not _table:
        grid = np.linspace(-8.0, 8.0, 64001)
        _table["x"] = grid
        _table["p"] = normal_cdf(grid)

    return np.interp(p, _table["p"], _table["x"])


# Lanczos, g = 7 with nine coefficients, which holds about fifteen digits
# over the whole positive half-line.
LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def log_gamma(x):
    """log Gamma(x) for x > 0, elementwise over an array."""
    z = np.asarray(x, dtype=float) - 1.0

    series = np.full(z.shape, LANCZOS[0]) if z.shape else LANCZOS[0]

    for i, coefficient in enumerate(LANCZOS[1:], start=1):
        series = series + coefficient / (z + i)

    t = z + 7.5

    return LOG_SQRT_TWO_PI + (z + 0.5) * np.log(t) - t + np.log(series)


def digamma(x):
    """
    The derivative of log Gamma.

    The asymptotic series is only good for large arguments, so small ones
    are walked up by psi(x) = psi(x + 1) - 1/x first.
    """
    x = np.array(x, dtype=float, copy=True)
    shift = np.zeros(x.shape)

    for _ in range(6):
        low = x < 6.0
        shift = np.where(low, shift - 1.0 / np.where(low, x, 1.0), shift)
        x = np.where(low, x + 1.0, x)

    inverse = 1.0 / x
    square = inverse * inverse

    series = np.log(x) - 0.5 * inverse - square * (
        1.0 / 12.0 - square * (
            1.0 / 120.0 - square * (
                1.0 / 252.0 - square * (1.0 / 240.0 - square / 132.0)
            )
        )
    )

    return shift + series


def beta_fraction(a, b, x, steps=300):
    """
    The continued fraction behind the incomplete beta, by Lentz's method.

    Called only on the half of the domain where it converges quickly; the
    caller reflects the other half onto this one.
    """
    tiny = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = np.ones(x.shape)
    d = 1.0 - qab * x / qap
    d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
    h = d.copy()

    for step in range(1, steps + 1):
        even = 2 * step

        term = step * (b - step) * x / ((qam + even) * (a + even))

        d = 1.0 + term * d
        d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + term / np.where(np.abs(c) < tiny, tiny, c)
        h = h * d * c

        term = -(a + step) * (qab + step) * x / ((a + even) * (qap + even))

        d = 1.0 + term * d
        d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + term / np.where(np.abs(c) < tiny, tiny, c)
        h = h * d * c

    return h


def incomplete_beta(a, b, x):
    """The regularised incomplete beta, which is the beta CDF."""
    a, b, x = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(x, dtype=float),
    )

    inside = np.clip(x, 1e-300, 1.0 - 1e-16)

    # The fraction converges slowly above this point and quickly below it,
    # and I_x(a, b) = 1 - I_{1-x}(b, a) moves one onto the other.
    reflect = inside > (a + 1.0) / (a + b + 2.0)

    first = np.where(reflect, b, a)
    second = np.where(reflect, a, b)
    point = np.where(reflect, 1.0 - inside, inside)

    front = np.exp(
        log_gamma(a + b) - log_gamma(first) - log_gamma(second)
        + first * np.log(point) + second * np.log1p(-point)
    )

    value = front * beta_fraction(first, second, point) / first
    value = np.where(reflect, 1.0 - value, value)

    return np.clip(np.where(x <= 0.0, 0.0, np.where(x >= 1.0, 1.0, value)),
                   0.0, 1.0)


def beta_log_density(x, mean, concentration):
    """
    A beta written by its mean and how tightly it holds to that mean,
    rather than by the two counts, so that the mean is one channel and the
    tightness is another.
    """
    a = mean * concentration
    b = (1.0 - mean) * concentration

    return (
        log_gamma(concentration) - log_gamma(a) - log_gamma(b)
        + (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)
    )


# ---------------------------------------------------------------------------
# The same densities again, one score at a time, for the compiler
#
# The fit reads a log density and its channel gradients tens of millions of
# times, and doing that with numpy means building an array of every
# intermediate quantity for every observation at once. Written as scalars
# and compiled, the whole calculation stays in registers. Every kernel below
# is the arithmetic of the matching class further down, and
# `check_kernels` in fit_skill_and_curves.py holds the two to each other.
#
# `value` and `grad` are indexed by the position of the channel in the
# family's `channels` tuple, so the order there is what these read.


@njit(inline="always")
def one_sigmoid(v):
    if v >= 0.0:
        return 1.0 / (1.0 + math.exp(-v))

    rise = math.exp(v)

    return rise / (1.0 + rise)


@njit(inline="always")
def one_softplus(v):
    return math.log1p(math.exp(-abs(v))) + max(v, 0.0)


@njit(inline="always")
def one_digamma(x):
    shift = 0.0

    while x < 6.0:
        shift -= 1.0 / x
        x += 1.0

    inverse = 1.0 / x
    square = inverse * inverse

    return shift + math.log(x) - 0.5 * inverse - square * (
        1.0 / 12.0 - square * (
            1.0 / 120.0 - square * (
                1.0 / 252.0 - square * (1.0 / 240.0 - square / 132.0)
            )
        )
    )


@njit(inline="always")
def held_inside(x):
    if x < 1e-6:
        return 1e-6
    if x > 1.0 - 1e-9:
        return 1.0 - 1e-9

    return x


@njit(inline="always")
def one_beta(x, mean, concentration, left, right):
    """
    A beta component: its log density at x, and how that moves with the
    logit of its mean and with the log of its concentration.

    `left` and `right` are log(x) and log(1 - x), which the caller already
    has and every component shares.
    """
    a = mean * concentration
    b = (1.0 - mean) * concentration

    at_a = one_digamma(a)
    at_b = one_digamma(b)

    density = (
        math.lgamma(concentration) - math.lgamma(a) - math.lgamma(b)
        + (a - 1.0) * left + (b - 1.0) * right
    )

    lean = concentration * mean * (1.0 - mean) * (at_b - at_a + left - right)
    tighten = concentration * (
        one_digamma(concentration) - mean * at_a - (1.0 - mean) * at_b
        + mean * left + (1.0 - mean) * right
    )

    return density, lean, tighten


@njit(inline="always")
def beta_kernel(x, value, grad):
    mass = one_sigmoid(value[2])

    if x >= 1.0:
        grad[0] = 0.0
        grad[1] = 0.0
        grad[2] = 1.0 - mass

        return math.log(mass)

    inside = held_inside(x)
    left = math.log(inside)
    right = math.log1p(-inside)

    density, lean, tighten = one_beta(
        inside, one_sigmoid(value[0]), math.exp(value[1]), left, right
    )

    grad[0] = lean
    grad[1] = tighten
    grad[2] = -mass

    return math.log1p(-mass) + density


@njit(inline="always")
def logit_normal_kernel(x, value, grad):
    mass = one_sigmoid(value[2])

    if x >= 1.0:
        grad[0] = 0.0
        grad[1] = 0.0
        grad[2] = 1.0 - mass

        return math.log(mass)

    inside = held_inside(x)
    left = math.log(inside)
    right = math.log1p(-inside)

    width = math.exp(value[1])
    z = (left - right - value[0]) / width

    grad[0] = z / width
    grad[1] = z * z - 1.0
    grad[2] = -mass

    return (
        math.log1p(-mass) - left - right
        - value[1] - LOG_SQRT_TWO_PI - 0.5 * z * z
    )


@njit(inline="always")
def beta_mixture_kernel(x, value, grad):
    mass = one_sigmoid(value[5])

    if x >= 1.0:
        for c in range(5):
            grad[c] = 0.0
        grad[5] = 1.0 - mass

        return math.log(mass)

    inside = held_inside(x)
    left = math.log(inside)
    right = math.log1p(-inside)

    drop = one_softplus(value[2])
    chance = one_sigmoid(value[4])

    clean, clean_lean, clean_tighten = one_beta(
        inside, one_sigmoid(value[0]), math.exp(value[1]), left, right
    )
    slipped, slip_lean, slip_tighten = one_beta(
        inside, one_sigmoid(value[0] - drop), math.exp(value[3]), left, right
    )

    good = math.log1p(-chance) + clean
    bad = math.log(chance) + slipped

    # How much of this score the falling-apart component accounts for.
    blame = one_sigmoid(bad - good)

    grad[0] = (1.0 - blame) * clean_lean + blame * slip_lean
    grad[1] = (1.0 - blame) * clean_tighten
    grad[2] = -blame * slip_lean * one_sigmoid(value[2])
    grad[3] = blame * slip_tighten
    grad[4] = blame - chance
    grad[5] = -mass

    top = max(good, bad)

    return math.log1p(-mass) + top + math.log1p(
        math.exp(min(good, bad) - top)
    )


# ---------------------------------------------------------------------------
# What a family is

LINK_CODES = {"rising": 0, "falling": 1, "free": 2}


@dataclass(frozen=True)
class Channel:
    """
    One number a family needs, as a function of skill.

    `link` is what the map side is allowed to build:

        rising   never falls as skill rises
        falling  never rises
        free     any shape, positive or negative, over the fitted range

    `pool` is how hard the per-map shape of this channel is pulled towards
    the average map's, which is the only thing holding a channel up on a
    map with eleven players. Swept, not guessed: see the README.
    """

    name: str
    link: str
    pool: float
    note: str


class Family:
    """
    A conditional distribution over one accuracy.

    Subclasses supply `channels` and the four methods below. Everything
    else on this class is written once against those.
    """

    name = ""
    channels = ()

    # The compiled twin of log_density and gradient, which is what the fit
    # itself runs. Everything else here is numpy, because it is called for
    # reporting and for drawing rather than inside the descent.
    kernel = None

    def log_density(self, x, values):
        """
        log p(x) under the mixed measure: a density on (0, 1), and the
        probability of the mass at 1 where x is 1.
        """
        raise NotImplementedError

    def gradient(self, x, values):
        """d log p(x) / d channel, one array per channel name."""
        raise NotImplementedError

    def cdf(self, x, values):
        """P(X <= x), which jumps to 1 at the mass."""
        raise NotImplementedError

    def sample(self, values, rng):
        raise NotImplementedError

    def mean(self, values):
        raise NotImplementedError

    def start(self, x):
        """
        Channel values that describe a pile of accuracies on their own,
        used to start the fit somewhere sane.
        """
        raise NotImplementedError

    # -- written once, against the four above

    @property
    def names(self):
        return tuple(channel.name for channel in self.channels)

    def below(self, values):
        """
        The CDF just under 1, so the mass at 1 covers what is left above.

        A score of exactly 1 has no single position inside the
        distribution, only that stretch at the top.
        """
        raise NotImplementedError

    def quantile(self, p, values, steps=60):
        """
        The accuracy at probability p, by bisecting the CDF.

        Every family here has a closed-form CDF and no closed-form inverse
        worth writing, and this is only ever called to draw a band.
        """
        p = np.asarray(p, dtype=float)

        low = np.full(p.shape, 1e-9)
        high = np.full(p.shape, 1.0 - 1e-12)

        for _ in range(steps):
            middle = 0.5 * (low + high)
            take = self.cdf(middle, values) < p

            low = np.where(take, middle, low)
            high = np.where(take, high, middle)

        return np.where(p >= self.below(values), 1.0, 0.5 * (low + high))

    def information(self, values, slopes, rng, draws=64):
        """
        How much one score on this map tells you about skill, at this
        skill: the expected square of d log p / d theta.

        `slopes` is d channel / d theta, so this is the chain rule under
        an expectation over outcomes the map itself would produce. It is
        in units of precision, so a value of 1 says a single score here is
        worth as much as the whole population prior.

        This replaces reading the slope of an expected outcome. A slope
        depends on the scale the outcome is written on, and there is no
        longer a scale; the information does not.
        """
        total = np.zeros(np.shape(slopes[self.names[0]]))

        for _ in range(draws):
            drawn = self.sample(values, rng)
            gradient = self.gradient(drawn, values)

            moved = sum(
                gradient[name] * slopes[name] for name in self.names
            )
            total = total + moved ** 2

        return total / draws


# ---------------------------------------------------------------------------
# The candidates


class BetaAtOne(Family):
    """
    A beta on (0, 1) with a point mass at 1.

    The beta is the plain choice for a bounded outcome: it is free to lean
    either way and to concentrate as far as the data asks. Written by its
    mean, the mean is exactly the channel that has to rise with skill.
    """

    name = "beta"
    kernel = staticmethod(beta_kernel)

    channels = (
        Channel("location", "rising", 2.0,
                "logit of the mean accuracy at this skill"),
        Channel("steadiness", "free", 50.0,
                "log of the beta concentration: higher is tighter"),
        Channel("perfect", "rising", 2.0,
                "logit of the chance the score is exactly 100%"),
    )

    def parts(self, x, values):
        mean = sigmoid(values["location"])
        concentration = np.exp(values["steadiness"])
        mass = sigmoid(values["perfect"])
        atom = np.asarray(x) >= 1.0

        return mean, concentration, mass, atom

    def log_density(self, x, values):
        mean, concentration, mass, atom = self.parts(x, values)
        inside = np.clip(x, *INSIDE)

        return np.where(
            atom,
            np.log(mass),
            np.log1p(-mass) + beta_log_density(inside, mean, concentration),
        )

    def gradient(self, x, values):
        mean, concentration, mass, atom = self.parts(x, values)
        inside = np.clip(x, *INSIDE)

        a = mean * concentration
        b = (1.0 - mean) * concentration

        left = np.log(inside)
        right = np.log1p(-inside)

        lean = digamma(b) - digamma(a) + left - right
        tighten = (
            digamma(concentration)
            - mean * digamma(a) - (1.0 - mean) * digamma(b)
            + mean * left + (1.0 - mean) * right
        )

        return {
            "location": np.where(
                atom, 0.0, concentration * mean * (1.0 - mean) * lean
            ),
            "steadiness": np.where(atom, 0.0, concentration * tighten),
            "perfect": np.where(atom, 1.0 - mass, -mass),
        }

    def cdf(self, x, values):
        mean, concentration, mass, _ = self.parts(x, values)

        spread = incomplete_beta(
            mean * concentration, (1.0 - mean) * concentration,
            np.clip(x, 0.0, 1.0),
        )

        return np.where(np.asarray(x) >= 1.0, 1.0, (1.0 - mass) * spread)

    def below(self, values):
        return 1.0 - sigmoid(values["perfect"])

    def sample(self, values, rng):
        mean = sigmoid(values["location"])
        concentration = np.exp(values["steadiness"])
        mass = sigmoid(values["perfect"])

        drawn = rng.beta(mean * concentration, (1.0 - mean) * concentration)

        return np.where(rng.random(np.shape(mass)) < mass, 1.0, drawn)

    def mean(self, values):
        mass = sigmoid(values["perfect"])

        return mass + (1.0 - mass) * sigmoid(values["location"])

    def start(self, x):
        inside = np.clip(x[x < 1.0], *INSIDE)
        centre = float(inside.mean())
        scatter = max(float(inside.var()), 1e-6)

        # Moment matching: the beta with this mean and variance has this
        # concentration.
        concentration = max(centre * (1.0 - centre) / scatter - 1.0, 0.5)
        mass = min(max(float((x >= 1.0).mean()), 1e-4), 0.5)

        return {
            "location": float(logit(np.array(centre))),
            "steadiness": math.log(concentration),
            "perfect": float(logit(np.array(mass))),
        }


class LogitNormalAtOne(Family):
    """
    A normal on the logit of accuracy, with a point mass at 1.

    Its tails are heavier than a beta's near either end, which is where
    chokes and near-misses live, and its spread is one free number rather
    than something tied to its mean.
    """

    name = "logit-normal"
    kernel = staticmethod(logit_normal_kernel)

    channels = (
        Channel("location", "rising", 2.0,
                "middle of the logit of accuracy at this skill"),
        Channel("spread", "free", 50.0,
                "log of the standard deviation on the logit scale"),
        Channel("perfect", "rising", 2.0,
                "logit of the chance the score is exactly 100%"),
    )

    def parts(self, x, values):
        centre = values["location"]
        width = np.exp(values["spread"])
        mass = sigmoid(values["perfect"])
        atom = np.asarray(x) >= 1.0

        return centre, width, mass, atom

    def log_density(self, x, values):
        centre, width, mass, atom = self.parts(x, values)
        inside = np.clip(x, *INSIDE)

        # The Jacobian of the logit, which is what makes this a density on
        # accuracy rather than on the logit of it.
        jacobian = -np.log(inside) - np.log1p(-inside)

        return np.where(
            atom,
            np.log(mass),
            np.log1p(-mass) + jacobian
            + gaussian_log_density(logit(inside), centre, width),
        )

    def gradient(self, x, values):
        centre, width, mass, atom = self.parts(x, values)
        inside = np.clip(x, *INSIDE)

        z = (logit(inside) - centre) / width

        return {
            "location": np.where(atom, 0.0, z / width),
            "spread": np.where(atom, 0.0, z * z - 1.0),
            "perfect": np.where(atom, 1.0 - mass, -mass),
        }

    def cdf(self, x, values):
        centre, width, mass, _ = self.parts(x, values)
        inside = np.clip(x, *INSIDE)

        spread = normal_cdf((logit(inside) - centre) / width)

        return np.where(np.asarray(x) >= 1.0, 1.0, (1.0 - mass) * spread)

    def below(self, values):
        return 1.0 - sigmoid(values["perfect"])

    def sample(self, values, rng):
        centre = values["location"]
        width = np.exp(values["spread"])
        mass = sigmoid(values["perfect"])

        drawn = sigmoid(rng.normal(centre, width))

        return np.where(rng.random(np.shape(mass)) < mass, 1.0, drawn)

    def mean(self, values, nodes=15):
        """No closed form, so the integral is taken by quadrature."""
        points, weights = np.polynomial.hermite.hermgauss(nodes)
        weights = weights / math.sqrt(math.pi)

        centre = values["location"]
        width = np.exp(values["spread"])
        mass = sigmoid(values["perfect"])

        total = np.zeros(np.shape(centre))

        for point, weight in zip(points, weights):
            total = total + weight * sigmoid(centre + SQRT2 * width * point)

        return mass + (1.0 - mass) * total

    def start(self, x):
        inside = logit(np.clip(x[x < 1.0], *INSIDE))
        mass = min(max(float((x >= 1.0).mean()), 1e-4), 0.5)

        return {
            "location": float(inside.mean()),
            "spread": math.log(max(float(inside.std()), 1e-3)),
            "perfect": float(logit(np.array(mass))),
        }


class BetaMixtureAtOne(Family):
    """
    Two betas and a point mass at 1: the run that went as it usually does,
    and the run that fell apart.

    The second component sits below the first by an amount the fit
    chooses, and how often it happens is allowed to fall with skill and
    not to rise. That is the shape a single beta cannot make: a tight
    cluster with a separate lump of bad runs under it, rather than one
    long tail stretched to cover both.
    """

    name = "beta-mixture"
    kernel = staticmethod(beta_mixture_kernel)

    channels = (
        Channel("location", "rising", 2.0,
                "logit of the mean accuracy of a run that goes normally"),
        Channel("steadiness", "free", 50.0,
                "log concentration of that component"),
        Channel("slip_drop", "free", 50.0,
                "how far below it a bad run lands, on the logit scale"),
        Channel("slip_steadiness", "free", 50.0,
                "log concentration of the component that goes wrong"),
        Channel("slip_chance", "falling", 2.0,
                "logit of how often a run goes that way"),
        Channel("perfect", "rising", 2.0,
                "logit of the chance the score is exactly 100%"),
    )

    def parts(self, x, values):
        drop = softplus(values["slip_drop"])

        clean = sigmoid(values["location"])
        slipped = sigmoid(values["location"] - drop)

        return (
            clean,
            np.exp(values["steadiness"]),
            slipped,
            np.exp(values["slip_steadiness"]),
            sigmoid(values["slip_chance"]),
            sigmoid(values["perfect"]),
            np.asarray(x) >= 1.0,
            drop,
        )

    def log_density(self, x, values):
        clean, hold, slipped, slip_hold, chance, mass, atom, _ = self.parts(
            x, values
        )
        inside = np.clip(x, *INSIDE)

        return np.where(
            atom,
            np.log(mass),
            np.log1p(-mass) + np.logaddexp(
                np.log1p(-chance) + beta_log_density(inside, clean, hold),
                np.log(chance) + beta_log_density(inside, slipped, slip_hold),
            ),
        )

    def gradient(self, x, values):
        clean, hold, slipped, slip_hold, chance, mass, atom, drop = self.parts(
            x, values
        )
        inside = np.clip(x, *INSIDE)

        left = np.log(inside)
        right = np.log1p(-inside)

        good = np.log1p(-chance) + beta_log_density(inside, clean, hold)
        bad = np.log(chance) + beta_log_density(inside, slipped, slip_hold)

        # How much of this score the falling-apart component accounts for.
        blame = sigmoid(bad - good)

        def piece(mean, concentration):
            a = mean * concentration
            b = (1.0 - mean) * concentration

            lean = digamma(b) - digamma(a) + left - right
            tighten = (
                digamma(concentration)
                - mean * digamma(a) - (1.0 - mean) * digamma(b)
                + mean * left + (1.0 - mean) * right
            )

            return (
                concentration * mean * (1.0 - mean) * lean,
                concentration * tighten,
            )

        clean_lean, clean_tighten = piece(clean, hold)
        slip_lean, slip_tighten = piece(slipped, slip_hold)

        def kept(value):
            return np.where(atom, 0.0, value)

        return {
            # The location moves both components, since the second is
            # written as an offset from the first.
            "location": kept((1.0 - blame) * clean_lean + blame * slip_lean),
            "steadiness": kept((1.0 - blame) * clean_tighten),
            "slip_drop": kept(-blame * slip_lean * sigmoid(
                values["slip_drop"]
            )),
            "slip_steadiness": kept(blame * slip_tighten),
            "slip_chance": kept(blame - chance),
            "perfect": np.where(atom, 1.0 - mass, -mass),
        }

    def cdf(self, x, values):
        clean, hold, slipped, slip_hold, chance, mass, _, _ = self.parts(
            x, values
        )
        point = np.clip(x, 0.0, 1.0)

        spread = (
            (1.0 - chance) * incomplete_beta(
                clean * hold, (1.0 - clean) * hold, point
            )
            + chance * incomplete_beta(
                slipped * slip_hold, (1.0 - slipped) * slip_hold, point
            )
        )

        return np.where(np.asarray(x) >= 1.0, 1.0, (1.0 - mass) * spread)

    def below(self, values):
        return 1.0 - sigmoid(values["perfect"])

    def sample(self, values, rng):
        drop = softplus(values["slip_drop"])

        clean = sigmoid(values["location"])
        slipped = sigmoid(values["location"] - drop)
        hold = np.exp(values["steadiness"])
        slip_hold = np.exp(values["slip_steadiness"])
        chance = sigmoid(values["slip_chance"])
        mass = sigmoid(values["perfect"])

        fell = rng.random(np.shape(chance)) < chance

        mean = np.where(fell, slipped, clean)
        concentration = np.where(fell, slip_hold, hold)

        drawn = rng.beta(mean * concentration, (1.0 - mean) * concentration)

        return np.where(rng.random(np.shape(mass)) < mass, 1.0, drawn)

    def mean(self, values):
        drop = softplus(values["slip_drop"])

        clean = sigmoid(values["location"])
        slipped = sigmoid(values["location"] - drop)
        chance = sigmoid(values["slip_chance"])
        mass = sigmoid(values["perfect"])

        return mass + (1.0 - mass) * (
            (1.0 - chance) * clean + chance * slipped
        )

    def start(self, x):
        single = BetaAtOne().start(x)

        return {
            "location": single["location"],
            "steadiness": single["steadiness"] + 0.7,
            "slip_drop": float(inverse_softplus(np.array(1.0))),
            "slip_steadiness": single["steadiness"] - 0.7,
            "slip_chance": float(logit(np.array(0.15))),
            "perfect": single["perfect"],
        }


FAMILIES = {
    family.name: family
    for family in (BetaAtOne(), LogitNormalAtOne(), BetaMixtureAtOne())
}
