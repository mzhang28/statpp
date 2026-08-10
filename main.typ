#set page(paper: "a4", margin: (x: 3.4cm, y: 3.0cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.68em, spacing: 1.1em, first-line-indent: 1.4em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): it => block(
  above: 1.7em, below: 0.8em,
  text(size: 12pt, weight: "bold", it),
)
#show heading.where(level: 2): it => block(
  above: 1.3em, below: 0.6em,
  text(size: 10.5pt, weight: "bold", style: "italic", it),
)

#align(center)[
  #text(size: 15pt, weight: "bold")[
    A latent-trait model for osu! performance with\
    item-specific response and dispersion functions
  ]
  #v(1.2em)
]

#block(inset: (x: 2.2em))[
  #set par(first-line-indent: 0em)
  #text(size: 9.5pt)[
    *Abstract.* Performance points (pp) are simultaneously the reward a
    beatmap confers and the only widely available measure of its
    difficulty, so that any attempt to identify beatmaps overvalued
    relative to their difficulty is circular if difficulty is itself
    estimated from pp. We specify an item-response model in which the
    observed quantity is the accuracy of a submitted score rather than
    the points awarded for it. Each player is assigned a scalar latent
    ability; each item, defined as a beatmap under a fixed modifier
    combination, is assigned a monotone response function giving expected
    accuracy as a function of ability, together with a strictly positive
    dispersion function giving the conditional standard deviation at that
    ability. Both are constructed from logistic ramps at fixed knots and
    are partially pooled across items. Inference proceeds by minimisation
    of a variational objective in which the conditional log-likelihood is
    integrated against each player's posterior by Gauss--Hermite
    quadrature. We give the identifiability argument that licenses
    interpretation of the ability scale, report the held-out predictive
    comparison against pp-based baselines, and state the model's known
    deficiencies.
  ]
]

#v(1.4em)

= Introduction

The osu! performance-points system assigns to each submitted score a
scalar reward computed from the star rating of the beatmap and the
quality of the play. A beatmap is commonly described as farm when it
confers more points than its difficulty warrants. Quantifying that
description requires a measure of difficulty obtained independently of
the points awarded, since a difficulty estimate derived from pp and then
compared against pp constitutes a single measurement compared with
itself.

An earlier component of this project fits the additive decomposition
$p_(i j) approx alpha_i + delta_j$, in which $p_(i j)$ denotes the points
awarded to player $i$ on item $j$, $alpha_i$ a player effect and
$delta_j$ an item effect. Diagnostics on the collected panel indicate
that this specification is inadequate in three respects. The residual
standard deviation is approximately 80 points, which is large relative to
the differences of interest. On items of low difficulty the awarded
points are nearly invariant to ability, so that such items contribute no
information about the ordering of players while receiving the same weight
as any other item. The departure of the fitted item effects from pp
correlates with beatmap duration and with the number of times a beatmap
has been played, both properties of the reward formula and of the
sampling design rather than of difficulty.

We therefore replace the observation model. The quantity modelled is the
accuracy of the score, which pp does not enter, so that a comparison
between fitted item difficulty and pp is a comparison of two measurements
of distinct quantities.

= Data

The panel is assembled by `sample.py`, which draws players from
logarithmically spaced pages of the global and per-country performance
rankings. The hundred best submitted scores of each sampled player are
retrieved, and further cells are obtained by direct request for a named
(player, beatmap) pair. We refer to the latter as probed cells.

An item is the pair (beatmap, modifier combination), the modifier
combination being reduced to its difficulty-relevant elements: Nightcore
is identified with Double Time, and modifiers that do not alter the
demands of the play are discarded. Scores recorded under modifier
settings that alter the beatmap are excluded.

The analyses below use probed cells only. A hundred-best list is
truncated at the player's hundredth-best score measured in points, so
that membership of the list depends on the outcome; a probe is issued
before the outcome is known, and membership of the probed panel therefore
does not. The restriction removes truncation on the response at the cost
of a smaller panel.

Two panels appear below. The predictive comparison of @sec-predictive was
computed on a frozen copy of the database containing 1,397 players, 500
items and 18,808 observations. The quantities of @sec-behaviour were
computed on a later panel of 1,648 players, 1,243 items and 48,883
observations, the sampler having continued to run in the interim.

= Model

== Response variable

Let $a_(i j) in (0, 1]$ denote the accuracy of the score of player $i$ on
item $j$. Accuracy is bounded above and the distribution of submitted
scores is concentrated near that bound, so that differences between
strong players occupy a small interval on the raw scale while
corresponding to large differences in difficulty of attainment. We
therefore model the transformed response

$ y_(i j) = -log_10 (max(1 - a_(i j), epsilon)) $ <eq-response>

with $epsilon = 5 times 10^(-4)$. Under this transformation an accuracy
of $0.9$ maps to $y = 1$, an accuracy of $0.99$ to $y = 2$ and an
accuracy of $0.999$ to $y = 3$; a unit increment corresponds to a
tenfold reduction in the complement of accuracy. Truncation at $epsilon$
is required because a maximal score gives $1 - a_(i j) = 0$. The chosen
value corresponds to approximately half of one 100-judgement on a beatmap
of 700 objects and therefore lies below the complement of accuracy of any
score other than a maximal one.

== Latent ability

Each player is characterised by a scalar $theta_i in RR$, interpreted as
ability, larger values denoting greater ability. In place of a point
estimate we carry a variational posterior

$ Q_i = cal(N)(mu_i, sigma_i^2), $

whose parameters $mu_i$ and $log sigma_i$ are free. The prior is
$P = cal(N)(0, 1)$. The width $sigma_i$ admits interpretation as the
precision with which the panel locates player $i$: it remains near its
prior value when few scores are available and contracts as evidence
accumulates.

== Item response and dispersion functions

Item $j$ is characterised by two functions of ability rather than by a
scalar difficulty. The response function $m_j : RR -> RR$ gives the
conditional expectation of $y$; the dispersion function
$s_j : RR -> RR_(>0)$ gives the conditional standard deviation. Only
$m_j$ is constrained to be non-decreasing. The dispersion is left free to
rise and fall across the ability range, since an item may discriminate
within a band of ability while players below that band fail uniformly and
players above it succeed uniformly.

Both are constructed from a common basis. Fix $K$ knots
$t_1 < dots.c < t_K$, equally spaced with separation $w$, and define at
each knot the logistic ramp

$ p_k (theta) = (1 + exp(-(theta - t_k) \/ w))^(-1). $ <eq-ramp>

The response function is an offset plus a non-negative combination of
ramps,

$ m_j (theta) = a_j + sum_(k=1)^K c_(j k) p_k (theta),
  quad c_(j k) = log(1 + exp(u_(j k))), $ <eq-mean>

in which $a_j in RR$ and $u_(j k) in RR$ are free. Since $c_(j k) >= 0$
by construction and each $p_k$ is increasing, $m_j$ is non-decreasing at
every admissible parameter value; monotonicity is imposed by the
parameterisation and requires no constrained optimisation. The derivative
is available in closed form,

$ m'_j (theta) = w^(-1) sum_(k=1)^K c_(j k) p_k (theta)(1 - p_k (theta)), $ <eq-slope>

and admits interpretation as the discriminating power of item $j$ at
ability $theta$, being the increase in expected transformed accuracy per
unit increase in ability. An item for which $m'_j$ is near zero over the
region occupied by its observed players yields scores nearly
uninformative about ability.

The dispersion function is specified on the logarithmic scale using the
derivatives of the same ramps, normalised to unit maximum,

$ phi_k (theta) = 4 p_k (theta)(1 - p_k (theta)), $

so that

$ log s_j (theta) = b_j + sum_(k=1)^K e_(j k) phi_k (theta), $ <eq-spread>

with $b_j in RR$ and $e_(j k) in RR$ free and unsigned. Exponentiation
guarantees positivity at every parameter value.

== Observation model

Conditional on ability, the response is taken to be Gaussian:

$ y_(i j) | theta ~ cal(N)(m_j (theta), s_j (theta)^2). $ <eq-likelihood>

@eq-likelihood is provisional. The realised outcome of a play is
not a single scalar: pass and failure, combo, judgement counts and
accuracy each behave differently and encounter distinct boundaries. The
interface between the item model and the ability model is confined to the
map $p_j : RR -> "Dist"(cal(X)_j)$, and the ability model consumes only
the conditional log-likelihood together with its first two derivatives in
$theta$, so that @eq-likelihood may be replaced without consequence
elsewhere.

= Estimation

== Variational objective

Let $O subset.eq {1, dots, I} times {1, dots, J}$ denote the set of
observed cells, $I$ the number of players and $J$ the number of items.
Estimation minimises

$ L = sum_((i,j) in O) EE_(theta ~ Q_i) [-log p_j (y_(i j) | theta)]
    + sum_(i=1)^I "KL"(Q_i || P)
    + R_"pop" + R_"item", $ <eq-objective>

in which the first term is the expected conditional negative
log-likelihood under the player's own posterior rather than at a point
estimate, so that a player located imprecisely by the panel contributes a
correspondingly diffuse likelihood to the item parameters. The
Kullback--Leibler divergence between univariate Gaussians is available in
closed form,

$ "KL"(cal(N)(mu, sigma^2) || cal(N)(0,1))
  = 1/2 (sigma^2 + mu^2 - 1) - log sigma. $

== Identifiability of the ability scale <sec-scale>

The likelihood alone does not determine the scale of $theta$. Under the
reparameterisation $theta arrow.r kappa theta$ with $kappa > 0$, the
substitutions $t_k arrow.r kappa t_k$ and $w arrow.r kappa w$ leave every
$p_k$, and hence every fitted value, unchanged. Location is free in the
same way. The ability scale is therefore not estimable from
@eq-likelihood, and interpretation of $theta_i$ in units of population
dispersion, or of $Phi(theta_i)$ as a population quantile, is unavailable
without further restriction.

The penalty $R_"pop"$ supplies that restriction. Writing
$macron(mu) = I^(-1) sum_i mu_i$ and
$V = I^(-1) sum_i (mu_i - macron(mu))^2 + I^(-1) sum_i sigma_i^2$ for the
mean and the total variance of the fitted population, we take

$ R_"pop" = lambda_p I (macron(mu)^2 + (V - 1)^2). $ <eq-pop>

The divergence term of @eq-objective already penalises departure of each
individual posterior from $cal(N)(0,1)$; @eq-pop constrains the first two
moments of the aggregate. Under these restrictions the fitted population
has mean zero and unit variance, $theta_i$ is measured in units of
population standard deviation, and $Phi(theta_i)$ estimates the
proportion of the population below player $i$.

The population so constrained is the sampled panel and not the playing
population. Players are drawn from logarithmically spaced ranking pages,
a design that over-represents high ability by construction. The
proportion $Phi(theta_i)$ is accordingly a quantile of the panel, and the
top-fifty stratum is located near the 87th percentile of the panel rather
than near the upper extreme of the game.

== Partial pooling across items

The panel contains items observed on few players, for which the twelve
shape parameters of @eq-mean and @eq-spread are not separately estimable.
Each item is shrunk towards the panel mean of the corresponding
parameter,

$ R_"item" = lambda_c sum_(j,k) (u_(j k) - macron(u)_k)^2
           + lambda_s sum_(j,k) (e_(j k) - macron(e)_k)^2
           + lambda_l sum_j [(a_j - macron(a))^2 + (b_j - macron(b))^2], $ <eq-pool>

a bar denoting the mean over items. The gradient of the first term with
respect to $u_(l k)$ is $2 lambda_c (u_(l k) - macron(u)_k)$, the
contributions through $macron(u)_k$ cancelling because deviations from a
mean sum to zero.

The separation of $lambda_c$ from $lambda_s$ is material and is
documented in @sec-hyper. The two were initially a single coefficient,
under which no value proved satisfactory: at small values the fitted
dispersion of a sparsely observed item contracts onto its own observed
scores and the held-out predictive density degrades severely, while at
values large enough to prevent this the response functions of all items
collapse onto a common shape.

== Numerical integration

The expectation in @eq-objective admits no closed form, $m_j$ and $s_j$
being non-linear in $theta$. It is evaluated by Gauss--Hermite quadrature,

$ EE_(theta ~ cal(N)(mu, sigma^2))[f(theta)]
  approx sum_(q=1)^Q W_q f(mu + sqrt(2) sigma x_q), $ <eq-quad>

with abscissae $x_q$ and weights $W_q = w_q \/ sqrt(pi)$ normalised to
sum to unity. We take $Q = 7$.

== Optimisation

All parameters, namely $mu_i$ and $log sigma_i$ for each player and
$a_j$, $u_(j k)$, $b_j$ and $e_(j k)$ for each item, are optimised
jointly by Adam. The quantity $log sigma_i$ is confined to
$[log 0.02, log 3]$, a posterior width approaching zero reducing
@eq-quad to evaluation at a point and eliminating the gradient with
respect to $sigma_i$.

The implementation depends on `numpy` alone and gradients of
@eq-objective are derived analytically rather than by automatic
differentiation. Invoking the estimation routine with `--check-gradient`
compares them with central differences at randomly chosen coordinates.
The largest relative discrepancy observed is $5.5 times 10^(-8)$ for the
gradient of the objective, $5.8 times 10^(-9)$ for the first derivative
of a single score's log-likelihood in $theta$ and $1.0 times 10^(-6)$ for
the second, the last being consistent with the truncation error of a
second-order central difference.

= Fixed quantities <sec-hyper>

@tab-hyper records each fixed quantity, its value and the basis on which
it was set. Where a value was selected by search, the criterion is the
mean held-out log predictive density defined in @sec-predictive, larger
values being preferred.

#v(0.4em)

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    align: (left, left, left),
    inset: (x: 0.5em, y: 0.42em),
    table.hline(),
    table.header([*Quantity*], [*Value*], [*Basis*]),
    table.hline(),

    [$epsilon$], [$5 times 10^(-4)$],
    [Fixed below the complement of accuracy attainable by any non-maximal
     score on a beatmap of typical object count.],

    [$K$], [6],
    [Search over ${3, 4, 6, 9}$ giving $-0.454$, $-0.456$, $-0.456$ and
     $-0.457$. The criterion is insensitive over this range.],

    [$t_1, t_K$], [$-2, +2$],
    [Fixed at two standard deviations of the constrained population.],

    [$w$], [$0.8$], [Determined by $K$ and the knot range.],

    [$Q$], [7], [Fixed.],

    [Adam iterations], [600],
    [Search against 2000, which alters the criterion by $0.001$.],

    [Adam step size], [$0.05$], [Fixed.],

    [$sigma$ bounds], [$[0.02, 3]$],
    [Fixed to exclude the degeneracy at zero width.],

    [$lambda_p$], [$1.0$],
    [Fixed. The realised population has mean $-0.012$ and standard
     deviation $0.999$.],

    [$lambda_c$], [$2.0$],
    [Search over ${0.5, 2, 10, 50}$ giving $-0.538$, $-0.456$, $-0.454$
     and $-0.455$. The smallest value within the plateau is preferred,
     larger values eliminating between-item variation in @eq-slope.],

    [$lambda_s$], [$50.0$],
    [Search over ${0.5, 2, 10, 50, 10^3}$ giving $-1.163$, $-0.517$,
     $-0.461$, $-0.455$ and $-0.458$. Under a single coefficient shared
     with $lambda_c$ at $0.1$ the criterion reaches $-146$.],

    [$lambda_l$], [$0.05$],
    [Fixed. Items differ substantially in mean response and in
     dispersion, and these parameters are estimable from the data.],
    table.hline(),
  ),
  caption: [Fixed quantities of the estimation procedure.],
) <tab-hyper>

= Empirical behaviour <sec-behaviour>

== Held-out predictive comparison <sec-predictive>

Fifteen per cent of observed cells were withheld at random, the model
estimated on the remainder and the mean log predictive density and root
mean squared error evaluated on the withheld cells. For the present model
the predictive density is obtained by integrating @eq-likelihood against
the fitted posterior $Q_i$ by @eq-quad. All candidates were evaluated on
an identical set of cells, restricted to those for which the covariates
required by the pp-based baseline are available.

#v(0.4em)

#figure(
  table(
    columns: (1fr, auto, auto),
    stroke: none,
    align: (left, right, right),
    inset: (x: 0.6em, y: 0.42em),
    table.hline(),
    table.header([*Predictor*], [*Log density*], [*RMSE*]),
    table.hline(),
    [Marginal Gaussian], [$-0.693$], [$0.484$],
    [Player pp and star rating], [$-0.634$], [$0.456$],
    [Player effect and item effect], [$-0.524$], [$0.401$],
    [The same, dispersion by item], [$-0.503$], [$0.401$],
    [Response function at $mu_i$], [$-0.495$], [$0.397$],
    [Response function under $Q_i$], [$bold(-0.456)$], [$bold(0.397)$],
    table.hline(),
  ),
  caption: [
    Held-out performance on 2,818 withheld cells. The response is
    @eq-response, so that both columns are in units of the transformed
    scale.
  ],
) <tab-predictive>

The second row is the comparison of principal interest, being the
predictive performance attainable from the official quantities alone,
namely the global pp of the player and the star rating of the item under
the modifier combination played, without observation of the score. The
proposed model improves upon it in both columns.

Two further comparisons are informative. The improvement from the third
row to the fourth is attributable to item-specific dispersion alone, the
conditional mean being unchanged. The improvement from the fifth row to
the sixth is attributable to integration against the posterior rather
than evaluation at its mode, and is of comparable magnitude, which
indicates that the variational treatment of ability is not a formal
refinement only.

== An item across the ability range

@tab-item reports the fitted response and dispersion for the most
frequently observed item of the panel at six values of ability. The
conditional mean is reported as accuracy, obtained by inverting
@eq-response.

#v(0.4em)

#figure(
  block(breakable: false)[
    #table(
      columns: 4,
      stroke: none,
      align: (right, right, right, left),
      inset: (x: 0.8em, y: 0.4em),
      table.hline(),
      table.header([$theta$], [$m_j (theta)$], [$s_j (theta)$],
                   [$m_j plus.minus s_j$]),
      table.hline(),
      [$-2$], [71.27%], [0.214], [52.97% to 82.45%],
      [$-1$], [93.72%], [0.322], [86.82% to 97.01%],
      [$0$],  [98.39%], [0.524], [94.61% to 99.52%],
      [$+1$], [99.35%], [0.636], [97.18% to 99.85%],
      [$+2$], [99.62%], [0.769], [97.77% to 99.94%],
      [$+3$], [99.72%], [0.788], [98.26% to 99.95%],
      table.hline(),
    )
  ],
  caption: [
    Fitted response and dispersion for beatmap 714001 without modifiers,
    observed on 840 players.
  ],
) <tab-item>

The two dispersion columns of @tab-item vary in opposite directions. The
conditional standard deviation on the transformed scale increases
monotonically from $0.214$ to $0.788$, while the corresponding interval
on the accuracy scale contracts from 29 percentage points to under two.
Both statements describe the same fitted object. The transformation
@eq-response expands the neighbourhood of unit accuracy, so that a fixed
increment of $s_j$ subtends a diminishing interval of accuracy as $m_j$
approaches the bound.

== Heterogeneity of dispersion

Evaluated at the median ability of the players observed on each item,
$s_j$ ranges over $[0.103, 0.877]$ across the panel with interquartile
range $[0.226, 0.360]$. Dispersion therefore differs between items by
approximately an order of magnitude, which accounts for the improvement
of the fourth row over the third in @tab-predictive.

Within an item, comparing $s_j$ one unit of ability below the median of
its observed players with the value one unit above, the median ratio is
$1.8$. Dispersion is thus not constant in ability and the specification
@eq-spread is not redundant.

== Calibration

For each observed cell define the probability integral transform

$ u_(i j) = EE_(theta ~ Q_i)
  [Phi((y_(i j) - m_j (theta)) \/ s_j (theta))]. $ <eq-pit>

Under a correctly specified model the $u_(i j)$ are marginally uniform on
$(0,1)$ irrespective of item and player, so that their histogram
constitutes a diagnostic requiring no held-out data. On the present panel
the histogram is close to level, which indicates that the fitted
dispersion is approximately correct in aggregate.

Stratifying @eq-pit by the discriminating power @eq-slope of the item
reveals a monotone departure. Scores on items of low discrimination fall
systematically below their predicted position and scores on items of high
discrimination above it, the mean of $u_(i j)$ departing from one half by
approximately $0.01$ at either extreme.

= Limitations

Three deficiencies are known.

First, the observation model @eq-likelihood is Gaussian over a single
scalar, whereas the realised outcome is multivariate and bounded. This is
the deficiency most readily remedied, for the reason given following
@eq-likelihood.

Second, the composition of $O$ is not modelled. Restriction to probed
cells removes truncation on points, but the choice of the player as to
what to play remains, and that choice depends on the anticipated outcome.
Selection of this kind is not ignorable and its omission biases the item
parameters.

Third, the constraint @eq-pop fixes the first two moments of the fitted
population and leaves higher moments free. On the panel restricted to
probed cells the realised distribution is close to standard normal, with
skewness $-0.20$, excess kurtosis $-0.07$ and largest absolute deviation
of the empirical distribution function from $Phi$ equal to $0.028$. On
the unrestricted panel, where the number of observations per player is
larger and the likelihood dominates the prior, the same deviation reaches
$0.115$ and the interpretation of $Phi(theta_i)$ as a quantile is
correspondingly weakened.
