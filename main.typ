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
    item-specific conditional distributions
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
    observed quantity is the accuracy of a submitted score rather than the
    points awarded for it, and in which the accuracy is modelled as it
    stands rather than after transformation. Each player is assigned a
    scalar latent ability. Each item, defined as a beatmap under a fixed
    modifier combination, is assigned a conditional distribution over
    accuracy indexed by ability, drawn from a family that the estimation
    procedure selects rather than fixes in advance. The ability model
    consumes that distribution through a fixed interface and is therefore
    independent of the family chosen. We describe the parameterisation,
    the variational objective, the identifiability argument that licenses
    interpretation of the ability scale, the criterion by which a family
    is selected, and the model's known deficiencies.
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
standard deviation is large relative to the differences of interest. On
items of low difficulty the awarded points are nearly invariant to
ability, so that such items contribute no information about the ordering
of players while receiving the same weight as any other item. The
departure of the fitted item effects from pp correlates with beatmap
duration and with the number of times a beatmap has been played, both
properties of the reward formula and of the sampling design rather than
of difficulty.

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

All observed cells enter the analysis. The two kinds of cell are not
observed under the same mechanism. A hundred-best list is truncated at
the player's hundredth-best score measured in points, so that membership
of the list depends on the outcome; a probe is issued before the outcome
is known, and membership of the probed panel therefore does not.
Restriction to probed cells would remove the first mechanism at the cost
of the majority of the observations, and would leave the second, namely
the player's choice of what to attempt, in place regardless. The origin
of each cell is recorded and is available to a selection model, which
@sec-limits identifies as the principal outstanding component.

= Model

== Response variable

Let $a_(i j) in (0, 1]$ denote the accuracy of the score of player $i$ on
item $j$. It is modelled as it stands. A transformation of the response
followed by a Gaussian conditional distribution imposes two assumptions
where one suffices, and the choice of transformation is not identified by
anything the data can be asked; the shape of the conditional distribution,
by contrast, is a question the data can answer, and @sec-selection puts
it to them.

Accuracy is bounded above and a non-negligible proportion of submitted
scores attain the bound exactly. The conditional distribution of
$a_(i j)$ is therefore specified with respect to the dominating measure

$ nu = lambda_((0,1)) + delta_1, $ <eq-measure>

the sum of Lebesgue measure on the open unit interval and a unit point
mass at one. Every candidate family below is a density with respect to
@eq-measure, comprising an absolutely continuous part on $(0,1)$ and an
atom at unity. Log-likelihoods computed against @eq-measure are
comparable across those families and are not comparable with a
log-likelihood computed against any other measure, a point that bears on
@sec-selection.

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

== Channels <sec-channels>

Item $j$ is characterised not by a scalar difficulty but by a conditional
distribution
$D_(j)(theta)$ over accuracy. A family fixes the functional form of that
distribution and declares the quantities it requires. Each such quantity
is a channel: a real-valued function of ability, carried per item, and
supplied to the family at whatever ability is being evaluated.

A channel declares one of three restrictions. A channel may be required
to be non-decreasing in ability, required to be non-increasing, or left
unconstrained. Restrictions are imposed where the substantive meaning of
the channel demands them: the location of the conditional distribution
must not fall as ability rises, whereas its dispersion is left free,
since an item may discriminate within a band of ability while players
below that band fail uniformly and players above it succeed uniformly.

All channels are constructed from a common basis. Fix $K$ knots
$t_1 < dots.c < t_K$, equally spaced with separation $w$, and define at
each knot the logistic ramp

$ p_k (theta) = (1 + exp(-(theta - t_k) \/ w))^(-1). $ <eq-ramp>

A non-decreasing channel is an offset plus a non-negative combination of
ramps,

$ c_j (theta) = ell_j + sum_(k=1)^K g_(j k) p_k (theta),
  quad g_(j k) = log(1 + exp(u_(j k))), $ <eq-rising>

in which $ell_j in RR$ and $u_(j k) in RR$ are free. Since
$g_(j k) >= 0$ by construction and each $p_k$ is increasing, $c_j$ is
non-decreasing at every admissible parameter value; the restriction is
imposed by the parameterisation and requires no constrained optimisation.
A non-increasing channel is the negation of @eq-rising.

An unconstrained channel is specified using the derivatives of the same
ramps, normalised to unit maximum,

$ phi_k (theta) = 4 p_k (theta)(1 - p_k (theta)), $

so that

$ c_j (theta) = ell_j + sum_(k=1)^K e_(j k) phi_k (theta), $ <eq-free>

with $ell_j in RR$ and $e_(j k) in RR$ free and unsigned. Outside the
knot range @eq-free returns to $ell_j$, which is the appropriate
behaviour for a quantity with no reason to trend. Every channel is
differentiable in $theta$ in closed form, a fact @eq-objective and
@sec-information both rely upon.

== Candidate families <sec-families>

Three families are implemented. Each specifies an atom
$omega_j (theta) in (0,1)$ at unit accuracy through a non-decreasing
channel, the probability of a maximal score being taken not to fall with
ability, and distributes the remaining mass over $(0,1)$.

The first is a beta distribution parameterised by its mean and
concentration. Writing $m_j (theta) in (0,1)$ for the mean and
$kappa_j (theta) > 0$ for the concentration, the density with respect to
@eq-measure is

$ f_j (a | theta) = cases(
  omega_j (theta) & a = 1,
  (1 - omega_j (theta)) dot "Beta"(a; m_j kappa_j, (1 - m_j) kappa_j)
    quad & a in (0,1),
) $ <eq-beta>

with $m_j$ the logistic transform of a non-decreasing channel and
$kappa_j$ the exponential of an unconstrained one. The mean is the
natural carrier of the monotonicity restriction under this
parameterisation.

The second is a normal distribution on the logit of accuracy. With
location $eta_j (theta)$ non-decreasing and scale $tau_j (theta) > 0$
unconstrained, the continuous part is

$ f_j (a | theta) = (1 - omega_j (theta)) dot
  (phi.alt((op("logit") a - eta_j) \/ tau_j)) / (tau_j a (1 - a)),
  quad a in (0,1), $ <eq-logitnormal>

$phi.alt$ denoting the standard normal density and the denominator the
Jacobian of the logit, which renders @eq-logitnormal a density on
accuracy rather than on its transform. Its dispersion is a free parameter
rather than a quantity tied to its location, and its tails near either
bound are heavier than those of @eq-beta.

The third is a two-component mixture of beta distributions, intended to
separate a run that proceeds as the player's ability implies from one
that does not. The second component is located below the first by an
amount $d_j (theta) > 0$ on the logit scale, carries its own
concentration, and is entered with probability $pi_j (theta)$ supplied by
a non-increasing channel, the frequency of such runs being taken not to
rise with ability. The family requires six channels against the three of
@eq-beta and @eq-logitnormal.

== Interface to the ability model <sec-interface>

The ability model does not observe which family is in use. It requires of
$D_j (theta)$ only

$ log f_j (a | theta), quad F_j (a | theta), quad F_j^(-1)(u | theta),
  quad "a draw from" D_j (theta), quad
  partial_theta log f_j (a | theta), $ <eq-interface>

namely a log-density with respect to @eq-measure, a distribution
function, a quantile function, a sampling routine and the derivative of
the log-density in ability. A family supplies the first, the second, the
fourth and the partial derivatives of the log-density with respect to
each of its channels; the quantile function is obtained by inversion of
$F_j$ and @eq-interface by the chain rule through @eq-rising and
@eq-free. Replacing a family, or introducing one over a richer outcome
space than accuracy alone, therefore requires no modification of the
estimation procedure.

= Estimation

== Variational objective

Let $O subset.eq {1, dots, I} times {1, dots, J}$ denote the set of
observed cells, $I$ the number of players and $J$ the number of items.
Estimation minimises

$ L = sum_((i,j) in O) EE_(theta ~ Q_i) [-log f_j (a_(i j) | theta)]
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
$p_k$, and hence every channel and every fitted value, unchanged.
Location is free in the same way. The ability scale is therefore not
estimable from the likelihood, and interpretation of $theta_i$ in units
of population dispersion, or of $Phi(theta_i)$ as a population quantile,
is unavailable without further restriction.

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
proportion $Phi(theta_i)$ is accordingly a quantile of the panel.

== Partial pooling across items

The panel contains items observed on few players, for which the $K + 1$
parameters of each channel are not separately estimable. Each item is
shrunk towards the panel mean of the corresponding parameter,

$ R_"item" = sum_(c) lambda_c sum_(j,k) (v^c_(j k) - macron(v)^c_k)^2
           + lambda_ell sum_(c, j) (ell^c_j - macron(ell)^c)^2, $ <eq-pool>

in which $c$ indexes the channels of the family in use, $v^c_(j k)$
denotes the shape parameter of channel $c$ at knot $k$ for item $j$,
whether $u_(j k)$ of @eq-rising or $e_(j k)$ of @eq-free, and a bar
denotes the mean over items. The gradient of the first term with respect
to a single $v^c_(l k)$ is $2 lambda_c (v^c_(l k) - macron(v)^c_k)$, the
contributions through $macron(v)^c_k$ cancelling because deviations from
a mean sum to zero.

The coefficient $lambda_c$ is declared per channel and per family rather
than shared. The requirement is not uniform across channels: a dispersion
channel weakly penalised will contract onto the observed scores of a
sparsely observed item and claim a predictive density no withheld score
can attain, whereas the same coefficient applied to a location channel
collapses the response of all items onto a common shape. The coefficient
$lambda_ell$ governs the per-item level of every channel and is shared.

== Numerical integration

The expectation in @eq-objective admits no closed form, the channels
being non-linear in $theta$. It is evaluated by Gauss--Hermite quadrature,

$ EE_(theta ~ cal(N)(mu, sigma^2))[f(theta)]
  approx sum_(q=1)^Q W_q f(mu + sqrt(2) sigma x_q), $ <eq-quad>

with abscissae $x_q$ and weights $W_q = w_q \/ sqrt(pi)$ normalised to
sum to unity.

== Optimisation

All parameters, namely $mu_i$ and $log sigma_i$ for each player and, for
each item, the level and the $K$ shape parameters of every channel, are
optimised jointly by Adam. The quantity $log sigma_i$ is confined to a
bounded interval, a posterior width approaching zero reducing @eq-quad to
evaluation at a point and eliminating the gradient with respect to
$sigma_i$.

Gradients of @eq-objective are derived analytically rather than by
automatic differentiation. Each family supplies the partial derivatives
of its log-density with respect to its own channels; the remainder of the
gradient, namely the propagation through @eq-rising and @eq-free to the
item parameters and through @eq-quad to $mu_i$ and $log sigma_i$, is
common to all families and is written once.

The inner loop is compiled. Evaluated in array form, the calculation
materialises every intermediate quantity across all observations
simultaneously, and the resulting memory traffic dominates the arithmetic.
Each family therefore supplies its log-density and channel gradients
twice: once in array form, used for reporting and for quantities computed
outside the descent, and once as a scalar routine compiled ahead of the
first iteration, used within it. The compiled loop evaluates the ramps at
one observation's ability, assembles that observation's channels, obtains
the log-density and its gradient, and accumulates into both the item and
the player parameters without leaving registers. Observations are
partitioned into contiguous blocks, one per thread, each accumulating
into a private gradient buffer, the buffers being summed on completion.

Two verifications are available under `--check-gradient`. The analytic
gradient of @eq-objective is compared with central differences at
randomly chosen coordinates. Separately, and for every family rather than
only the one in use, the compiled scalar routine is compared with the
array implementation from which it was written, since the descent
exercises only the former and every reported quantity only the latter.

= Discriminating power <sec-information>

The contribution of an item to the location of a player is quantified by
the Fisher information carried by a single score,

$ I_j (theta) = EE_(a ~ D_j (theta))
  [(partial_theta log f_j (a | theta))^2], $ <eq-information>

evaluated by sampling from $D_j (theta)$, which @eq-interface provides.
An item for which $I_j$ is near zero over the region occupied by its
observed players yields scores nearly uninformative about ability.

@eq-information replaces the derivative of a conditional mean, which was
the corresponding quantity under a model with a scalar response. A
derivative of that kind is expressed in units of the scale on which the
response was recorded and is therefore not comparable across a change of
that scale; the response here is the accuracy itself and no such scale
remains. @eq-information is expressed in units of precision, so that an
item attaining unity contributes as much as the prior $P$ of
@sec-scale in locating a player who has attempted it.

The same quantity governs incremental revision. Writing
$l(theta) = log f_j (a_(i j) | theta)$ and taking
$g = l'(mu_i)$ and $h = -l''(mu_i)$, a Laplace step about $mu_i$ gives

$ sigma_"new"^(-2) = sigma_i^(-2) + h, quad
  mu_"new" = mu_i + g \/ (sigma_i^(-2) + h). $ <eq-laplace>

The quantity $h$ is a realised curvature rather than its expectation, and
a distribution with an atom against a bound is not log-concave throughout;
$h$ may therefore be negative and drive the revised precision to zero or
below, in which case @eq-laplace is rejected and the posterior stands.
Where the score is consequential, re-estimation against @eq-objective is
to be preferred to a step.

= Selection of the family <sec-selection>

The family is not fixed by assumption. Candidates are compared on
withheld cells by two criteria.

The first is the mean log predictive density on cells withheld at random,
the predictive density being obtained by integrating $f_j$ against the
fitted posterior $Q_i$ by @eq-quad. Comparison is legitimate because
every candidate of @sec-families is a density with respect to the common
measure @eq-measure. A model of the response on any other scale cannot be
entered into the same comparison, its log-density carrying the units of
whatever measure it was taken against.

The second is calibration. For each withheld cell define the probability
integral transform

$ u_(i j) = EE_(theta ~ Q_i)[F_j (a_(i j) | theta)]. $ <eq-pit>

Under a correctly specified model the $u_(i j)$ are marginally uniform on
$(0,1)$ irrespective of item and player. Two summaries are reported: the
mean of $u_(i j)$, which should equal one half and whose departure
indicates the direction of the error, and the largest absolute deviation
of their empirical distribution function from that of the uniform.

A maximal score requires care under @eq-pit. Such a score occupies no
single position within its predicted distribution, the distribution
function jumping at unity by $omega_j$, and assigning it the upper
endpoint would concentrate a fixed proportion of the transform at one and
destroy the uniformity on which the diagnostic rests. A point is
therefore drawn uniformly from the interval spanned by the jump. Where a
deterministic value is required, as when the diagnostic is displayed and
must not vary between evaluations, the midpoint of that interval is taken
instead, at the cost of the property just described.

Two classes of baseline are evaluated alongside the candidates, each
fitted under the same family and therefore against the same measure.
The first ignores ability entirely, comprising a single distribution
fitted to the whole panel and a distribution per item. The second is
fitted to the official quantities alone, namely the global pp of the
player and the star rating of the item under the modifier combination
played, entered as a linear predictor for each channel. The second is the
comparison of principal interest, being the predictive performance
attainable without observation of the score. An item lacking a star
rating is not discarded from that comparison; the rating and its
interaction are set to zero and an indicator column admitted, so that the
baseline estimates what the official quantities are worth when they are
silent, and all candidates are evaluated on an identical set of cells.

The star rating enters no part of the model itself. It is a covariate of
the baseline alone, and no withheld set is restricted to the cells on
which it happens to be available.

= Limitations <sec-limits>

Four deficiencies are known.

First, the outcome modelled is a scalar, whereas the realised outcome of
a play is not. Pass and failure, maximum combo and the counts of each
judgement each behave differently and encounter distinct boundaries. The
appropriate specification is a distribution over the vector of those
quantities. This is the deficiency most readily remedied, the interface
@eq-interface being indifferent to the outcome space over which
$D_j (theta)$ is defined.

Second, the composition of $O$ is not modelled. Membership of a
hundred-best list is truncated on points, and the choice of the player as
to what to attempt depends on the anticipated outcome. Selection of this
kind is not ignorable and its omission biases the item parameters. The
origin of each cell is recorded, so that the information a selection
model would require is available; no such model is specified.

Third, the constraint @eq-pop fixes the first two moments of the fitted
population and leaves higher moments free. Where the number of
observations per player is large the likelihood dominates the prior, the
realised distribution departs appreciably from standard normal, and the
interpretation of $Phi(theta_i)$ as a quantile is correspondingly
weakened.

Fourth, the atom channel of @sec-families is estimated with a level per
item but with substantially no dependence on ability, the fitted
$omega_j (theta)$ being nearly constant across the ability range for
almost every item. The observed frequency of maximal scores does depend
on ability, and markedly so within a single item, so that the fitted
behaviour reflects the estimation procedure rather than the data. Whether
the cause is the penalty of @eq-pool applied to a channel whose gradient
is small, the initialisation of that channel near a flat configuration,
or both, has not been determined. Until it is, the third channel of each
family should be read as an item-level rate and not as a function of
ability.
