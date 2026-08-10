#set page(paper: "a4", margin: (x: 3.2cm, y: 2.8cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.72em, spacing: 1.15em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => block(
  above: 1.7em, below: 0.75em,
  text(size: 13pt, weight: "bold", it),
)
#show heading.where(level: 2): it => block(
  above: 1.3em, below: 0.55em,
  text(size: 11pt, weight: "bold", it),
)

#align(center)[
  #text(size: 17pt, weight: "bold")[The skill and curve model]
  #v(0.3em)
  #text(size: 10pt)[What every symbol means, and where every constant came from]
]

#v(1.5em)

= The number a score turns into

A score has an accuracy. Accuracy is a fraction between 0 and 1. Almost
every score worth comparing sits above 0.9. The interesting differences
between strong players are in the last fraction of a percent.

That is awkward to model directly. The gap from 0.99 to 0.999 is small as
a number. It is large as an achievement. So the fit does not use accuracy
itself. It uses this:

$ y_(i j) = -log_10 (max(1 - a_(i j), 5 dot 10^(-4))) $

Here $a_(i j)$ is the accuracy player $i$ got on map $j$. The subscripts
are always in that order: player first, map second.

Read $y$ as the count of nines. An accuracy of 0.9 gives $y = 1$. An
accuracy of 0.99 gives $y = 2$. An accuracy of 0.999 gives $y = 3$. Each
whole step means the miss fraction fell by a factor of ten.

The $max$ exists because a perfect score gives $1 - a = 0$, and the
logarithm of zero is not a number. The floor of $5 dot 10^(-4)$ is about
half of one 100-hit on a map with 700 objects. No real score short of a
perfect one gets under it.

= What a player is

Every player has one number. It is written $theta_i$ and it is called
their skill. Higher means better. It ranges over the whole real line.

The fit is never certain about that number. A player with nine scores is
placed less firmly than a player with ninety. So the fit does not store
$theta_i$. It stores a belief about $theta_i$, which is a normal
distribution:

$ Q_i = cal(N)(mu_i, sigma_i^2) $

$mu_i$ is the middle of that belief, and it is the number shown as the
player's skill on the page. $sigma_i$ is the width of the belief, and it
is the number shown as their uncertainty. Both are free parameters. The
fit chooses them.

There is also a prior, which is what the fit assumes about a player before
it has seen any of their scores:

$ P = cal(N)(0, 1) $

== Why the skill scale means anything <scale>

Nothing in the paragraphs above says that $theta$ is measured in standard
deviations. It has no units at all so far.

Worse, the scale is not even pinned down. Suppose you double every
player's skill. You could then halve how fast every map's curve rises,
and every prediction would come out the same. The data cannot tell those
two arrangements apart.

Two terms in the objective fix that, and they appear in
@objective below. The first pulls each belief towards $cal(N)(0,1)$. The
second holds the whole fitted population at an average of zero and a
spread of one.

Once those hold, $theta$ is measured in standard deviations of the fitted
population, and $Phi(theta_i)$ reads as the share of that population
sitting below player $i$. The units are a consequence of those two terms.
Take them away and the units are gone.

One caveat travels with this. The population in question is the sampled
panel, and the panel was drawn from ranking pages spread up and down the
ladder rather than at random. It holds far more strong players than the
game does. A skill of zero is the middle of the panel and not the middle
of osu!.

= What a map is

A map is not one number in this model. It is two functions of skill.

The first is the curve, written $m_j (theta)$. It gives the expected value
of $y$ that a player of skill $theta$ produces on map $j$. It is required
to rise with skill and never fall.

The second is the spread, written $s_j (theta)$. It gives the standard
deviation of $y$ around that curve, again at skill $theta$. It is required
to stay positive and is otherwise free to rise and fall.

Both are functions rather than numbers, so a map can behave one way at low
skill and another way at high skill.

== The pieces both are built from

Fix $K$ points along the skill axis, called knots and written $t_k$. They
are evenly spaced. Give them a width $w$, which is the spacing between
neighbours.

At each knot put a ramp:

$ p_k (theta) = 1 / (1 + e^(-(theta - t_k) \/ w)) $

A ramp is near 0 well below its knot and near 1 well above it, and it
climbs smoothly in between.

== The curve

The curve is the map's floor plus a positive amount of every ramp:

$ m_j (theta) = a_j + sum_(k=1)^K op("softplus")(u_(j k)) dot p_k (theta) $

$a_j$ is the value the curve takes below the whole skill range, so it is
the map's floor. $u_(j k)$ is a free parameter, one per map per knot.

The softplus, $op("softplus")(u) = log(1 + e^u)$, is always positive.
That is what forces the curve to rise. A sum of increasing ramps with
positive weights can only increase.

The derivative has a closed form, and it is the quantity the page calls
"separates by":

$ m'_j (theta) = 1/w sum_(k=1)^K op("softplus")(u_(j k)) dot p_k (theta) (1 - p_k (theta)) $

It answers one question. If a player were one step better, how much more
$y$ would this map give them? A value near zero means the map gives
everyone about the same result, so a score on it says almost nothing
about who set it.

== The spread

Take the derivative of a ramp and you get a bump, which peaks at its own
knot and falls away on both sides. Scale it to peak at 1:

$ "bump"_k (theta) = 4 p_k (theta) (1 - p_k (theta)) $

The spread is built from those, in the logarithm:

$ log s_j (theta) = b_j + sum_(k=1)^K e_(j k) dot "bump"_k (theta) $

$b_j$ sets the map's overall level of scatter. $e_(j k)$ bends that level
up or down near knot $k$, and it may be any sign. Taking the exponential
of the result keeps $s_j$ positive whatever the parameters do.

= What one score looks like to the fit

Put the two together. Given a skill, a score on a map is drawn from a
normal distribution centred on the curve, with the spread as its standard
deviation:

$ y_(i j) | theta ~ cal(N)(m_j (theta), s_j (theta)^2) $

This is the whole observation model. Everything else is how its parameters
get chosen.

= The objective <objective>

Write $O$ for the set of (player, map) pairs that actually have a score.
Write $I$ for the number of players. The fit minimises

$ L = underbrace(sum_((i,j) in O) EE_(theta ~ Q_i) [-log cal(N)(y_(i j); m_j (theta), s_j (theta)^2)], "fit to the scores") \
    + underbrace(sum_i "KL"(Q_i || cal(N)(0,1)), "cost of moving a player") \
    + underbrace(lambda_p dot I dot (macron(mu)^2 + ("var" mu + "mean" sigma^2 - 1)^2), "hold the population") \
    + underbrace(lambda_c sum_(j,k) (u_(j k) - macron(u)_k)^2 + lambda_s sum_(j,k) (e_(j k) - macron(e)_k)^2 + lambda_l sum_j [(a_j - macron(a))^2 + (b_j - macron(b))^2], "pull maps together") $

Each term is worth taking on its own.

== Fit to the scores

The first term is the usual one: make the observed scores likely.

The detail that matters is $EE_(theta ~ Q_i)$. The likelihood is not read
at the player's best-guess skill. It is averaged over the fit's whole
belief about them. A player the fit has barely placed contributes a
smeared-out likelihood, so their scores pull the map curves gently. A
player it has placed firmly pulls hard.

== Cost of moving a player

The second term is the price of putting a player anywhere other than the
prior. For two normals it has a closed form:

$ "KL"(cal(N)(mu, sigma^2) || cal(N)(0,1)) = 1/2 (sigma^2 + mu^2 - 1) - log sigma $

It pulls each $mu_i$ towards 0 and each $sigma_i$ towards 1. A player with
few scores is not dragged far from the middle, because their scores cannot
pay the price.

== Hold the population

The third term is what earns the skill scale, as @scale explained.
$macron(mu)$ is the average of the fitted skills. $"var" mu + "mean"
sigma^2$ is the total variance of the fitted population, counting both the
spread between players and the width of each belief.

The term is zero when the average is 0 and the total variance is 1. It
grows as either drifts. Multiplying by $I$ keeps its size comparable to
the first term as the panel grows.

== Pull maps together

The last term is pooling, and it exists because most maps have very few
scores. A map with eleven players cannot support six free shape
parameters on its own.

$macron(u)_k$ is the average of $u_(j k)$ over all maps, so the term
charges each map for differing from the average map. A map with a lot of
data can afford to differ. A map with almost none is held near the
average, which is the sensible guess for it.

The three weights are set very differently, and @constants gives the
numbers.

= How the fit is computed

The expectation over $Q_i$ has no closed form, because $m_j$ and $s_j$ are
not linear. It is computed by Gauss--Hermite quadrature, which replaces
the integral with a weighted sum at fixed points:

$ EE_(theta ~ cal(N)(mu, sigma^2))[f(theta)] approx sum_(q=1)^Q W_q dot f(mu + sqrt(2) sigma x_q) $

The nodes $x_q$ and weights $W_q$ come from `numpy.polynomial.hermite`,
scaled so that the weights sum to 1. Seven nodes are used.

Every parameter is then moved by Adam, a gradient method: the skills
$mu_i$ and $log sigma_i$, and the map parameters $a_j$, $u_(j k)$, $b_j$
and $e_(j k)$, all at once.

The gradients are worked out by hand rather than by an autodiff library,
because the project depends on numpy alone. Running the script with
`--check-gradient` compares them against central differences of the
objective. The worst relative error is $5.5 dot 10^(-8)$.

= The constants <constants>

Some of these were chosen and some were swept. The distinction matters, so
each row says which.

The numbers in the "evidence" column are the held-out log density: 15% of
the observed cells are hidden, the fit runs on the rest, and this is the
average log probability it assigns to a hidden score. Higher is better.

#v(0.5em)

#table(
  columns: (auto, auto, 1fr),
  stroke: none,
  align: (left, left, left),
  inset: (x: 0.5em, y: 0.45em),
  table.hline(),
  table.header([*Constant*], [*Value*], [*How it was chosen*]),
  table.hline(),

  [accuracy floor], [$5 dot 10^(-4)$],
  [Chosen. It is about half of one 100-hit on a 700-object map, which is
   below any real score short of a perfect one.],

  [knots $K$], [6],
  [Swept over 3, 4, 6 and 9. The held-out density was −0.454, −0.456,
   −0.456 and −0.457, so the choice does not matter over that range.],

  [knot reach], [$plus.minus 2$],
  [Chosen. The fitted population has a spread of 1, so the knots cover two
   standard deviations either side of the middle.],

  [knot width $w$], [0.8],
  [Follows from the other two, as the spacing between neighbouring knots.],

  [quadrature $Q$], [7],
  [Chosen.],

  [Adam steps], [600],
  [Swept against 2000, which changed the held-out density by 0.001.],

  [Adam rate], [0.05], [Chosen.],

  [$sigma$ range], [0.02 to 3],
  [Chosen. A belief width running to zero turns the expectation back into
   a point estimate and takes the gradient with it.],

  [$lambda_p$], [1.0],
  [Chosen. The fitted population comes out at an average of −0.012 and a
   spread of 0.999, so it is doing its job.],

  [$lambda_c$], [2.0],
  [Swept over 0.5, 2, 10 and 50, giving −0.538, −0.456, −0.454 and −0.455.
   Two is the smallest value inside the flat region. Fifty scores as well
   but collapses every map onto the same shape, which defeats the point of
   fitting a shape.],

  [$lambda_s$], [50.0],
  [Swept over 0.5, 2, 10, 50 and 1000, giving −1.163, −0.517, −0.461,
   −0.455 and −0.458. A weak value lets a map with few players shrink its
   spread onto its own scores and claim a density no held-out score can
   match.],

  [$lambda_l$], [0.05],
  [Chosen. Maps genuinely differ in average accuracy and in scatter, so
   these two are pooled only lightly.],
  table.hline(),
)

== A note on the two sweeps

The curve weight and the spread weight were one setting at first. That was
a mistake, and separating them is what fixed it.

A weak shared weight blew the held-out density up to −146, which came
entirely from the spread. A strong shared weight fixed that but flattened
every map's curve to the same shape. Only with two weights can the spread
be held tightly while the curve stays free.

= What the numbers look like

Two tables, both from the fit as it currently stands. They are here so the
symbols above have something concrete attached.

== One map across the skill range

This is Reol -- No title \[jieusieu's Lemur\], the most played map in the
panel, with 840 players on it.

#v(0.4em)

#block(breakable: false)[
#table(
  columns: 4,
  stroke: none,
  align: (right, right, right, left),
  inset: (x: 0.7em, y: 0.4em),
  table.hline(),
  table.header([$theta$], [$m_j (theta)$], [$s_j (theta)$], [one spread either side]),
  table.hline(),
  [−2], [71.27%], [0.214], [52.97% to 82.45%],
  [−1], [93.72%], [0.322], [86.82% to 97.01%],
  [0],  [98.39%], [0.524], [94.61% to 99.52%],
  [+1], [99.35%], [0.636], [97.18% to 99.85%],
  [+2], [99.62%], [0.769], [97.77% to 99.94%],
  [+3], [99.72%], [0.788], [98.26% to 99.95%],
  table.hline(),
)
]

The curve column is shown as accuracy rather than as $y$, since that is
what a person recognises.

Notice that the two spread columns point opposite ways. The standard
deviation grows from 0.214 to 0.788. The accuracy band it covers shrinks
from 29 percentage points to under 2. Both are true at once, because the
$y$ scale stretches as accuracy approaches 1, so the same $s_j$ buys less
accuracy up there.

== Spread across maps

Measured at each map's own players, $s_j$ runs from 0.103 to 0.877, with
the middle half between 0.226 and 0.360. So maps differ in scatter by
roughly a factor of eight.

Inside a single map, comparing one step of skill below its players against
one step above, the median map's $s_j$ grows by a factor of 1.8.

= Where the model is known to fall short

The observation model is a normal distribution over one number. Real
outcomes are not one number. A score has a pass or a fail, a full combo or
not, a miss count and an accuracy, and each behaves differently and runs
into its own ceiling.

Nothing above the observation model depends on that choice. The skill side
reads only the log-likelihood and its first two derivatives in $theta$. A
better outcome model can be dropped in without touching anything else.

Which cells appear in $O$ has no model at all. A player chooses what to
play, and that choice depends on how well they expect to do. The panel is
restricted to cells the sampler asked for by name, which removes the
top-100 cut, and leaves that choice unmodelled.
