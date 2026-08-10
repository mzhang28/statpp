# statpp

An empirical performance-rating model for osu!, built by jointly inferring
player ability and map difficulty from submitted best scores, and comparing
the result against official pp to locate where the two disagree.

The framing matters: "true difficulty" is not identifiable on its own, but
**discrepancy between empirical difficulty and official pp** is well-posed.
That discrepancy is the deliverable.

## Estimand

The target is **best demonstrated submitted performance**, not average
performance and not attempt probability. This is a deliberate choice, and
it settles several questions that would otherwise look like confounds.

Unsubmitted plays and non-PB plays are out of scope by construction. Grind
is part of what "demonstrated" means, so a player who ran a map 400 times
and set a high PB has demonstrated that PB, so attempt count is part of the
definition rather than a nuisance variable to regress out.

## Sampling

`sample.py` draws players from log-spaced slices of the performance
ranking rather than crawling by map overlap. Overlap-based crawling walks
toward heavily-played maps, which is close to the definition of farm. It
would build the reference backbone out of exactly the region the model is
later supposed to call distorted.

Strata are roughly log-spaced in **global rank**, from #1 to ~#438k:

- `/rankings` caps at page 200 (rank ~10k) and **does not error past it**.
  Page 201+ silently returns page 200's contents, so pages are clamped.
- Below rank 10k, coverage comes from country rankings, which give ~10k
  players *per country*. Country choice is by playerbase size: a country
  needs enough ranked players for deep pages to hold real accounts
  (`EE:200` is global rank ~2.7M at 2pp, i.e. dead accounts).

Spacing is logarithmic because adjacent strata must share map vocabulary
for a joint player/map scale to be identifiable. Top-50 players and rank-10k
players have nearly disjoint top-100s, and nothing bridges them directly.
A popular map's top-50 leaderboard is all elite players, so it adds no
vertical spread. The only bridge is the chain of intermediate strata, and
`sample.py report` measures whether that chain actually holds.

Expansion is round-robin across strata, so a run that exhausts its request
budget still leaves balanced coverage instead of a depth-first dive into
the top stratum.

```
uv run sample.py grow       # sample and fill in one process, until the budget ends
uv run sample.py sample     # fetch strata, expand players' top-100s
uv run sample.py fill       # probe (player, map) cells to densify the panel
uv run sample.py report     # coverage + inter-stratum item overlap
uv run sample.py maps ID... # top-50 leaderboard for specific maps
```

Analysis reads the same database through a read-only connection, so it can
run while a sampler is writing:

```
uv run diagnose.py          # residual correlation within and between strata
uv run find_good_data.py    # the part of the data worth fitting on
uv run fit_ability_and_difficulty.py   # solve for both together
uv run fit_skill_and_curves.py         # skill and map curves, against the score
uv run fit_skill_and_curves.py --compare-families
```

Requires `OSU_CLIENT_ID` / `OSU_CLIENT_SECRET` in `.env`.

## An outside opinion about which maps test players

o!TR publishes a weekly replica of its tournament database. It records
which beatmaps each tournament put in its mappool, every match and game,
and each score set, all keyed by osu! ids. `otr_dataset.py` downloads it,
restores it into a PostgreSQL container of its own, and writes the parts
this project can use to `otr.sqlite`.

```
uv run otr_dataset.py
```

A mappool is the opposite of farm: somebody picked those maps to tell
players apart. Of the maps in the current panel, 631 have been pooled in a
verified tournament and 545 never pooled. Their fitted steepness is the
same to within 0.006 at matched star rating, so being chosen for a
mappool does not show up in what this model measures.

The dataset also carries four million tournament scores with accuracy,
misses and mods, on maps that overlap the panel. Nothing fits on those
yet.

Dataset by the [osu! Tournament Rating project](https://otr.stagec.net/),
whose terms require that credit.

`explorer/` is a page for reading the last of those. It has a tab for the
maps and one for the players, each sortable and filterable. A third shows
where each score landed inside the distribution the model predicted for it.

```
cd explorer
uv run --project .. --group explorer reflex run                  # localhost
uv run --project .. --group explorer reflex run \
    --env prod --single-port --backend-port 3000                 # anywhere
```

The second form answers on every interface and puts the page and its
websocket on one port, so it works through whatever hostname reaches the
machine. That matters because the address a tunnel or a DHCP lease hands
out is not known when the app starts, and a page served on one origin
whose socket points at another connects to nothing. Nothing asks who is
calling, so it is reachable by anything that can route to the port.

It fits on first load, a couple of minutes, and keeps the result on disk
so later loads are immediate. "Fit again from the database" reads whatever
the sampler has collected since.

### Growing the graph

`sample` and `fill` are both re-runnable and additive. `sample` widens the
space, `fill` densifies it, and neither re-requests anything already
stored, so running either again picks up where the last run stopped.
`grow` alternates the two until the request budget ends.

They share a process rather than running as a pair, because the rate
limiter is per-process: a sampler and a filler side by side would each
pace themselves to 57 requests per minute and put the pair over the
60/minute cap.

A ranking page holds exactly 50 players, which caps how dense a single
stratum can get, so widening the space means sampling new pages. `grow`
proposes them by rotating between policies, skipping every page already
stored:

- Pages beside an existing stratum, which add players at nearly the same
  ability and so raise the number available to co-play one map.
- The geometric midpoint of the widest gap in a country's ladder, which is
  where the chain of shared map vocabulary is thinnest.
- The deepest page of a country not yet sampled, biggest playerbase first,
  since page 200 of a small country holds dead accounts.

Each cycle expands before it fills, because new strata bring new players
and those players widen the panel the filling stage then works on.

### Filling the panel

Top-100 sampling on its own leaves the score matrix too sparse to
correlate. Within one stratum, 50 players spread 5,000 scores over ~3,000
maps, so few map pairs share enough players to be compared. `fill` probes
single (player, map) cells to close that gap, and it reaches the plays
below a player's top-100 cutoff that sampling cannot see at all.

The panel is built per stratum, because map popularity is stratum-specific.
Each stratum contributes the maps its own members play most, probed against
its own players. The globally most-played maps are ones a rank-400k player
has usually never touched, so probing those against them mostly returns 404.

A share of the probes (`--fill-explore`, default 0.15) goes to random cells
outside those rectangles. Those are the bridges: nothing else connects one
stratum's maps to another stratum's players, and each rectangle on its own
is only internally comparable. A 404 is an observation here rather than a
wasted request, because it says the player has never submitted a play on
the map.

A probe asks for every score the player has submitted on the map, not only
their best. The item is `(beatmap, mod_key)`, so one map carries several
items, and the mod combinations are spread widely enough that no single one
dominates. Asking for all of them costs the same single request.

Probes are recorded in `Probe` whether or not a score comes back, so a miss
is never requested twice.

Cells are probed map by map, round-robin across strata, so a run that stops
early leaves complete columns spread evenly over the ability range. Any pair
of complete columns shares every player and so can be compared; a
player-by-player order would leave every column partial instead.

## What the schema records that isn't obvious

**The item is `(beatmap, mod_key)`, not the beatmap.** `Score.mod_key` is
the canonical difficulty-relevant mod combination (`NC`→`DT`, cosmetic mods
dropped). `Score.mod_settings` flags lazer scores that include mod settings
(custom rates, AR overrides). Those aren't comparable to the plain
combination, so they're marked rather than merged.

**Truncation bounds**, recorded because they're free at ingest time rather
than because anything currently consumes them:

- `Player.best_cutoff_pp` is a player's 100th-best play. If `best_count`
  is 100, every map absent from their list is worth less than this.
- `Beatmap.leaderboard_cutoff_pp` is the 50th-place score, bounding
  everyone absent from the board.

Both are cheap to keep populated and available if a later model wants them.

## Learning a corrected score scale

The fit that exists takes pp as the observation and solves pp = ability +
difficulty. Measurements on the collected data say that form is wrong, and
say so specifically:

- A single score scatters about 80pp around the fit, which is large next to
  the differences being measured.
- On an easy map, pp barely moves with ability. For Reol - No title
  [byfaR's Hard] a player's residual tracks their overall level at -0.99
  with slope -1, meaning everyone earns about the same pp there. That map
  says nothing about who is better, and the fit weights it like any other.
- The discrepancy against pp tracks map length at +0.33 and playcount at
  -0.51. Duration and how the sample was drawn are being read as farm.
- Fitting to pp and then reporting a departure from pp measures one
  quantity against itself.

### Skill as a percentile

A player's skill is one number on the real line, read as a percentile of
the whole playerbase:

```
theta_i in R          q_i = Phi(theta_i) in (0, 1)
```

That reading has to be earned. `Phi` maps the line to `(0, 1)` whatever the
numbers mean, so `q_i` is a percentile only where the population
distribution of skill is itself standard normal:

```
theta_i ~ N(0, 1)
```

That is the population prior `P_i` the fit is regularised towards, and it
has to be held up during training: the empirical spread of the fitted
skills is what must stay centred and scaled, since nothing stops a
maximiser from drifting the whole population and rescaling the map curves
to match. Without that constraint `Phi` is an arbitrary change of
coordinate and the percentile reading is not available.

What is stored per player is a belief rather than a point, a Gaussian
`Q_i(theta) = N(theta; mu_i, sigma_i^2)`.

### Each map is a distribution over the outcome

A map is not a number and not a curve. It is a conditional distribution
over the score itself:

```
p_j : R -> Dist(X_j)      D = p_j(theta)
```

with `X_j` the outcome space of map `j`. Everything above this line asks
`D` for five things and knows nothing else about it:

```
log p(x)          how likely this score was
F(x)              where the score fell inside it
F^-1(u)           the score at any percentile
a draw            a score the map would produce
d log p(x)/dtheta how the likelihood moves with skill
```

The outcome fitted so far is the accuracy, exactly as it was set. It is
not put on another scale first. A scale followed by a normal distribution
is two assumptions where one will do, and it hides the question that
matters, which is what shape the outcomes really have. Accuracy is bounded
above and 1.3% of the panel sits exactly on the bound, so every candidate
is a density on (0, 1) mixed with a point mass at 1.

Each family describes its distribution by a few **channels**, and each
channel is a function of skill. The families that exist are:

- **beta + a mass at 1**, with channels for the logit of the mean, the log
  of the concentration, and the logit of the chance of a 100%.
- **logit-normal + a mass at 1**, with the same three, its middle and
  spread taken on the logit scale rather than tied to each other.
- **a two-component beta mixture + a mass at 1**, which adds how far a bad
  run lands below a normal one and how often a run goes that way.

Only the channels that must rise with skill are held monotone: the
location, and the chance of a 100%. Everything else is free to rise and
fall across the range, which is what a map does when it separates players
in its middle band while everyone below flounders alike and everyone above
clears alike. Every channel is a spline over skill with its shape partly
pooled across maps, since no single map has the data to fit its own.

The map's difficulty is that whole distribution rather than a separate
offset.

### Choosing the family

Not by looking at a fit. By held-out likelihood and calibration, both on
the same measure, which is what makes the numbers comparable at all: every
row below puts its mass on the accuracy itself.

```
uv run fit_skill_and_curves.py --compare-families
```

On 175,723 training cells and 31,288 held out, per held-out cell:

```
family        model                           log density   centre    gap
beta          one per map, no skill                 2.292
beta          player pp + map star rating           2.174
beta          skill belief + map curves             2.457    0.521   0.062

logit-normal  one per map, no skill                 2.384
logit-normal  player pp + map star rating           2.324
logit-normal  skill belief + map curves             2.584    0.501   0.014

beta-mixture  one per map, no skill                 2.373
beta-mixture  player pp + map star rating           2.358
beta-mixture  skill belief + map curves             2.575    0.503   0.015
```

`centre` is the average place a score took inside its own predicted
distribution and should be 0.5. `gap` is the largest departure of those
places from flat.

The logit-normal is the default. The mixture is within 0.009 of it, which
is nothing, so the argument between them is calibration and cost: 0.014
against 0.015, and 15 seconds a fit against 85.

Getting to that table meant fixing the pooling, and the fix was worth more
than the choice of family. Swept on one panel and one holdout, the strength
holding each channel's per-map level runs flat from about 2 to 40 and falls
away sharply below 1. It had been set at 0.05. Every family gained: the
mixture went from 2.245 to 2.601 and the logit-normal from 2.570 to 2.604.
Each family now declares its own strength per channel in `outcomes.py`,
because six channels carry 42 parameters per map against three channels'
21 and need holding twice as hard, so the table above is a fair fight
without any flags.

The beta trails on likelihood and is well behind on calibration, at 0.062
against 0.014, and no pooling moved that.

Because every number here is a density on accuracy, none of them compares
against anything measured on another scale.

### Performance against expectation

Where a score fell is the CDF of the distribution it was predicted from:

```
u = F_{j,theta}(x)
```

If the model is right these are uniform on (0, 1) whatever the map and
whoever the player, so their histogram is a calibration check that needs
no held-out data. A score of exactly 100% has no single place inside the
distribution, only the stretch the point mass covers, and it is given a
point in that stretch at random. Without that the uniformity is lost and
the histogram cannot be read.

### The objective

Over the observed cells `O` in `I x J`:

```
L = - sum_{(i,j) in O} E_{theta ~ Q_i} [ log p_j(y_ij | theta) ]
    + sum_i KL( Q_i || P_i )
    + R_maps
```

Map parameters and player beliefs are learned together, and training means
optimising each `(mu_i, sigma_i)` against this objective directly.

The inner loop is compiled with numba, which is what makes iterating on
families affordable. A whole run of the default fit, two fits of 600 steps
each over 207,000 cells and every table below, takes under a minute. Each
family supplies a compiled kernel for one score beside its numpy version,
and `--check-gradient` holds the two to each other as well as checking the
whole gradient against central differences.

### What one score does to a player

There is also a cheap way to fold in a single new score without refitting,
which is what an incremental update needs. Write the score's log-likelihood
as a function of skill, and take its slope and curvature at the player's
current mean:

```
l(theta) = log p_j(y_ij | theta)

g =  l'(mu_i)
h = -l''(mu_i)
```

A Laplace step around `mu_i` then gives

```
1 / sigma_new^2 = 1 / sigma_i^2 + h

mu_new = mu_i + g / ( 1 / sigma_i^2 + h )
```

This holds only near `mu_i`, and only while the precision it produces stays
positive. `h` is a curvature, and a mixture over fail and pass, or an
outcome piling up against a ceiling, is not log-concave everywhere, so `h`
can come out negative and drive `1/sigma_i^2 + h` to zero or below. Such a
step has to be rejected or damped. Where a score matters enough to be worth
getting right, refit against the objective above rather than stepping.

`g` says which way the result disagrees with where the player currently
sits, and how strongly. `h` says how much this map and this result reveal
about skill at all, so a large `h` cuts the uncertainty. A likelihood that
barely moves with skill gives both near zero, and the score changes nothing.

### How much a map tells you

What a map is worth is the Fisher information one score there carries
about skill:

```
I_j(theta) = E_{x ~ p_j(theta)} [ ( d log p_j(x | theta) / dtheta )^2 ]
```

This replaces reading the slope of an expected outcome, which could only
be read against the scale the outcome was written in. Information carries
no such dependence, and it is in units of precision: a map at 1.0 is worth
as much as the whole population prior.

Across the panel this runs from 0.05 to 54, with the middle four fifths
between 0.73 and 8.13.

Reol - No title [byfaR's Hard] is the map to look at first, since pp on it
tracks a player's own level at -0.99 with slope -1, meaning everyone earns
about the same pp there. On the raw outcome it comes out at 0.16 unmodded
across 582 players, against a panel top of 54, so the flatness is a fact
about the map and not only about pp.

Across held-out scores, how much a score actually moves a belief tracks the
map's information at +0.52 rank correlation, and the median belief narrows
by 0.2% on the least informative third of maps against 0.7% on the most.
That is the claim that a map nobody is separated on stops counting.

### Which cells are observed still needs its own model

What decides whether a score reaches the data is the top-100 cut and the
player's choice of what to play, and neither is censoring of the outcome
`p_j` describes.

A top-100 list is truncated on pp, so a play is visible only when its pp
beat the player's hundredth best, and `Player.best_cutoff_pp` records where
that cut fell. It bounds the pp of the plays not seen, which is a statement
about a selection score rather than about the outcome. Selection here
depends on the result, so ignoring it biases the fit.

A probe that came back empty says the player has never submitted a play on
the map. That is missing data, not a poor performance.

The panel carries both kinds of cell, so both mechanisms are live in it.
Fitting the probed cells alone removes the pp cut at the price of three
quarters of the data. It leaves the player's own choice of what to play
unmodelled either way. Which cell came from where is on the row, in
`Probe`, so a selection model can read it instead of the panel being cut
to suit one. The playcount effect measured above is that selection showing
up as farm.

*Validate:* held-out cells are predicted at 2.600 per cell against 2.332
from the official numbers alone. Maps everyone clears come out
uninformative, Reol [byfaR's Hard] at 0.16 against a panel top of 54.

The fitted population comes out at mean +0.201 and spread 0.949, with the
largest gap from a standard normal at 0.111 of the population against
0.048 on the probed cells alone, so the percentile reading has loosened.
The pp discrepancy still tracks playcount, at -0.71 against the pp fit's
-0.45, and the probed cells gave -0.58. Both of those are the top-100 cut
arriving with the data it came with, and both are what the selection model
above is for.


## Roadmap

Ordered by what unblocks what. Each step has a validation, because most of
these fail silently: a biased fit still produces plausible-looking numbers.

Steps 3 to 5 below describe the pp-based fit that exists now. The section
above replaces its observation model and supersedes it where they differ.

**1. Descriptive diagnostics, before fitting anything.**
Within a stratum, take items played by many of its players, subtract each
player's mean, and compute the residual correlation matrix across items.
Its eigenspectrum bounds how many latent dimensions the data can support.
Comparing that structure *across* strata tests whether difficulty ordering
is rank-invariant.
*Validate:* if item A beats item B for top players but reverses at rank
10k, a scalar model is dead on arrival, and that costs a few hundred
requests to find out rather than a month of modelling.

**2. Find the part of the graph worth fitting on.**
Players and maps are the two sides of a bipartite graph and an observed
score is an edge. How well ability and difficulty can be separated depends
on how concentrated that graph is, so measure it: connected components, how
deep the k-core goes, and the density inside it. Then fit on the part that
holds up rather than on everything collected.

This step measures rather than infers, so there is nothing here to validate
against. The number that decides whether step 3 is worth starting is
observations per parameter: one parameter per player plus one per map, and
a core carrying only a few observations each cannot pin them down.

**3. Scalar joint model with the gauge fixed explicitly.**
Infer ability and difficulty together. When evidence says a map is easier
than currently valued, lower it, recompute affected players, and iterate
toward equilibrium. Keep it one-sided at first: detect overvaluation
confidently, without trying to rescue underrated maps.

Ability and difficulty are unidentifiable up to a shift (and a scale), so
pin an anchor set or renormalize each iteration. A deflate-only update rule
ratchets: iterate it and the whole scale drifts down while relative order,
which is what the output actually depends on, stays put.
*Validate:* known farm maps deflate while unrelated established maps stay
stable.

**4. Uncertainty on every inferred quantity.**
New maps, new players, and weak latent factors stay uncertain. Uncertain
entities must not strongly validate other uncertain entities.

**5. Discrepancy against official pp.**
This is the actual deliverable, and it's what makes "farm" non-circular.
pp is the observed quantity, empirical difficulty is inferred
independently, and farm is the gap between them.
*Validate:* known farm maps show large positive discrepancy while
established maps stay near zero.

**6. Latent factors, cautiously.**
Learn correlations in residual performance rather than declaring aim/speed
by hand. Mods, AR, and scoring system are *observed* covariates so factors
don't waste themselves rediscovering "has +HD". Shrink unsupported factors
to zero.
*Validate:* leave-map-out, leave-player-out, temporal holdout,
cross-mapper. A factor that resolves to "mapper X players" is a failure.

**7. Factor inspectability.**
Periodically PCA/rotate the learned space, then Procrustes-align each
retrain against the previous one so factors don't arbitrarily swap meaning
between runs. Inspect the strongest player and map loadings by hand.
*Validate:* recognizable skill clusters and specialists recur across
retraining and bootstrap runs.

**8. Staleness through variance only.**
Keep the ability mean, inflate its variance with elapsed time
(σ²(t+Δt) = σ²(t) + qΔt). New evidence re-establishes confidence.
*Validate:* inactive players become uncertain rather than automatically
worse.

**9. Uncertainty-driven fetching, last.**
Deliberately after a validated likelihood. Active learning before then
creates a co-adaptation loop: the sampler stops fetching a region, the
model reads the absence as certainty, and the region freezes. Prefer
observations that *connect* uncertain regions to established ones over
observations inside the uncertain cluster.
*Validate:* measure expected against actual posterior-variance reduction
per fetched score.

The success criterion for the whole system: a newly discovered farm pattern
should start uncertain. Once enough well-connected players demonstrate
systematic overperformance, it should be confidently identified and
devalued. Ratings that depended on it should fall with it, and nothing in
unrelated parts of the graph should move.
