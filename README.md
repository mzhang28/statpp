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
```

Requires `OSU_CLIENT_ID` / `OSU_CLIENT_SECRET` in `.env`.

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

A player's skill is one number on the real line, and what it means is fixed
by reading it as a percentile of the whole playerbase:

```
theta_i in R          q_i = Phi(theta_i) in (0, 1)
```

What is stored per player is a belief rather than a point, a Gaussian
`Q_i(theta) = N(theta; mu_i, sigma_i^2)`. Reading skill through `Phi` sets
the units, which come from the distribution of players.

### Each map is a curve and a spread

A map is described by where performance sits at each skill level and by how
much it varies there:

```
m_j : R -> R          expected performance at skill theta
s_j : R -> R_{>0}     conditional spread at skill theta
```

Both are monotone splines with parameters partly pooled across maps, since
no single map has the data to fit its own. The first workable outcome model
is Gaussian:

```
p_j(y | theta) = N( y ; m_j(theta), s_j(theta)^2 )
```

The map's difficulty is the curve itself rather than a separate offset.

### The objective

Over the observed cells `O` in `I x J`:

```
L = - sum_{(i,j) in O} log p_j(y_ij | theta_i) + R_maps + R_time
```

Carrying the player Gaussians properly makes it variational instead:

```
L = - sum_{(i,j) in O} E_{theta ~ Q_i} [ log p_j(y_ij | theta) ]
    + sum_i KL( Q_i || P_i )
    + R_maps
```

Map parameters and player beliefs are learned together.

### What one score does to a player

Write the score's log-likelihood as a function of skill, and take its slope
and curvature at the player's current mean:

```
l(theta) = log p_j(y_ij | theta)

g =  l'(mu_i)
h = -l''(mu_i)
```

The local Gaussian update is then

```
1 / sigma_new^2 = 1 / sigma_i^2 + h

mu_new = mu_i + g / ( 1 / sigma_i^2 + h )
```

`g` says which way the result disagrees with where the player currently
sits, and how strongly. `h` says how much this map and this result reveal
about skill at all, so a large `h` cuts the uncertainty. A likelihood that
barely moves with skill gives both near zero, and the score changes nothing.

### Why the flat maps stop counting

Take an easy map where everyone above the bottom tenth full-combos it and
accuracy among them says nothing. Then `m_j'(theta)` is near zero across
most of the range, the likelihood hardly moves as skill moves, and the score
updates almost nothing. Reol - No title [byfaR's Hard] is measured to behave
exactly like this, tracking a player's own level at -0.99 with slope -1.

A map whose expected accuracy turns over around the 80th percentile has a
large `|m_j'(theta)|` right there, so scores on it separate the 70th from
the 80th from the 90th.

### The Gaussian is a placeholder

Real outcomes are not one scalar. Fail against pass, full combo against
not, misses, and accuracy each behave differently and run into ceilings, so
`p_j` wants to be a bounded or mixture distribution over them.

None of that reaches the skill side, because the interface between the two
is only

```
p_j : R -> Dist(X_j)
```

with `X_j` the outcome space of map `j`. The skill system reads the
log-likelihood together with its gradient and curvature, and nothing else
about how they were produced.

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

Probes are the way in, because which cells get probed is our decision rather
than the player's or the pp cut's. Inside the probed panel the pp truncation
is gone, leaving the player's own choice of what to play as the one
mechanism to model. The playcount effect measured above is that selection
showing up as farm.

*Validate:* hold out observed cells and predict them, against predicting
them from pp directly. Check `m_j'` on maps everyone clears, which for the
Reol Hard should come out near zero. Then recompute the length and playcount
correlations: if they have not fallen towards zero, the correction did not
work.


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
