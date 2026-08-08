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

## Roadmap

Ordered by what unblocks what. Each step has a validation, because most of
these fail silently: a biased fit still produces plausible-looking numbers.

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
score is an edge. Fitting ability and difficulty together only works where
that graph is connected and dense: two players in separate components share
no chain of maps, so nothing in the data says how their abilities compare.
Measure the components, how deep the k-core goes, and the density inside
it, then fit on the part that holds up.
*Validate:* the core is a single component, so every player and map in it
reaches every other through shared plays.

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
