# statpp

An empirical performance-rating model for osu!, built by jointly inferring
player ability and map difficulty from submitted best scores, and comparing
the result against official pp to locate where the two disagree.

The framing matters: "true difficulty" is not identifiable on its own, but
**discrepancy between empirical difficulty and official pp** is well-posed.
That discrepancy is the deliverable.

## Sampling

`sample.py` draws players from log-spaced slices of the performance
ranking rather than crawling by map overlap. Overlap-based crawling walks
toward heavily-played maps, which is close to the definition of farm — it
would build the reference backbone out of exactly the region the model is
later supposed to call distorted.

Strata are roughly log-spaced in **global rank**, from #1 to ~#438k.
Two constraints shape the ladder:

- `/rankings` caps at page 200 (rank ~10k) and **does not error past it** —
  page 201+ silently returns page 200's contents. Pages are clamped.
- Below rank 10k, coverage comes from country rankings, which give ~10k
  players *per country*. Country choice is by playerbase size: a country
  needs enough ranked players for deep pages to hold real accounts
  (`EE:200` is global rank ~2.7M at 2pp, i.e. dead accounts).

Spacing is logarithmic because adjacent strata must share map vocabulary
for a joint player/map scale to be identifiable. Top-50 players and rank-10k
players have nearly disjoint top-100s, and nothing bridges them directly —
a popular map's top-50 leaderboard is all elite players, so it adds no
vertical spread. The only bridge is the chain of intermediate strata, and
`sample.py report` measures whether that chain actually holds.

Expansion is round-robin across strata, so a run that exhausts its request
budget still leaves balanced coverage instead of a depth-first dive into
the top stratum.

```
uv run sample.py sample     # fetch strata, expand players' top-100s
uv run sample.py report     # coverage + inter-stratum item overlap
uv run sample.py maps ID... # top-50 leaderboard for specific maps
```

Requires `OSU_CLIENT_ID` / `OSU_CLIENT_SECRET` in `.env`.

## What the schema records that isn't obvious

**Truncation bounds.** Both data sources are censored, not merely
incomplete, and the censoring threshold is observable — which makes the
missing data usable rather than discardable:

- `Player.best_cutoff_pp` — a player's 100th-best play. If `best_count`
  is 100, every map absent from their list is known to be worth *less than*
  this, which is a real constraint on the likelihood rather than a gap.
- `Beatmap.leaderboard_cutoff_pp` — the 50th-place score, bounding everyone
  absent from the board.

Selection here is on the outcome variable: a player's top-100 is exactly
the set of their plays with the highest positive residuals. Fitting to it
naively biases every observed residual upward.

**The item is `(beatmap, mod_key)`, not the beatmap.** `Score.mod_key` is
the canonical difficulty-relevant mod combination (`NC`→`DT`, cosmetic mods
dropped). `Score.mod_settings` flags lazer scores carrying mod settings
(custom rates, AR overrides) — those aren't comparable to the plain
combination and are marked rather than merged.

## Roadmap

Ordered by what unblocks what. Each step has a validation, because most of
these fail silently — a biased fit still produces plausible-looking
numbers.

**1. Descriptive diagnostics, before fitting anything.**
Within a stratum, take items played by many of its players, subtract each
player's mean, and compute the residual correlation matrix across items.
Its eigenspectrum bounds how many latent dimensions the data can support.
Comparing that structure *across* strata tests whether difficulty ordering
is rank-invariant.
*Validate:* if item A beats item B for top players but reverses at rank
10k, a scalar model is dead on arrival — and that costs a few hundred
requests to find out rather than a month of modelling.

**2. Censored likelihood.**
Use `best_cutoff_pp` and `leaderboard_cutoff_pp` as bounds instead of
dropping unobserved pairs. Selection is on the outcome variable, so naive
fitting biases every observed residual upward, and differentially — worse
for players with a wide top-100 spread.
*Validate:* fit on an artificially truncated subset of a well-covered
player and check the estimate recovers the untruncated one.

**3. Attempt counts as a covariate.**
Pull `/users/{id}/beatmapsets/most_played` and put attempt count in the
model. See the confound below; this has to land before farm detection
means anything.
*Validate:* the ability estimate for a high-playcount player should drop
relative to a low-playcount peer with the same top-100.

**4. Scalar joint model with the gauge fixed explicitly.**
Ability and difficulty are unidentifiable up to a shift (and a scale), so
pin an anchor set or renormalize each iteration. A deflate-only update
rule ratchets: iterate it and the whole scale drifts down while relative
order — the only thing that matters — stays put.
*Validate:* reproduce known leaderboard orderings; bootstrap and confirm
established estimates barely move.

**5. Uncertainty on every inferred quantity.**
Uncertain entities must not strongly validate other uncertain entities.
*Validate:* a synthetic isolated cluster stays high-σ no matter how dense
it gets internally.

**6. Discrepancy against official pp.**
This is the actual deliverable, and it's what makes "farm" non-circular —
pp is the observed quantity, empirical difficulty is inferred
independently, farm is the gap.
*Validate:* known farm maps show large positive discrepancy while
established maps stay near zero.

**7. Latent factors, cautiously.**
Mods, AR, and scoring system are *observed* covariates so factors don't
waste themselves rediscovering "has +HD". Shrink unsupported factors to
zero.
*Validate:* leave-map-out, leave-player-out, temporal holdout,
cross-mapper. A factor that resolves to "mapper X players" is a failure.

**8. Factor inspectability.**
Procrustes-align each retrain against the previous one so factors don't
arbitrarily swap meaning between runs.
*Validate:* recognizable skill clusters recur across bootstrap runs.

**9. Staleness through variance only.**
Keep the ability mean, inflate its variance with elapsed time
(σ²(t+Δt) = σ²(t) + qΔt).
*Validate:* inactive players become uncertain rather than automatically
worse.

**10. Uncertainty-driven fetching — last.**
Deliberately after a validated likelihood. Active learning before then
creates a co-adaptation loop: the sampler stops fetching a region, the
model reads the absence as certainty, and the region freezes. Prefer
observations that *connect* uncertain regions to established ones over
observations inside the uncertain cluster.
*Validate:* measure expected vs. actual posterior-variance reduction per
fetched score.

The success criterion for the whole thing: a newly discovered farm pattern
starts uncertain, becomes confidently identified once enough well-connected
players demonstrate systematic overperformance, gets devalued, drags down
the ratings that depended on it, and propagates none of that into
unrelated parts of the graph.

## Known confound, not yet handled

Best-of-*n* is a max-statistic: the max over *n* attempts grows about
σ√(2 ln n), so a player who ran a map 400 times has a higher expected best
than an identically skilled player who ran it 4 times. Unmodelled, map
difficulty absorbs popularity and player ability absorbs grind — which is
fatal for farm detection, since "farm map" and "map people retry" become
the same signal. `/users/{id}/beatmapsets/most_played` exposes per-player
per-map attempt counts, so this is observable and should enter the model as
a covariate rather than being retrofitted later.
