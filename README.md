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

## Known confound, not yet handled

Best-of-*n* is a max-statistic: the max over *n* attempts grows about
σ√(2 ln n), so a player who ran a map 400 times has a higher expected best
than an identically skilled player who ran it 4 times. Unmodelled, map
difficulty absorbs popularity and player ability absorbs grind — which is
fatal for farm detection, since "farm map" and "map people retry" become
the same signal. `/users/{id}/beatmapsets/most_played` exposes per-player
per-map attempt counts, so this is observable and should enter the model as
a covariate rather than being retrofitted later.
