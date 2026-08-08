#!/usr/bin/env python3
"""
Fit one ability number per player and one difficulty number per map.

Each observed score is read as a player's ability plus a map's difficulty,
and the two are solved for together by alternating: hold difficulty fixed
and average out each player's ability, then hold ability fixed and average
out each map's difficulty. Repeat until the numbers stop moving.

Only their sum is pinned by the data, so adding a constant to every ability
and taking the same constant off every difficulty fits equally well. The
run therefore recentres difficulty on an anchor set each time round. The
anchors are the maps whose players span the most strata, since those are
the ones placed against the widest part of the range.

Updates to difficulty are one-sided: a map's number may fall but not rise.
The point is to find maps worth less than they are credited with, and a
one-sided rule cannot talk itself into inflating a map on thin evidence.
The whole scale drifts down as this iterates, which is why only the order
and the gaps mean anything, not the level.

Run with --two-sided to let difficulty rise as well, which is the plain
fit and a useful thing to compare against.

Reads through connect_readonly(), so it runs while the sampler writes.
"""

import argparse
from collections import defaultdict

import numpy as np

from diagnose import load_scores, trim
from find_good_data import coverage, flatten
from sample import connect_readonly


def observations(players):
    """Flat arrays of player index, item index, and observed pp."""
    roster = sorted(players)
    items = sorted({key for held in players.values() for key in held})

    player_at = {p: i for i, p in enumerate(roster)}
    item_at = {k: j for j, k in enumerate(items)}

    rows, cols, values = [], [], []

    for player, held in players.items():
        for key, pp in held.items():
            rows.append(player_at[player])
            cols.append(item_at[key])
            values.append(pp)

    return (
        roster,
        items,
        np.array(rows),
        np.array(cols),
        np.array(values, dtype=float),
    )


def group_means(index, values, size):
    """Mean of values within each group, and the count per group."""
    totals = np.bincount(index, weights=values, minlength=size)
    counts = np.bincount(index, minlength=size)

    return np.divide(totals, np.maximum(counts, 1)), counts


def weighted_means(index, values, weights, size):
    """
    Mean weighted by how well each observation is pinned down, plus the
    variance of that mean.

    A score only says as much about a player as the map's difficulty is
    known: if the map's own number is a guess, the score cannot settle the
    player's. Weighting by the inverse of that uncertainty is what stops
    two poorly determined things from confirming each other.
    """
    precision = np.bincount(index, weights=weights, minlength=size)
    totals = np.bincount(index, weights=weights * values, minlength=size)

    safe = np.maximum(precision, 1e-12)

    return totals / safe, 1.0 / safe


def fit_with_uncertainty(
    rows, cols, values, n_players, n_items, anchors, ability, difficulty,
    iterations,
):
    """
    Re-fit weighting every observation by the other side's uncertainty,
    carrying a variance for each player and each map alongside the means.

    Difficulty is measured against the anchor maps, so every uncertainty
    here is uncertainty relative to that anchor set rather than an absolute.
    """
    spread = float((values - ability[rows] - difficulty[cols]).var())

    _, player_counts = group_means(rows, values, n_players)
    _, item_counts = group_means(cols, values, n_items)

    ability_var = spread / np.maximum(player_counts, 1)
    item_var = spread / np.maximum(item_counts, 1)

    for _ in range(iterations):
        weights = 1.0 / (spread + item_var[cols])
        ability, ability_var = weighted_means(
            rows, values - difficulty[cols], weights, n_players
        )

        weights = 1.0 / (spread + ability_var[rows])
        difficulty, item_var = weighted_means(
            cols, values - ability[rows], weights, n_items
        )

        centre = difficulty[anchors].mean()
        difficulty = difficulty - centre

        spread = float((values - ability[rows] - difficulty[cols]).var())

    return ability, np.sqrt(ability_var), difficulty, np.sqrt(item_var)


def fit(rows, cols, values, n_players, n_items, anchors, iterations, one_sided):
    """
    Alternating fit of ability and difficulty.

    Difficulty starts at each map's mean observed pp, which is what the
    data says before any player's level is taken into account.
    """
    difficulty, _ = group_means(cols, values, n_items)
    ability = np.zeros(n_players)

    history = []

    for _ in range(iterations):
        previous = difficulty.copy()

        ability, _ = group_means(rows, values - difficulty[cols], n_players)

        proposed, _ = group_means(cols, values - ability[rows], n_items)

        if one_sided:
            # A map's number may fall but not rise.
            difficulty = np.minimum(difficulty, proposed)
        else:
            difficulty = proposed

        # Only the sum is identified, so pin the anchors and let the rest
        # sit relative to them.
        difficulty = difficulty - difficulty[anchors].mean()

        shift = float(np.abs(difficulty - previous).max())
        history.append(shift)

        if shift < 1e-6:
            break

    return ability, difficulty, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--db", default=None)

    parser.add_argument("--min-players", type=int, default=20)
    parser.add_argument("--min-items", type=int, default=30)

    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--uncertainty-iterations",
        type=int,
        default=30,
        help="weighted iterations carrying a variance per player and per map",
    )
    parser.add_argument("--anchors", type=int, default=50)
    parser.add_argument("--top", type=int, default=15)

    parser.add_argument(
        "--two-sided",
        action="store_true",
        help="let difficulty rise as well as fall",
    )

    parser.add_argument("--collapse-mods", action="store_true")

    args = parser.parse_args()

    conn = connect_readonly(args.db)
    players, stratum_of = flatten(load_scores(conn, args.collapse_mods))

    if not players:
        print("No scores with a stratum and a pp value yet.")
        return

    core = trim(players, args.min_players, args.min_items)

    if len(core) < 2:
        print(
            f"Nothing survives {args.min_players} players x "
            f"{args.min_items} items. Keep filling."
        )
        return

    holders, strata = coverage(core, stratum_of)
    roster, items, rows, cols, values = observations(core)

    # Anchors are the maps placed against the widest part of the range.
    ranked = sorted(items, key=lambda k: (-len(strata[k]), -len(holders[k])))
    anchor_keys = ranked[:min(args.anchors, len(ranked))]
    item_at = {k: j for j, k in enumerate(items)}
    anchors = np.array([item_at[k] for k in anchor_keys])

    print(
        f"core: {len(roster)} players, {len(items)} items, "
        f"{len(values)} observations"
    )
    print(
        f"anchors: {len(anchors)} maps spanning "
        f"{min(len(strata[k]) for k in anchor_keys)}-"
        f"{max(len(strata[k]) for k in anchor_keys)} strata"
    )
    print(f"update rule: {'two-sided' if args.two_sided else 'deflate only'}")

    start, _ = group_means(cols, values, len(items))
    start = start - start[anchors].mean()

    ability, difficulty, history = fit(
        rows, cols, values,
        len(roster), len(items), anchors,
        args.iterations, not args.two_sided,
    )

    residual = values - ability[rows] - difficulty[cols]

    print(
        f"stopped after {len(history)} iterations, "
        f"last move {history[-1]:.2e}, "
        f"residual sd {residual.std():.1f}pp"
    )

    ability, ability_sd, difficulty, item_sd = fit_with_uncertainty(
        rows, cols, values,
        len(roster), len(items), anchors,
        ability, difficulty, args.uncertainty_iterations,
    )

    change = difficulty - start
    order = np.argsort(change)

    print()
    print("maps the fit values furthest below their own mean")
    print()
    print(
        f"{'item':<18}{'players':>8}{'strata':>7}"
        f"{'mean pp':>10}{'fitted':>18}{'change':>9}"
    )
    print("-" * 71)

    for j in order[:args.top]:
        key = items[j]
        fitted = f"{difficulty[j]:.1f} +/- {item_sd[j]:.1f}"
        print(
            f"{key:<18}{len(holders[key]):>8}{len(strata[key]):>7}"
            f"{start[j]:>10.1f}{fitted:>18}{change[j]:>9.1f}"
        )

    if args.two_sided:
        print()
        print("and furthest above")
        print()

        for j in order[::-1][:args.top]:
            key = items[j]
            print(
                f"{key:<18}{len(holders[key]):>8}{len(strata[key]):>7}"
                f"{start[j]:>10.1f}{difficulty[j]:>10.1f}{change[j]:>9.1f}"
            )

    counts = np.array([len(holders[k]) for k in items], dtype=float)

    print()
    print(
        f"map uncertainty: median +/-{np.median(item_sd):.1f}pp, "
        f"widest +/-{item_sd.max():.1f}pp"
    )
    print(
        f"player uncertainty: median +/-{np.median(ability_sd):.1f}pp, "
        f"widest +/-{ability_sd.max():.1f}pp"
    )

    thin = counts <= np.quantile(counts, 0.1)
    thick = counts >= np.quantile(counts, 0.9)
    print(
        f"least-played tenth of maps: +/-{np.median(item_sd[thin]):.1f}pp, "
        f"most-played tenth: +/-{np.median(item_sd[thick]):.1f}pp"
    )

    # Counting players alone would give this. The gap between it and the
    # fitted uncertainty is what the players' own uncertainty added, so it
    # shows whether weighting by the other side changed anything or just
    # reproduced the counts.
    from_counts = np.sqrt(residual.var() / np.maximum(counts, 1))
    inflation = item_sd / from_counts

    widest = int(np.argmax(inflation))

    print(
        f"uncertainty above what player counts alone give: "
        f"median {np.median(inflation):.2f}x, "
        f"most {inflation[widest]:.2f}x on {items[widest]}"
    )

    print()
    print("ability by stratum, to check the fit orders the range sensibly")
    print()

    by_stratum = defaultdict(list)

    for i, player in enumerate(roster):
        by_stratum[stratum_of[player]].append((ability[i], ability_sd[i]))

    print(f"{'stratum':<14}{'players':>9}{'median ability':>20}")
    print("-" * 43)

    for label in sorted(
        by_stratum, key=lambda s: -np.median([a for a, _ in by_stratum[s]])
    ):
        got = by_stratum[label]
        middle = np.median([a for a, _ in got])
        error = np.median([e for _, e in got])
        print(f"{label:<14}{len(got):>9}{f'{middle:.1f} +/- {error:.1f}':>20}")


if __name__ == "__main__":
    main()
