#!/usr/bin/env python3
"""
Fit one ability number per player and one difficulty number per map.

Each observed score is read as a player's ability plus a map's difficulty,
and the two are solved for together by alternating: hold difficulty fixed
and average out each player's ability, then hold ability fixed and average
out each map's difficulty. Repeat until the numbers stop moving.

Only their sum is pinned by the data, so adding a constant to every ability
and taking the same constant off every difficulty fits equally well. The
run therefore recentres difficulty on an anchor set after every sweep. The
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


def fit(rows, cols, values, n_players, n_items, anchors, sweeps, one_sided):
    """
    Alternating fit of ability and difficulty.

    Difficulty starts at each map's mean observed pp, which is what the
    data says before any player's level is taken into account.
    """
    difficulty, _ = group_means(cols, values, n_items)
    ability = np.zeros(n_players)

    history = []

    for sweep in range(sweeps):
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

    parser.add_argument("--sweeps", type=int, default=200)
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
        args.sweeps, not args.two_sided,
    )

    residual = values - ability[rows] - difficulty[cols]

    print(
        f"stopped after {len(history)} sweeps, "
        f"last move {history[-1]:.2e}, "
        f"residual sd {residual.std():.1f}pp"
    )

    change = difficulty - start
    order = np.argsort(change)

    print()
    print("maps the fit values furthest below their own mean")
    print()
    print(
        f"{'item':<18}{'players':>8}{'strata':>7}"
        f"{'mean pp':>10}{'fitted':>10}{'change':>9}"
    )
    print("-" * 62)

    for j in order[:args.top]:
        key = items[j]
        print(
            f"{key:<18}{len(holders[key]):>8}{len(strata[key]):>7}"
            f"{start[j]:>10.1f}{difficulty[j]:>10.1f}{change[j]:>9.1f}"
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

    print()
    print("ability by stratum, to check the fit orders the range sensibly")
    print()

    by_stratum = defaultdict(list)

    for i, player in enumerate(roster):
        by_stratum[stratum_of[player]].append(ability[i])

    print(f"{'stratum':<14}{'players':>9}{'median ability':>16}")
    print("-" * 39)

    for label in sorted(by_stratum, key=lambda s: -np.median(by_stratum[s])):
        got = by_stratum[label]
        print(f"{label:<14}{len(got):>9}{np.median(got):>16.1f}")


if __name__ == "__main__":
    main()
