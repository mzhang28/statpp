#!/usr/bin/env python3
"""
Find the part of the collected data that is worth fitting on.

Most of it is thin: an item held by three players, a player holding two
items. Anything fitted on that is guesswork. This finds the part that
isn't, so later work has somewhere to start.

Two filters. The first is a k-core: drop items held by fewer than
--min-players and players holding fewer than --min-items, repeatedly,
since each drop pushes the other side under its threshold. The second
keeps items whose players come from more than one stratum, because an
item held inside a single stratum cannot be placed against the rest of
the ability range.

Then it resamples players with replacement and recomputes each item's
mean residual, to show how far the number moves. An item carried by a
few players swings; that is worth seeing before using its value for
anything. This measures the sample, not the model, and no model is
fitted here.

Reads through connect_readonly(), so it runs while the sampler writes.
"""

import argparse
from collections import defaultdict

import numpy as np

from diagnose import load_scores, residuals, trim
from sample import connect_readonly


def flatten(by_stratum):
    """One table of players, plus the stratum each belongs to."""
    players = {}
    stratum_of = {}

    for label, members in by_stratum.items():
        for player, items in members.items():
            players[player] = dict(items)
            stratum_of[player] = label

    return players, stratum_of


def coverage(players, stratum_of):
    """Who holds each item, and which strata they come from."""
    holders = defaultdict(set)

    for player, items in players.items():
        for key in items:
            holders[key].add(player)

    strata = {
        key: {stratum_of[p] for p in who}
        for key, who in holders.items()
    }

    return holders, strata


def item_means(residual, subset=None):
    """
    Mean residual per item: how far players fall below or rise above
    their own level on it. Lower means harder than the rest of what they
    play.
    """
    totals = defaultdict(float)
    counts = defaultdict(int)

    for player in (subset if subset is not None else residual):
        for key, value in residual[player].items():
            totals[key] += value
            counts[key] += 1

    return {key: totals[key] / counts[key] for key in totals}


def resample(residual, replicates, rng):
    """
    Spread of each item's mean under resampling players with replacement.

    Players are the unit resampled, because players are what the sampler
    drew. An item held up by three of them moves a long way when those
    three are redrawn.
    """
    roster = sorted(residual)
    spread = defaultdict(list)

    for _ in range(replicates):
        drawn = [roster[i] for i in rng.integers(0, len(roster), len(roster))]

        for key, value in item_means(residual, drawn).items():
            spread[key].append(value)

    return {
        key: (float(np.mean(v)), float(np.std(v)))
        for key, v in spread.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--db", default=None)

    parser.add_argument(
        "--min-players",
        type=int,
        default=20,
        help="players an item needs to stay in the core",
    )

    parser.add_argument(
        "--min-items",
        type=int,
        default=30,
        help="items a player needs to stay in the core",
    )

    parser.add_argument(
        "--min-strata",
        type=int,
        default=4,
        help="strata an item's players must span to be kept",
    )

    parser.add_argument(
        "--resample",
        type=int,
        default=200,
        help="resampling replicates",
    )

    parser.add_argument("--top", type=int, default=20, help="rows to print")

    parser.add_argument(
        "--collapse-mods",
        action="store_true",
        help="treat a beatmap as one item regardless of mods",
    )

    args = parser.parse_args()

    conn = connect_readonly(args.db)
    players, stratum_of = flatten(load_scores(conn, args.collapse_mods))

    if not players:
        print("No scores with a stratum and a pp value yet.")
        return

    cells = sum(len(v) for v in players.values())
    items = len({k for v in players.values() for k in v})

    print(f"all data: {len(players)} players, {items} items, {cells} observations")

    core = trim(players, args.min_players, args.min_items)

    if len(core) < 2:
        print(
            f"Nothing survives {args.min_players} players x "
            f"{args.min_items} items. Keep filling."
        )
        return

    holders, strata = coverage(core, stratum_of)
    core_cells = sum(len(v) for v in core.values())

    print(
        f"core:     {len(core)} players, {len(holders)} items, "
        f"{core_cells} observations, "
        f"density {core_cells / (len(core) * len(holders)):.0%}"
    )

    kept = [k for k in holders if len(strata[k]) >= args.min_strata]

    print(
        f"spanning {args.min_strata}+ strata: {len(kept)} of {len(holders)} items"
    )

    if not kept:
        print(
            "Nothing spans strata, so no item can be placed against the "
            "range. The exploration cells are what produce these."
        )
        return

    spread = resample(residuals(core), args.resample, np.random.default_rng(0))

    rows = sorted(kept, key=lambda k: (spread[k][1], -len(holders[k])))

    print()
    print(f"items whose mean moves least under resampling ({args.resample}x)")
    print()
    print(f"{'item':<18}{'players':>9}{'strata':>8}{'mean':>10}{'sd':>9}")
    print("-" * 54)

    for key in rows[:args.top]:
        mean, sd = spread[key]
        print(
            f"{key:<18}{len(holders[key]):>9}{len(strata[key]):>8}"
            f"{mean:>10.1f}{sd:>9.2f}"
        )

    print()
    print("and the ones that move most")
    print()

    for key in rows[-min(args.top, len(rows)):][::-1]:
        mean, sd = spread[key]
        print(
            f"{key:<18}{len(holders[key]):>9}{len(strata[key]):>8}"
            f"{mean:>10.1f}{sd:>9.2f}"
        )

    sds = np.array([spread[k][1] for k in kept])
    counts = np.array([len(holders[k]) for k in kept], dtype=float)

    print()
    print(f"sd across kept items: median {np.median(sds):.2f}, worst {sds.max():.2f}")

    # If more players did not mean a steadier number, the filters are not
    # selecting for what they are meant to select for.
    if len(counts) >= 3 and counts.std() > 0 and sds.std() > 0:
        r = float(np.corrcoef(counts, sds)[0, 1])
        note = "more players, less movement" if r < 0 else "more players did not help"
        print(f"player count against sd: {r:+.2f} ({note})")

    print()
    print("players holding the most of these items")
    print()
    print(f"{'player':<12}{'stratum':<14}{'items held':>12}")
    print("-" * 38)

    keep = set(kept)
    ranked = sorted(core, key=lambda p: -sum(1 for k in core[p] if k in keep))

    for player in ranked[:args.top]:
        held = sum(1 for k in core[player] if k in keep)
        print(f"{player:<12}{stratum_of[player]:<14}{held:>12}")


if __name__ == "__main__":
    main()
