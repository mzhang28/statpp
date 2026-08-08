#!/usr/bin/env python3
"""
Find the part of the collected data that is worth fitting on.

Most of it is thin: an item held by three players, a player holding two
items. Fitting ability and difficulty together needs observations that
link players to each other through shared items, so what matters is how
connected the player/item graph is, not how many rows it has.

Three things get measured.

Connected components: two players in different components share no chain
of items at all, so nothing in the data says how their abilities compare.
A fit over several components produces one arbitrary offset per
component.

The k-core: drop items held by fewer than k players and players holding
fewer than k items, repeatedly, since each drop pushes the other side
under the threshold. The largest k that leaves anything is how deeply
connected the data is. Sweeping k shows how fast the graph thins out.

Density inside the core: what share of the player-by-item rectangle is
actually observed. That is the number the sampler moves.

Also prints each item's mean residual, which is how far players score
above or below their own average on it.

Reads through connect_readonly(), so it runs while the sampler writes.
"""

import argparse
from collections import defaultdict

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


def components(players):
    """
    Sizes of the connected components of the player/item graph.

    Players and items are both nodes; an observation is an edge. One
    component means every player is reachable from every other through
    shared items, which is what a single joint scale requires.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):
        ra, rb = find(a), find(b)

        if ra != rb:
            parent[ra] = rb

    for player, items in players.items():
        for key in items:
            union(("player", player), ("item", key))

    sizes = defaultdict(int)

    for node in list(parent):
        sizes[find(node)] += 1

    return sorted(sizes.values(), reverse=True)


def core_depth(players, limit=200):
    """How far the k-core survives, and its size at each k."""
    rows = []

    for k in range(2, limit):
        core = trim(players, k, k)

        if len(core) < 2:
            break

        items = {key for held in core.values() for key in held}
        cells = sum(len(v) for v in core.values())

        rows.append((k, len(core), len(items), cells, cells / (len(core) * len(items))))

    return rows


def item_means(residual):
    """Mean residual per item, and how many players it rests on."""
    totals = defaultdict(float)
    counts = defaultdict(int)

    for items in residual.values():
        for key, value in items.items():
            totals[key] += value
            counts[key] += 1

    return {key: (totals[key] / counts[key], counts[key]) for key in totals}


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

    parser.add_argument("--top", type=int, default=15, help="rows to print")

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
    items = {key for held in players.values() for key in held}

    print(
        f"all data: {len(players)} players, {len(items)} items, "
        f"{cells} observations, "
        f"density {cells / (len(players) * len(items)):.2%}"
    )

    sizes = components(players)

    print(
        f"components: {len(sizes)}, largest holds "
        f"{sizes[0]} of {len(players) + len(items)} nodes"
        + ("" if len(sizes) == 1 else f", rest {sizes[1:args.top + 1]}")
    )

    print()
    print("how far the k-core survives")
    print()
    print(f"{'k':>4}{'players':>9}{'items':>8}{'observations':>14}{'density':>9}")
    print("-" * 44)

    depth = core_depth(players)

    if not depth:
        print("  nothing survives k=2; the graph is still isolated points")
        return

    for k, n_players, n_items, n_cells, density in depth:
        if k % 5 == 0 or k in (2, 3, depth[-1][0]):
            print(
                f"{k:>4}{n_players:>9}{n_items:>8}{n_cells:>14}{density:>8.1%}"
            )

    print(f"\ndeepest k-core: k={depth[-1][0]}")

    core = trim(players, args.min_players, args.min_items)

    if len(core) < 2:
        print(
            f"\nNothing survives {args.min_players} players x "
            f"{args.min_items} items. Keep filling."
        )
        return

    holders, strata = coverage(core, stratum_of)
    core_cells = sum(len(v) for v in core.values())
    core_sizes = components(core)

    print()
    print(
        f"core at {args.min_players} players x {args.min_items} items: "
        f"{len(core)} players, {len(holders)} items, "
        f"{core_cells} observations, "
        f"density {core_cells / (len(core) * len(holders)):.1%}"
    )
    print(
        f"core components: {len(core_sizes)}"
        + ("" if len(core_sizes) == 1 else f", sizes {core_sizes[:args.top]}")
    )

    span = sorted(len(strata[k]) for k in holders)

    print(
        f"strata per item: median {span[len(span) // 2]}, "
        f"min {span[0]}, max {span[-1]}"
    )

    means = item_means(residuals(core))
    ranked = sorted(holders, key=lambda k: -len(holders[k]))

    print()
    print("items the most players hold")
    print()
    print(f"{'item':<18}{'players':>9}{'strata':>8}{'mean residual':>15}")
    print("-" * 50)

    for key in ranked[:args.top]:
        mean, _ = means[key]
        print(
            f"{key:<18}{len(holders[key]):>9}{len(strata[key]):>8}{mean:>15.1f}"
        )

    print()
    print("players holding the most core items")
    print()
    print(f"{'player':<12}{'stratum':<14}{'items held':>12}")
    print("-" * 38)

    for player in sorted(core, key=lambda p: -len(core[p]))[:args.top]:
        print(f"{player:<12}{stratum_of[player]:<14}{len(core[player]):>12}")


if __name__ == "__main__":
    main()
