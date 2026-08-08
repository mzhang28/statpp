#!/usr/bin/env python3
"""
Find maps that hand out more pp than their star rating says they should.

The fit gives each map a number for how much pp it pays a player beyond
that player's usual haul. That on its own is not a complaint about the
map: a genuinely hard map ought to pay more. It only becomes one when the
map pays more than other maps rated equally hard.

So each map is compared against its own peers. Maps are grouped by mod
combination, since a mod changes what the map actually demands, and
within each group the payout is regressed on the official star rating.
The gap between a map's fitted payout and what that curve predicts for
its rating is the number reported here.

Reading it: a large positive gap means the map pays well for how hard
osu! rates it. A gap near zero means the rating and the payout agree.

pp is the observed quantity and the payout is inferred from the whole
player graph without reference to it, so the two are not the same
measurement twice.

Reads through connect_readonly(), so it runs while the sampler writes.
"""

import argparse
from collections import defaultdict

import numpy as np

from diagnose import load_scores, trim
from find_good_data import coverage, flatten
from fit_ability_and_difficulty import (
    fit,
    fit_with_uncertainty,
    observations,
)
from sample import connect_readonly


def beatmap_facts(conn):
    """Star rating and difficulty name for every map we have."""
    return {
        int(row[0]): (row[1], row[2])
        for row in conn.execute(
            "select id, stars, version from Beatmap where stars is not null"
        )
    }


def expected_payout(stars, payout, window, min_neighbours):
    """
    What maps of about this star rating pay, taken from those maps alone.

    A curve fitted across a whole mod combination extrapolates at the ends
    of its star range, and the ends are exactly where the sparsest maps
    sit. Comparing each map only against others within `window` stars of
    it keeps every comparison local, and returns nothing at all for a map
    with too few neighbours rather than an invented number.
    """
    expected = np.full(len(stars), np.nan)

    for i, rating in enumerate(stars):
        near = np.abs(stars - rating) <= window
        near[i] = False

        if near.sum() >= min_neighbours:
            expected[i] = np.median(payout[near])

    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--db", default=None)

    parser.add_argument("--min-players", type=int, default=20)
    parser.add_argument("--min-items", type=int, default=30)

    parser.add_argument(
        "--min-group",
        type=int,
        default=25,
        help="maps a mod combination needs before it is used at all",
    )

    parser.add_argument(
        "--star-window",
        type=float,
        default=0.5,
        help="how far in stars to look for maps to compare against",
    )

    parser.add_argument(
        "--min-neighbours",
        type=int,
        default=8,
        help="maps of similar rating needed before a comparison is made",
    )

    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--uncertainty-iterations", type=int, default=30)
    parser.add_argument("--anchors", type=int, default=50)
    parser.add_argument("--top", type=int, default=20)

    args = parser.parse_args()

    conn = connect_readonly(args.db)
    facts = beatmap_facts(conn)

    players, stratum_of = flatten(load_scores(conn, False))
    core = trim(players, args.min_players, args.min_items)

    if len(core) < 2:
        print(
            f"Nothing survives {args.min_players} players x "
            f"{args.min_items} items. Keep filling."
        )
        return

    holders, strata = coverage(core, stratum_of)
    roster, items, rows, cols, values = observations(core)

    ranked = sorted(items, key=lambda k: (-len(strata[k]), -len(holders[k])))
    item_at = {k: j for j, k in enumerate(items)}
    anchors = np.array([item_at[k] for k in ranked[:args.anchors]])

    # Two-sided, because this wants an even-handed payout for every map
    # rather than the one-sided rule that only ever marks maps down.
    ability, payout, _ = fit(
        rows, cols, values,
        len(roster), len(items), anchors,
        args.iterations, False,
    )

    ability, _, payout, payout_sd = fit_with_uncertainty(
        rows, cols, values,
        len(roster), len(items), anchors,
        ability, payout, args.uncertainty_iterations,
    )

    print(f"core: {len(roster)} players, {len(items)} maps-with-mods")

    by_mods = defaultdict(list)

    for j, key in enumerate(items):
        beatmap_id, mods = key.split(":", 1)
        beatmap_id = int(beatmap_id)

        if beatmap_id in facts:
            by_mods[mods].append((j, beatmap_id))

    gaps = {}

    for mods, group in sorted(by_mods.items(), key=lambda kv: -len(kv[1])):
        if len(group) < args.min_group:
            continue

        index = np.array([j for j, _ in group])
        stars = np.array([facts[b][0] for _, b in group], dtype=float)

        predicted = expected_payout(
            stars, payout[index], args.star_window, args.min_neighbours
        )

        for spot, j in enumerate(index):
            if not np.isnan(predicted[spot]):
                gaps[j] = (payout[j] - predicted[spot], predicted[spot], mods)

        compared = int(np.isfinite(predicted).sum())

        print(
            f"  {mods:<12} {len(group):>4} maps, "
            f"{stars.min():.1f}-{stars.max():.1f} stars, "
            f"{compared} with enough neighbours"
        )

    if not gaps:
        print(
            f"No mod combination has {args.min_group} maps yet, so there is "
            "nothing to compare a map against. Keep filling."
        )
        return

    order = sorted(gaps, key=lambda j: -gaps[j][0])

    print()
    print("maps paying most above their star rating")
    print()
    print(
        f"{'beatmap':>9} {'mods':<9}{'stars':>6}{'players':>8}"
        f"{'pays':>8}{'rating says':>12}{'gap':>8}  difficulty"
    )
    print("-" * 84)

    for j in order[:args.top]:
        gap, predicted, mods = gaps[j]
        beatmap_id = int(items[j].split(":", 1)[0])
        stars, version = facts[beatmap_id]

        print(
            f"{beatmap_id:>9} {mods:<9}{stars:>6.2f}"
            f"{len(holders[items[j]]):>8}"
            f"{payout[j]:>8.0f}{predicted:>12.0f}{gap:>+8.0f}"
            f"  {version[:28]}"
        )

    print()
    print("and paying least, which is the other end of the same measure")
    print()

    for j in order[::-1][:5]:
        gap, predicted, mods = gaps[j]
        beatmap_id = int(items[j].split(":", 1)[0])
        stars, version = facts[beatmap_id]

        print(
            f"{beatmap_id:>9} {mods:<9}{stars:>6.2f}"
            f"{len(holders[items[j]]):>8}"
            f"{payout[j]:>8.0f}{predicted:>12.0f}{gap:>+8.0f}"
            f"  {version[:28]}"
        )

    spread = np.array([gaps[j][0] for j in gaps])

    print()
    print(
        f"gaps run {spread.min():+.0f} to {spread.max():+.0f}pp, "
        f"half within +/-{np.percentile(np.abs(spread), 50):.0f}pp, "
        f"typical uncertainty +/-{np.median(payout_sd):.0f}pp"
    )


if __name__ == "__main__":
    main()
