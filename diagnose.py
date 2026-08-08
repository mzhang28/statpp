#!/usr/bin/env python3
"""
Residual correlation across items, within and between strata.

Roadmap step 1: ask what the data can support before fitting anything.

Subtracting a player's mean from their scores leaves the part of their
performance that their overall level does not explain. Correlating those
residuals across items asks whether the same players over- and
under-perform together. The eigenspectrum of that correlation matrix
bounds how many latent dimensions the data can carry: one dominant
eigenvalue means a scalar ability model is all the data can support, and
a long flat tail means the panel is still too sparse to tell.

Comparing item difficulty between strata is the separate question of
whether difficulty ordering is rank-invariant. If item A beats item B for
the top 50 players but the order reverses at rank 10k, no scalar model
holds across the ability range, and that is worth knowing before fitting
rather than after.

Reads through connect_readonly(), so it runs while the sampler writes.
"""

import argparse
from collections import defaultdict

import numpy as np

from sample import connect_readonly


def load_scores(conn, collapse_mods):
    """
    Best pp per (stratum, player, item).

    One player can hold scores on the same beatmap under several mods, and
    the probe endpoint returns whichever of them is their best, so a
    beatmap arrives under more than one mod_key. The modelled item is
    (beatmap, mod_key), because DT is nearly as common as no mod at all
    and is not the same thing to play.
    """
    item = "s.beatmap" if collapse_mods else "s.beatmap || ':' || s.mod_key"

    rows = conn.execute(f"""
        select p.stratum, s.player, {item}, max(s.pp)
        from Score s join Player p on p.id = s.player
        where s.pp is not null and p.stratum is not null
        group by p.stratum, s.player, {item}
    """)

    by_stratum = defaultdict(lambda: defaultdict(dict))

    for label, player, key, pp in rows:
        by_stratum[label][player][key] = pp

    return by_stratum


def trim(players, min_players, min_items):
    """
    Drop items too few players hold, and players holding too few items.

    Dropping on one side can push the other side under its threshold, so
    this repeats until nothing further falls out.
    """
    players = {p: dict(items) for p, items in players.items()}

    while players:
        held = defaultdict(int)

        for items in players.values():
            for key in items:
                held[key] += 1

        keep = {key for key, n in held.items() if n >= min_players}

        trimmed = {}

        for p, items in players.items():
            kept = {k: v for k, v in items.items() if k in keep}

            if len(kept) >= min_items:
                trimmed[p] = kept

        if trimmed == players:
            break

        players = trimmed

    return players


def residuals(players):
    """Each player's scores with their own mean taken out."""
    out = {}

    for p, items in players.items():
        mean = sum(items.values()) / len(items)
        out[p] = {k: v - mean for k, v in items.items()}

    return out


def correlate(residual, min_pair):
    """
    Correlation between items, over the players that hold both.

    Every pair rests on whichever players share those two items, so the
    matrix is assembled from different player sets and is not guaranteed
    positive semi-definite. Negative eigenvalues measure exactly that, and
    so track how far the panel still is from complete. Pairs below
    min_pair are left at zero rather than estimated from a handful of
    players.
    """
    holders = defaultdict(set)

    for p, items in residual.items():
        for key in items:
            holders[key].add(p)

    keys = sorted(holders)
    n = len(keys)

    corr = np.eye(n)
    measured = 0

    for a in range(n):
        for b in range(a + 1, n):
            shared = holders[keys[a]] & holders[keys[b]]

            if len(shared) < min_pair:
                continue

            xa = np.fromiter((residual[p][keys[a]] for p in shared), float)
            xb = np.fromiter((residual[p][keys[b]] for p in shared), float)

            xa -= xa.mean()
            xb -= xb.mean()

            spread = np.sqrt((xa @ xa) * (xb @ xb))

            if spread == 0:
                continue

            corr[a, b] = corr[b, a] = (xa @ xb) / spread
            measured += 1

    return keys, corr, measured


def average_ranks(values):
    """Ranks with ties averaged, which is what Spearman needs."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1

        shared = (i + j) / 2.0 + 1.0

        for k in range(i, j + 1):
            ranks[order[k]] = shared

        i = j + 1

    return ranks


def spearman(xs, ys):
    a = np.array(average_ranks(xs))
    b = np.array(average_ranks(ys))

    a -= a.mean()
    b -= b.mean()

    spread = np.sqrt((a @ a) * (b @ b))

    return float((a @ b) / spread) if spread else float("nan")


def difficulty(residual):
    """
    Mean residual per item: how far players fall below or rise above their
    own level on it. Lower means harder than the rest of what they play.
    """
    totals = defaultdict(list)

    for items in residual.values():
        for key, value in items.items():
            totals[key].append(value)

    return {key: sum(v) / len(v) for key, v in totals.items()}


def describe(corr):
    """Eigenspectrum summary of one stratum's correlation matrix."""
    values = np.linalg.eigvalsh(corr)[::-1]

    positive = values[values > 0].sum()

    return {
        "top": values[:3],
        "share1": values[0] / positive if positive else float("nan"),
        "share3": values[:3].sum() / positive if positive else float("nan"),
        "negative": -values[values < 0].sum() / positive if positive else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--db", default=None)

    parser.add_argument(
        "--min-players",
        type=int,
        default=10,
        help="players a stratum must have on an item to keep it",
    )

    parser.add_argument(
        "--min-items",
        type=int,
        default=3,
        help="items a player must hold to stay in the matrix",
    )

    parser.add_argument(
        "--min-pair",
        type=int,
        default=5,
        help="players an item pair needs before its correlation is used",
    )

    parser.add_argument(
        "--collapse-mods",
        action="store_true",
        help="treat a beatmap as one item regardless of mods, which trades "
             "an item that means one thing for roughly twice the overlap",
    )

    args = parser.parse_args()

    conn = connect_readonly(args.db)
    by_stratum = load_scores(conn, args.collapse_mods)

    if not by_stratum:
        print("No scores with a stratum and a pp value yet.")
        return

    print(
        f"item = {'beatmap' if args.collapse_mods else 'beatmap x mod_key'}, "
        f"items need {args.min_players} players, "
        f"players need {args.min_items} items, "
        f"pairs need {args.min_pair} players"
    )

    print()
    print(
        f"{'stratum':<13}{'items':>7}{'players':>9}{'density':>9}{'pairs':>8}"
        f"{'eig1':>8}{'eig2':>8}{'eig3':>8}{'top1':>7}{'top3':>7}{'neg':>7}"
    )
    print("-" * 91)

    kept = {}

    for label in sorted(by_stratum):
        players = trim(by_stratum[label], args.min_players, args.min_items)

        if len(players) < 2:
            print(f"{label:<13}{'too sparse to correlate':>44}")
            continue

        residual = residuals(players)
        keys, corr, measured = correlate(residual, args.min_pair)

        if len(keys) < 2:
            print(f"{label:<13}{'too sparse to correlate':>44}")
            continue

        cells = sum(len(v) for v in players.values())
        density = cells / (len(players) * len(keys))
        possible = len(keys) * (len(keys) - 1) // 2

        stats = describe(corr)
        kept[label] = difficulty(residual)

        print(
            f"{label:<13}{len(keys):>7}{len(players):>9}{density:>8.0%}"
            f"{measured:>5}/{possible:<7}"
            f"{stats['top'][0]:>7.2f}{stats['top'][1]:>8.2f}"
            f"{stats['top'][2]:>8.2f}"
            f"{stats['share1']:>7.0%}{stats['share3']:>7.0%}"
            f"{stats['negative']:>7.0%}"
        )

    print()
    print("eig1..3: leading eigenvalues. top1/top3: share of positive")
    print("eigenvalue mass they hold, so top1 near 1 means one latent")
    print("dimension is all the data supports. neg: negative eigenvalue")
    print("mass, which measures panel incompleteness rather than structure.")

    labels = [lab for lab in sorted(kept) if len(kept[lab]) >= 2]

    if len(labels) < 2:
        return

    print()
    print("difficulty ordering between strata (rank correlation / shared items)")
    print("a scalar model needs these positive: the same items harder for everyone")
    print()

    print(" " * 13 + "".join(f"{lab:>14}" for lab in labels))

    for a in labels:
        row = f"{a:<13}"

        for b in labels:
            if a == b:
                row += f"{'-':>14}"
                continue

            shared = sorted(set(kept[a]) & set(kept[b]))

            if len(shared) < 3:
                row += f"{f'{len(shared)} shared':>14}"
                continue

            rho = spearman(
                [kept[a][k] for k in shared],
                [kept[b][k] for k in shared],
            )
            row += f"{f'{rho:+.2f}/{len(shared)}':>14}"

        print(row)


if __name__ == "__main__":
    main()
