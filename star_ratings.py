#!/usr/bin/env python3
"""
Star ratings with the mods applied.

The rating stored on a beatmap is the unmodded one. DT takes Reol - No
title [Insane] from 5.2 to 7.5, so comparing a DT score against maps of
the same stored rating compares it against maps that are far easier than
what was actually played. osu! computes pp from the modded rating, so
anything comparing against pp has to use the modded rating too.

rosu-pp is the same difficulty calculation osu! uses, and it needs the
.osu file for the map. Those come from https://osu.ppy.sh/osu/{id}, which
is a plain file download rather than the API, and are cached on disk so a
map is fetched once.

Run this file directly to fetch the files for every map in the database.
"""

import argparse
import asyncio
import os
from pathlib import Path

import httpx
import rosu_pp_py
from aiolimiter import AsyncLimiter

BEATMAP_URL = "https://osu.ppy.sh/osu/{}"

CACHE = Path(os.environ.get("OSU_BEATMAP_CACHE", "beatmap-files"))

# Plain file downloads off the website rather than API calls, so this is
# politeness rather than a published limit.
DOWNLOADS_PER_SECOND = 5

# A few requests in flight so the rate is set by the limiter rather than
# by how long each round trip happens to take.
CONCURRENCY = 4


def cached_path(beatmap_id):
    return CACHE / f"{beatmap_id}.osu"


def settle(path, content):
    """
    Write through a temporary name and rename over the target.

    Renaming within a directory is atomic, so a run killed at any moment
    leaves either no file or a whole one. Writing in place would leave a
    truncated file that the next run mistakes for a complete download.
    """
    partial = path.with_suffix(".part")
    partial.write_bytes(content)
    os.replace(partial, path)


async def download(beatmap_id, client, limiter, attempts=3):
    """
    Fetch one .osu file.

    True when the file is now on disk, False when the map is gone for
    good, None when this attempt failed in a way worth retrying later. A
    map is only recorded as gone on an answer from the server that says
    so, never on a dropped connection, so a bad network cannot
    permanently mark maps as missing.
    """
    path = cached_path(beatmap_id)

    if path.exists():
        return path.stat().st_size > 0

    for attempt in range(attempts):
        try:
            async with limiter:
                response = await client.get(BEATMAP_URL.format(beatmap_id))
        except httpx.HTTPError:
            await asyncio.sleep(2 ** attempt)
            continue

        if response.status_code == 404 or not response.content.strip():
            # Deleted or unavailable. An empty file records that, so the
            # map is not requested again.
            settle(path, b"")
            return False

        if response.status_code >= 500:
            await asyncio.sleep(2 ** attempt)
            continue

        response.raise_for_status()
        settle(path, response.content)

        return True

    return None


def mods_of(mod_key):
    """Turn a stored mod_key such as 'DT,HD' into a list rosu-pp accepts."""
    if not mod_key or mod_key == "NM":
        return []

    return [part for part in mod_key.split(",") if part]


def star_rating(beatmap_id, mod_key, cache={}):
    """
    Modded star rating, or None if the file is missing or the mods are
    ones the calculator does not model.
    """
    key = (beatmap_id, mod_key)

    if key in cache:
        return cache[key]

    path = cached_path(beatmap_id)

    if not path.exists() or path.stat().st_size == 0:
        cache[key] = None
        return None

    try:
        beatmap = rosu_pp_py.Beatmap(path=str(path))
        stars = rosu_pp_py.Difficulty(mods=mods_of(mod_key)).calculate(beatmap)
        cache[key] = stars.stars
    except Exception:
        cache[key] = None

    return cache[key]


async def fetch_all(beatmap_ids, rate=DOWNLOADS_PER_SECOND, quiet=False):
    """
    Download every .osu file not already cached.

    Only the missing ones are requested, so this can be stopped and
    started freely and picks up where it left off. Re-running it after
    more sampling fetches exactly the maps that have appeared since.
    """
    CACHE.mkdir(parents=True, exist_ok=True)

    for stray in CACHE.glob("*.part"):
        stray.unlink()

    wanted = [b for b in beatmap_ids if not cached_path(b).exists()]

    if not wanted:
        return 0, 0, 0

    # Capacity of one rather than of `rate`: a bucket that starts full
    # would let the first burst go out all at once.
    limiter = AsyncLimiter(1, 1.0 / rate)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    done = got = gone = 0

    async def one(beatmap_id):
        nonlocal done, got, gone

        async with semaphore:
            outcome = await download(beatmap_id, client, limiter)

        done += 1

        if outcome is True:
            got += 1
        elif outcome is False:
            gone += 1

        if not quiet and done % 200 == 0:
            print(
                f"  {done}/{len(wanted)} tried, {got} on disk, "
                f"{gone} gone, {done - got - gone} left for next run"
            )

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=30
    ) as client:
        await asyncio.gather(*(one(b) for b in wanted))

    return len(wanted), got, gone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--min-scores",
        type=int,
        default=1,
        help="only fetch maps carrying at least this many scores",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DOWNLOADS_PER_SECOND,
        help="downloads per second across all connections",
    )

    args = parser.parse_args()

    from sample import connect_readonly

    conn = connect_readonly(args.db)

    ids = [
        int(row[0])
        for row in conn.execute(
            "select beatmap, count(*) n from Score "
            "group by beatmap having n >= ?",
            (args.min_scores,),
        )
    ]

    have = sum(1 for b in ids if cached_path(b).exists())

    print(
        f"{len(ids)} maps with {args.min_scores}+ scores, "
        f"{have} already cached, {len(ids) - have} to fetch"
    )

    tried, got, gone = asyncio.run(fetch_all(ids, args.rate))

    print(
        f"tried {tried}: {got} downloaded, {gone} gone from the site, "
        f"{tried - got - gone} unresolved and left for the next run"
    )
    print(f"cache in {CACHE}/")


if __name__ == "__main__":
    main()
