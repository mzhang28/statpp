#!/usr/bin/env python3
"""
Pull the o!TR tournament dataset into a local sqlite file.

o!TR publishes a weekly replica of its database as a compressed
PostgreSQL dump. It records which beatmaps tournaments put in their
mappools, every match and game played, and the score each player set. The
players and beatmaps carry their osu! ids, so all of it joins straight
onto osu.sqlite.

That matters here because a mappool is the opposite of farm. Someone
picked those maps to tell players apart, so they are an outside opinion
about the thing this project measures: a map's curve should be steeper on
a pooled map than on an unpooled one of the same star rating.

The dump is PostgreSQL, so it is restored into a PostgreSQL container
rather than parsed by hand. The container is named statpp-otr-pg, holds
nothing else, and is removed when the work is done.

    uv run otr_dataset.py            # fetch, load, extract, clean up
    uv run otr_dataset.py --keep     # leave the container running

The dataset is published by the osu! Tournament Rating project,
https://otr.stagec.net/, and its terms require that credit.
"""

import argparse
import csv
import hashlib
import io
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import requests

from sample import mod_key

INDEX = "https://data.otr.stagec.net/"
CONTAINER = "statpp-otr-pg"
IMAGE = "postgres:17-alpine"

WORK = Path(os.environ.get("OTR_WORK", Path.home() / ".cache" / "statpp-otr"))

# Legacy mod bits, as osu! has always encoded them. o!TR stores this
# integer rather than the acronym list the modern API returns.
MOD_BITS = [
    (1, "NF"), (2, "EZ"), (4, "TD"), (8, "HD"), (16, "HR"), (32, "SD"),
    (64, "DT"), (128, "RX"), (256, "HT"), (512, "NC"), (1024, "FL"),
    (2048, "AT"), (4096, "SO"), (8192, "AP"), (16384, "PF"),
    (1 << 30, "MR"),
]

# The osu! standard ruleset. o!TR numbers the others 1 to 3.
STANDARD = 0


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def newest_dump():
    """The most recent replica on the index page, and its checksum file."""
    page = requests.get(INDEX, timeout=60, headers={"User-Agent": "statpp"})
    page.raise_for_status()

    links = re.findall(r'href=(https://\S+?otr-public-replica_\S+?\.gz)>', page.text)

    if not links:
        raise SystemExit("No replica found on the index page.")

    return links[0], links[0] + ".sha256"


def fetch(url, checksum_url):
    """Download the dump once, and check it against its published hash."""
    WORK.mkdir(parents=True, exist_ok=True)
    target = WORK / url.rsplit("/", 1)[-1]

    wanted = requests.get(checksum_url, timeout=60).text.split()[0]

    if target.exists():
        if digest_of(target) == wanted:
            print(f"already have {target.name}")
            return target

        print(f"{target.name} does not match its checksum, fetching again")

    print(f"fetching {target.name}")

    with requests.get(url, stream=True, timeout=600) as reply:
        reply.raise_for_status()

        with open(target, "wb") as handle:
            for block in reply.iter_content(1 << 20):
                handle.write(block)

    if digest_of(target) != wanted:
        raise SystemExit("The download does not match its published checksum.")

    print(f"  {target.stat().st_size / 1e6:.0f}MB, checksum matches")

    return target


def digest_of(path):
    running = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            running.update(block)

    return running.hexdigest()


def container_state():
    found = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{CONTAINER}$",
         "--format", "{{.State}}"],
        capture_output=True, text=True,
    )

    return found.stdout.strip()


def start_postgres():
    """
    A PostgreSQL of our own, holding only this dataset.

    No port is published: everything talks to it through `docker exec`, so
    it cannot collide with another PostgreSQL on this machine.
    """
    state = container_state()

    if state == "running":
        print(f"{CONTAINER} is already running")
        return

    if state:
        run("docker", "rm", "-f", CONTAINER, stdout=subprocess.DEVNULL)

    print(f"starting {CONTAINER}")
    run(
        "docker", "run", "-d", "--name", CONTAINER,
        "-e", "POSTGRES_PASSWORD=statpp", "-e", "POSTGRES_DB=otr",
        IMAGE,
        stdout=subprocess.DEVNULL,
    )

    for _ in range(60):
        ready = subprocess.run(
            ["docker", "exec", CONTAINER, "pg_isready", "-U", "postgres"],
            capture_output=True,
        )

        if ready.returncode == 0:
            return

        import time

        time.sleep(1)

    raise SystemExit("PostgreSQL did not come up.")


def restore(dump):
    """Feed the dump through psql, decompressing as it goes."""
    already = query("select count(*) from information_schema.tables "
                    "where table_schema='public'")

    if already and int(already[0][0]) > 10:
        print("dump is already restored")
        return

    print(f"restoring {dump.name}, a few minutes")

    with open(dump, "rb") as handle:
        gunzip = subprocess.Popen(
            ["gunzip", "-c"], stdin=handle, stdout=subprocess.PIPE
        )
        psql = subprocess.Popen(
            ["docker", "exec", "-i", CONTAINER,
             "psql", "-q", "-U", "postgres", "-d", "otr"],
            stdin=gunzip.stdout, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        gunzip.stdout.close()
        _, complaints = psql.communicate()

    if psql.returncode != 0:
        raise SystemExit(complaints.decode(errors="replace")[-2000:])


def query(sql):
    """One statement, as rows of strings."""
    reply = subprocess.run(
        ["docker", "exec", CONTAINER, "psql", "-U", "postgres", "-d", "otr",
         "-tAF", "\t", "-c", sql],
        capture_output=True, text=True,
    )

    if reply.returncode != 0:
        return []

    return [line.split("\t") for line in reply.stdout.splitlines() if line]


def rows_of(sql):
    """Stream one query out as CSV, so a table of millions never lands in
    memory whole."""
    reply = subprocess.Popen(
        ["docker", "exec", CONTAINER, "psql", "-U", "postgres", "-d", "otr",
         "-c", f"copy ({sql}) to stdout with csv"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    stream = io.TextIOWrapper(reply.stdout, encoding="utf-8", newline="")

    yield from csv.reader(stream)

    reply.wait()


def mods_from(bits):
    """The stored mod bits, as the same mod key the sampler writes."""
    bits = int(bits or 0)
    acronyms = [name for bit, name in MOD_BITS if bits & bit]

    return mod_key(acronyms)[0]


SCHEMA = """
create table Tournament (
    id integer primary key,
    name text,
    abbreviation text,
    rank_range_lower integer,
    lobby_size integer,
    verified integer,
    start_time text,
    end_time text
);

create table Pool (
    tournament integer,
    beatmap integer,
    primary key (tournament, beatmap)
);

create table TournamentMatch (
    id integer primary key,
    osu_id integer,
    tournament integer,
    name text,
    start_time text,
    verified integer
);

create table Game (
    id integer primary key,
    osu_id integer,
    match integer,
    beatmap integer,
    mod_key text,
    start_time text,
    verified integer
);

create table TournamentPlayer (
    osu_id integer primary key,
    username text,
    country text
);

create table TournamentScore (
    game integer,
    player integer,
    accuracy real,
    total_score integer,
    max_combo integer,
    misses integer,
    passed integer,
    mod_key text,
    placement integer,
    verified integer
);

create index Pool_beatmap on Pool (beatmap);
create index Game_beatmap on Game (beatmap);
create index TournamentScore_player on TournamentScore (player);
create index TournamentScore_game on TournamentScore (game);
"""


def extract(path):
    """
    Write the parts this project can use, keyed by osu! ids.

    o!TR's own row ids stay only where one of its tables points at
    another. Everything that names a player or a beatmap uses the osu! id,
    so these tables join onto osu.sqlite without a lookup.

    Verification is kept rather than filtered on. o!TR marks a tournament
    or a match verified, rejected, or neither, and which of those to trust
    is a decision for the analysis and not for the download.
    """
    if path.exists():
        path.unlink()

    out = sqlite3.connect(path)
    out.executescript(SCHEMA)

    counts = {}

    counts["Tournament"] = fill(out, "Tournament", """
        select id, name, abbreviation, rank_range_lower_bound, lobby_size,
               verification_status, start_time, end_time
        from tournaments where ruleset = 0
    """, 8)

    counts["Pool"] = fill(out, "Pool", """
        select distinct j.tournaments_pooled_in_id, b.osu_id
        from join_pooled_beatmaps j
        join beatmaps b on b.id = j.pooled_beatmaps_id
        join tournaments t on t.id = j.tournaments_pooled_in_id
        where t.ruleset = 0 and b.ruleset = 0
    """, 2)

    counts["TournamentMatch"] = fill(out, "TournamentMatch", """
        select m.id, m.osu_id, m.tournament_id, m.name, m.start_time,
               m.verification_status
        from matches m join tournaments t on t.id = m.tournament_id
        where t.ruleset = 0
    """, 6)

    counts["Game"] = fill(out, "Game", """
        select g.id, g.osu_id, g.match_id, b.osu_id, g.mods, g.start_time,
               g.verification_status
        from games g
        left join beatmaps b on b.id = g.beatmap_id
        where g.ruleset = 0
    """, 7, mods_at=4)

    counts["TournamentPlayer"] = fill(out, "TournamentPlayer", """
        select distinct p.osu_id, p.username, p.country
        from players p join game_scores s on s.player_id = p.id
        where s.ruleset = 0
    """, 3)

    counts["TournamentScore"] = fill(out, "TournamentScore", """
        select s.game_id, p.osu_id, s.accuracy, s.score, s.max_combo,
               s.stat_miss, s.pass, s.mods, s.placement,
               s.verification_status
        from game_scores s join players p on p.id = s.player_id
        where s.ruleset = 0
    """, 10, mods_at=7, booleans=(6,))

    out.commit()
    out.execute("vacuum")
    out.close()

    return counts


def fill(out, table, sql, width, mods_at=None, booleans=()):
    """Copy one query into one table, in batches."""
    placeholders = ",".join("?" * width)
    statement = f"insert or replace into {table} values ({placeholders})"

    batch = []
    total = 0

    for row in rows_of(sql.strip()):
        row = [None if value == "" else value for value in row]

        if mods_at is not None:
            row[mods_at] = mods_from(row[mods_at])

        for spot in booleans:
            row[spot] = 1 if row[spot] == "t" else 0

        batch.append(row)

        if len(batch) >= 20000:
            out.executemany(statement, batch)
            total += len(batch)
            batch = []

    if batch:
        out.executemany(statement, batch)
        total += len(batch)

    print(f"  {table}: {total:,}")

    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out",
        default="otr.sqlite",
        help="sqlite file to write",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the PostgreSQL container running afterwards",
    )

    args = parser.parse_args()

    url, checksum_url = newest_dump()
    dump = fetch(url, checksum_url)

    start_postgres()
    restore(dump)

    print("extracting")
    counts = extract(Path(args.out))

    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}, {size:.0f}MB")

    if not args.keep:
        print(f"removing {CONTAINER}")
        run("docker", "rm", "-f", CONTAINER, stdout=subprocess.DEVNULL)

    print(
        "\nDataset by the osu! Tournament Rating project, "
        "https://otr.stagec.net/"
    )

    return 0 if counts else 1


if __name__ == "__main__":
    sys.exit(main())
