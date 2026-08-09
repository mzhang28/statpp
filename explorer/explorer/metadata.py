"""
Song titles for the maps the page is showing, fetched when it shows them.

The sampler stores whatever the API happened to return, and a beatmap
first seen inside a score response carries no beatmapset block, so it has
no artist or title and reads as a bare set number. Those names are worth
having on screen and worth nothing to the fit, so they are not in
osu.sqlite. They live here instead, in their own file, fetched for the few
hundred maps a page actually shows and dropped again when the file grows
past its cap.

Least recently shown goes first, so the maps someone keeps looking at stay
and the ones they scrolled past once do not.
"""

import os
import sqlite3
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

TOKEN_URL = "https://osu.ppy.sh/oauth/token"
API_BASE = "https://osu.ppy.sh/api/v2"
API_VERSION = "20220705"

# The beatmaps endpoint takes this many ids per call.
BATCH = 50

# osu! allows 60 requests a minute. The page asks for a few batches at a
# time, so spacing them is enough and no bucket is needed.
SECONDS_BETWEEN = 1.2

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "metadata.sqlite"

# Roughly 150 bytes a row, so this caps the file around 3MB.
KEEP_ROWS = 20000

_lock = threading.Lock()
_token = {"value": None, "expires": 0.0}
_last_call = {"at": 0.0}


def connect():
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(CACHE, timeout=30)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("""
        create table if not exists beatmap (
            id integer primary key,
            artist text,
            title text,
            version text,
            creator text,
            set_id integer,
            fetched_at real,
            used_at real
        )
    """)
    conn.commit()

    return conn


def named(beatmap_ids):
    """
    The names already held for these maps, and a note that they were
    wanted just now.

    Marking them used on the way out is what makes the cap evict by what
    nobody is looking at rather than by what was fetched longest ago.
    """
    if not beatmap_ids:
        return {}

    wanted = list({int(b) for b in beatmap_ids})
    found = {}

    with _lock:
        conn = connect()

        try:
            for part in chunks(wanted, 500):
                holes = ",".join("?" * len(part))
                rows = conn.execute(
                    f"select id, artist, title, version, creator, set_id "
                    f"from beatmap where id in ({holes})",
                    part,
                ).fetchall()

                for beatmap_id, artist, title, version, creator, set_id in rows:
                    found[beatmap_id] = {
                        "artist": artist or "",
                        "title": title or "",
                        "version": version or "",
                        "creator": creator or "",
                        "set_id": set_id,
                    }

                conn.execute(
                    f"update beatmap set used_at = ? where id in ({holes})",
                    [time.time(), *part],
                )

            conn.commit()
        finally:
            conn.close()

    return found


def store(entries):
    with _lock:
        conn = connect()

        try:
            now = time.time()
            conn.executemany(
                "insert or replace into beatmap "
                "(id, artist, title, version, creator, set_id, "
                " fetched_at, used_at) values (?,?,?,?,?,?,?,?)",
                [
                    (
                        entry["id"], entry["artist"], entry["title"],
                        entry["version"], entry["creator"], entry["set_id"],
                        now, now,
                    )
                    for entry in entries
                ],
            )
            conn.commit()
        finally:
            conn.close()


def purge(keep=KEEP_ROWS):
    """Drop the least recently shown rows once the file holds too many."""
    with _lock:
        conn = connect()

        try:
            held = conn.execute("select count(*) from beatmap").fetchone()[0]

            if held <= keep:
                return held

            conn.execute(
                "delete from beatmap where id in ("
                "  select id from beatmap order by used_at desc limit -1 "
                f"  offset {int(keep)}"
                ")"
            )
            conn.commit()
            conn.execute("vacuum")

            return conn.execute("select count(*) from beatmap").fetchone()[0]
        finally:
            conn.close()


def held():
    """How many maps are cached, and what the file costs on disk."""
    if not CACHE.exists():
        return 0, 0

    with _lock:
        conn = connect()

        try:
            rows = conn.execute("select count(*) from beatmap").fetchone()[0]
        finally:
            conn.close()

    return rows, CACHE.stat().st_size


def token():
    if _token["value"] and time.time() < _token["expires"] - 60:
        return _token["value"]

    reply = requests.post(
        TOKEN_URL,
        data={
            "client_id": os.environ["OSU_CLIENT_ID"],
            "client_secret": os.environ["OSU_CLIENT_SECRET"],
            "grant_type": "client_credentials",
            "scope": "public",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    reply.raise_for_status()
    body = reply.json()

    _token["value"] = body["access_token"]
    _token["expires"] = time.time() + float(body.get("expires_in", 3600))

    return _token["value"]


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fetch(beatmap_ids, budget=6):
    """
    Ask the API for the maps not cached yet, newest request first.

    `budget` caps how many calls one page view can spend, so a list of
    several hundred unnamed maps names what it can now and the rest on the
    next look rather than holding the page for a minute.
    """
    wanted = [int(b) for b in beatmap_ids]

    if not wanted:
        return {}

    got = {}

    for part in list(chunks(wanted, BATCH))[:budget]:
        gap = SECONDS_BETWEEN - (time.time() - _last_call["at"])

        if gap > 0:
            time.sleep(gap)

        _last_call["at"] = time.time()

        try:
            reply = requests.get(
                API_BASE + "/beatmaps",
                params={"ids[]": part},
                headers={
                    "Authorization": f"Bearer {token()}",
                    "Accept": "application/json",
                    "x-api-version": API_VERSION,
                },
                timeout=30,
            )
            reply.raise_for_status()
            body = reply.json()
        except (requests.RequestException, KeyError, ValueError):
            # A page missing a few song titles is not worth an error, and
            # the next look will ask again.
            break

        entries = []

        for entry in body.get("beatmaps", []):
            listing = entry.get("beatmapset") or {}

            entries.append({
                "id": int(entry["id"]),
                "artist": listing.get("artist") or "",
                "title": listing.get("title") or "",
                "version": entry.get("version") or "",
                "creator": listing.get("creator") or "",
                "set_id": entry.get("beatmapset_id"),
            })

        if entries:
            store(entries)

            for entry in entries:
                got[entry["id"]] = {
                    "artist": entry["artist"],
                    "title": entry["title"],
                    "version": entry["version"],
                    "creator": entry["creator"],
                    "set_id": entry["set_id"],
                }

    purge()

    return got


def label(entry, version):
    """
    How a map reads on screen.

    The difficulty name is on the beatmap either way; the artist and title
    are what this cache exists to add, so a row with neither falls back to
    what the fit already knew.
    """
    if not entry or not (entry.get("artist") or entry.get("title")):
        return None

    difficulty = entry.get("version") or version or ""

    return f"{entry['artist']} - {entry['title']} [{difficulty}]"
