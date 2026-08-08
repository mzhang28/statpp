#!/usr/bin/env python3
"""
Stratified sampler for the osu! score graph.

Players are drawn from log-spaced slices of the global performance ranking
rather than crawled by map-overlap. The point is coverage of the ability
range: adjacent strata need to share map vocabulary for a joint
player/map scale to be identifiable at all, and log spacing is what keeps
that chain connected.

Expansion is round-robin across strata, so a run that hits its request
budget early still leaves balanced coverage instead of an over-explored
top stratum.
"""

import argparse
import asyncio
import itertools
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from pony.orm import (
    Database,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    db_session,
    select,
)

load_dotenv()

API_BASE = "https://osu.ppy.sh/api/v2"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"
API_VERSION = "20220705"

# osu! caps at 60 requests/minute. Leave a little headroom for the token
# refreshes and retries a long run inevitably spends.
REQUESTS_PER_MINUTE = 57

# Median API latency is ~1.3s, which is longer than the ~1.05s the rate
# limit allows between requests. Sequential code therefore tops out around
# 1/latency (~46/min) and can never reach the cap however good its
# limiter is. Keeping a few requests in flight is what closes that gap;
# the limiter, not the concurrency, is what bounds the rate.
CONCURRENCY = 4

RANKING_PAGE_SIZE = 50

# /rankings caps at page 200 (rank ~10k) and, crucially, does NOT error
# past it: page 201+ silently returns page 200's contents. Requesting more
# would quietly duplicate a stratum under a wrong label.
MAX_RANKING_PAGE = 200

# Global rankings only reach the top ~0.5% of players. Country rankings
# give ~10k per country, so deep pages of a small country land far below
# global rank 10k and extend the ability range downward.
# Roughly log-spaced in global rank. The comment on each line is the
# observed global rank span, since a country page number says nothing
# about where it lands globally. Country choice is by playerbase size:
# a country needs enough ranked players for deep pages to hold real
# accounts (EE page 200 is global ~2.7M at 2pp, i.e. dead accounts).
DEFAULT_PAGES = [
    (None, 1),      # #1-50
    (None, 2),      # #51-100
    (None, 4),      # ~#200
    (None, 8),      # ~#400
    (None, 16),     # ~#800
    (None, 32),     # ~#1.6k
    (None, 64),     # ~#3.2k
    (None, 128),    # ~#6.4k
    (None, 200),    # ~#10k  (global rankings cap out here)
    ("US", 80),     # ~#20k
    ("US", 140),    # ~#35k
    ("US", 200),    # ~#51k
    ("RU", 200),    # ~#102k
    ("DE", 200),    # ~#202k
    ("ID", 120),    # ~#260k
    ("ID", 200),    # ~#438k
]

# Countries to reach below global rank ~10k with, roughly by playerbase
# size. Order matters more than membership: page 200 of a small country is
# dead accounts, not low-ranked players, so the big ones get used first.
DEEP_COUNTRIES = [
    "US", "RU", "PH", "ID", "BR", "DE", "PL", "FR", "GB", "JP",
    "TH", "MY", "KR", "CA", "TW", "VN", "MX", "IT", "ES", "AU",
    "CL", "UA", "NL", "FI", "SE", "AR", "TR", "SG", "HK", "NO",
]

# Keep the initial graph focused on things relevant to performance ratings.
INTERESTING_STATUSES = {"ranked", "approved"}

# Mods that don't change how hard the resulting score was to produce.
# NC is DT with different audio; DA rewrites the map and is kept visible
# so those scores can be excluded later.
COSMETIC_MODS = {"CL", "NF", "SD", "PF", "MR"}
MOD_ALIASES = {"NC": "DT"}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

db = Database()


@db.on_connect(provider="sqlite")
def sqlite_pragmas(_db, connection):
    cursor = connection.cursor()

    # WAL is what lets analysis read while the sampler writes. Under the
    # default journal mode a writer takes an exclusive lock, so a long
    # sampling run would block every reader for its whole duration.
    cursor.execute("PRAGMA journal_mode = WAL")

    # Under WAL a writer still briefly excludes other writers; wait rather
    # than failing outright.
    cursor.execute("PRAGMA busy_timeout = 30000")

    # WAL already gives durability across process crashes. NORMAL only
    # risks the very last commits, and only on OS-level crash.
    cursor.execute("PRAGMA synchronous = NORMAL")

    cursor.close()


def bind_db(path):
    db.bind(
        provider="sqlite",
        filename=str(Path(path).resolve()),
        create_db=True,
    )
    db.generate_mapping(create_tables=True)


def connect_readonly(path=None):
    """
    Connection for analysis running alongside a live sampler.

    Deliberately not `mode=ro`: a genuinely read-only connection cannot
    create the -shm index a WAL database needs, so it fails whenever no
    writer happens to be attached. `query_only` gives the same protection
    without that failure mode.
    """
    import sqlite3

    path = path or os.environ.get("OSU_DB", "osu.sqlite")

    conn = sqlite3.connect(str(Path(path).resolve()), timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA query_only = 1")

    return conn


class Beatmap(db.Entity):
    id = PrimaryKey(int, size=64)

    beatmapset_id = Optional(int, size=64)
    version = Optional(str)

    stars = Optional(float)
    ar = Optional(float)
    od = Optional(float)
    cs = Optional(float)
    hp = Optional(float)
    bpm = Optional(float)

    length = Optional(int)
    playcount = Optional(int)

    mode = Optional(str)
    status = Optional(str)

    leaderboard_crawled = Required(bool, default=False)

    # pp of the worst score on the top-50 leaderboard: everyone absent
    # from it scored below this. A censoring bound, not a missing value.
    leaderboard_cutoff_pp = Optional(float)

    raw = Required(Json)

    scores = Set("Score")
    probes = Set("Probe")


class Stratum(db.Entity):
    """One sampled slice of a performance ranking."""

    label = PrimaryKey(str)

    # Empty for the global ranking, else a country code. Country strata
    # have low country ranks but arbitrarily deep global ranks.
    country = Optional(str)

    page = Required(int)
    rank_low = Required(int)
    rank_high = Required(int)

    fetched = Required(bool, default=False)

    players = Set("Player")


class Player(db.Entity):
    id = PrimaryKey(int, size=64)

    username = Optional(str)
    country = Optional(str)

    # Ranking snapshot at sample time. Global rank moves, so it is only
    # meaningful together with sampled_at.
    global_rank = Optional(int)
    pp = Optional(float)
    play_count = Optional(int)
    hit_accuracy = Optional(float)

    stratum = Optional(Stratum)
    sampled_at = Optional(datetime)

    # Selected for best-score expansion (vs. merely seen on a leaderboard).
    selected = Required(bool, default=False)

    # Number of fetched map leaderboards we've encountered them on.
    leaderboard_hits = Required(int, default=0)

    # Have we fetched /users/{id}/scores/best?
    best_crawled = Required(bool, default=False)

    # Truncation bound from the top-100: if best_count == 100, this player
    # has no unseen play worth more than best_cutoff_pp. Every map missing
    # from their list is censored at that value rather than unobserved.
    best_count = Optional(int)
    best_cutoff_pp = Optional(float)

    raw = Required(Json)

    scores = Set("Score")
    probes = Set("Probe")


class Probe(db.Entity):
    """
    One directly requested (player, beatmap) cell.

    Recorded whether or not a score came back: a miss means the player has
    no submitted play on the map at all, which is a different fact from a
    low one, and both are worth not re-requesting.
    """

    player = Required(Player)
    beatmap = Required(Beatmap)

    found = Required(bool)
    probed_at = Required(datetime)

    PrimaryKey(player, beatmap)


class Score(db.Entity):
    id = PrimaryKey(int, size=64)

    player = Required(Player)
    beatmap = Required(Beatmap)

    accuracy = Required(float)
    max_combo = Required(int)

    total_score = Required(int, size=64)
    legacy_total_score = Optional(int, size=64)

    pp = Optional(float)
    rank = Optional(str)
    passed = Required(bool)

    misses = Required(int)
    ended_at = Required(datetime)

    mods = Required(Json)

    # Canonical difficulty-relevant mod combination, e.g. "DT,HD".
    # The modelled item is (beatmap, mod_key), not the beatmap alone.
    mod_key = Required(str)

    # Lazer mods carry settings (custom rates, AR overrides). Those scores
    # aren't comparable to the plain combination and are flagged, not merged.
    mod_settings = Required(bool, default=False)

    statistics = Required(Json)
    maximum_statistics = Optional(Json)

    first_seen_via = Required(str)
    raw = Required(Json)


# ---------------------------------------------------------------------------
# osu! API
# ---------------------------------------------------------------------------

class RequestBudgetExhausted(Exception):
    pass


class NotFound(Exception):
    """The API has no such resource, e.g. a player with no play on a map."""


class OsuAPI:
    """
    Rate is bounded by a leaky bucket, not by spacing between calls, so
    concurrent callers share one budget and the cap holds no matter how
    many coroutines are in flight. Still per-process: two copies of the
    program running at once double the real rate.
    """

    def __init__(self, max_requests: int):
        self.client_id = os.environ["OSU_CLIENT_ID"]
        self.client_secret = os.environ["OSU_CLIENT_SECRET"]

        # Capacity of one, not of REQUESTS_PER_MINUTE. A leaky bucket
        # starts full, so AsyncLimiter(57, 60) would let the first 57
        # requests go out in a burst before throttling at all, which is
        # well over the cap on any sliding-window measure of it. Capacity
        # one paces evenly instead, and concurrency still helps because
        # the wait no longer includes each response's latency.
        self.limiter = AsyncLimiter(1, 60 / REQUESTS_PER_MINUTE)
        self.client = httpx.AsyncClient(timeout=30)
        self.token = None

        self.max_requests = max_requests
        self.requests_used = 0

        # Serialises token refresh so a 401 seen by several in-flight
        # requests doesn't trigger one refresh per request.
        self.token_lock = asyncio.Lock()

    async def __aenter__(self):
        await self.refresh_token()
        return self

    async def __aexit__(self, *exc):
        await self.client.aclose()

    async def refresh_token(self, stale=None):
        async with self.token_lock:
            # Another coroutine already refreshed while we waited.
            if stale is not None and self.token != stale:
                return

            r = await self.client.post(
                TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                    "scope": "public",
                },
                headers={"Accept": "application/json"},
            )
            r.raise_for_status()
            self.token = r.json()["access_token"]

    async def get(self, path, **params):
        for attempt in range(6):
            if self.requests_used >= self.max_requests:
                raise RequestBudgetExhausted()

            token = self.token

            async with self.limiter:
                self.requests_used += 1

                try:
                    r = await self.client.get(
                        API_BASE + path,
                        params=params,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Accept": "application/json",
                            "x-api-version": API_VERSION,
                        },
                    )
                except httpx.HTTPError as exc:
                    # Dropped connections and timeouts are as routine as
                    # 5xx over a run this long, and retry the same way.
                    print(f"        transport error, retrying: {exc}")
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue

            if r.status_code == 404:
                raise NotFound(path)

            if r.status_code == 401:
                await self.refresh_token(stale=token)
                continue

            if r.status_code == 429:
                delay = float(r.headers.get("Retry-After", 2 ** attempt))
                await asyncio.sleep(max(delay, 1))
                continue

            if 500 <= r.status_code < 600:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue

            r.raise_for_status()
            return r.json()

        raise RuntimeError(f"API request repeatedly failed: {path}")


async def run_pool(coro_fn, items, concurrency=CONCURRENCY):
    """
    Apply an async function over items with bounded concurrency.

    Order of completion is not order of submission, so anything that needs
    balanced partial results must order `items` accordingly rather than
    rely on this.
    """
    semaphore = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()

    async def one(item):
        if stop.is_set():
            return

        async with semaphore:
            if stop.is_set():
                return

            try:
                await coro_fn(item)
            except RequestBudgetExhausted:
                stop.set()

    await asyncio.gather(*(one(item) for item in items))

    if stop.is_set():
        raise RequestBudgetExhausted()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_time(value):
    if not value:
        return utcnow()

    # Store UTC-like naive datetime in SQLite.
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).replace(tzinfo=None)


def mod_key(mods):
    """
    Canonical mod combination, plus whether any mod carried settings.

    Lazer returns [{"acronym": "DT", "settings": {...}}, ...]; legacy
    scores may return bare strings.
    """
    acronyms = []
    has_settings = False

    for mod in mods or []:
        if isinstance(mod, dict):
            acronym = mod.get("acronym", "")
            if mod.get("settings"):
                has_settings = True
        else:
            acronym = str(mod)

        acronym = MOD_ALIASES.get(acronym, acronym)

        if not acronym or acronym in COSMETIC_MODS:
            continue

        acronyms.append(acronym)

    return ",".join(sorted(set(acronyms))) or "NM", has_settings


def beatmap_is_interesting(data):
    return (
        data.get("mode") == "osu"
        and data.get("status") in INTERESTING_STATUSES
    )


def page_for_rank(rank):
    page = (rank + RANKING_PAGE_SIZE - 1) // RANKING_PAGE_SIZE
    return max(1, min(MAX_RANKING_PAGE, page))


def stratum_label(country, page):
    low = (page - 1) * RANKING_PAGE_SIZE + 1

    if country:
        return f"{country}-r{low:05d}"

    return f"r{low:05d}"


# ---------------------------------------------------------------------------
# Entity ingestion
# ---------------------------------------------------------------------------

def upsert_beatmap(data):
    beatmap_id = int(data["id"])

    b = Beatmap.get(id=beatmap_id)

    values = {
        "beatmapset_id": data.get("beatmapset_id"),
        "version": data.get("version"),
        "stars": data.get("difficulty_rating"),
        "ar": data.get("ar"),
        "od": data.get("accuracy"),
        "cs": data.get("cs"),
        "hp": data.get("drain"),
        "bpm": data.get("bpm"),
        "length": data.get("total_length"),
        "playcount": data.get("playcount"),
        "mode": data.get("mode"),
        "status": data.get("status"),
    }

    if b is None:
        b = Beatmap(
            id=beatmap_id,
            raw=data,
            **values,
        )
    else:
        for key, value in values.items():
            if value is not None:
                setattr(b, key, value)

        # Prefer whichever response gave us more information.
        if len(data) >= len(b.raw):
            b.raw = data

    return b


def upsert_player(data):
    player_id = int(data["id"])
    p = Player.get(id=player_id)

    if p is None:
        return Player(
            id=player_id,
            username=data.get("username"),
            country=data.get("country_code"),
            raw=data,
        )

    if data.get("username"):
        p.username = data["username"]

    if data.get("country_code"):
        p.country = data["country_code"]

    if len(data) >= len(p.raw):
        p.raw = data

    return p


def ingest_score(score, beatmap, source, known_player=None):
    score_id = int(score["id"])

    existing = Score.get(id=score_id)
    if existing is not None:
        return existing

    user_data = score.get("user")

    if user_data:
        player = upsert_player(user_data)
    elif known_player is not None:
        player = known_player
    else:
        player = Player.get(id=int(score["user_id"]))

        if player is None:
            player = Player(
                id=int(score["user_id"]),
                raw={},
            )

    stats = score.get("statistics") or {}
    key, has_settings = mod_key(score.get("mods"))

    return Score(
        id=score_id,
        player=player,
        beatmap=beatmap,
        accuracy=float(score["accuracy"]),
        max_combo=int(score.get("max_combo") or 0),
        total_score=int(score.get("total_score") or 0),
        legacy_total_score=score.get("legacy_total_score"),
        pp=score.get("pp"),
        rank=score.get("rank"),
        passed=bool(score.get("passed", True)),
        misses=int(stats.get("miss") or 0),
        ended_at=parse_time(score.get("ended_at")),
        mods=score.get("mods") or [],
        mod_key=key,
        mod_settings=has_settings,
        statistics=stats,
        maximum_statistics=score.get("maximum_statistics"),
        first_seen_via=source,
        raw=score,
    )


# ---------------------------------------------------------------------------
# Stratum sampling
# ---------------------------------------------------------------------------

def select_from_stratum(stratum, per_stratum, rng):
    """
    Bring a stratum's selected count up to per_stratum.

    Sampling is from within the page rather than off the top of it, so a
    stratum isn't systematically represented by its own strongest edge.
    """
    members = list(stratum.players)
    unselected = [p for p in members if not p.selected]
    shortfall = min(per_stratum, len(members)) - (len(members) - len(unselected))

    if shortfall <= 0:
        return 0

    chosen = (
        unselected if shortfall >= len(unselected)
        else rng.sample(unselected, shortfall)
    )

    for p in chosen:
        p.selected = True

    return len(chosen)


async def fetch_stratum(api, country, page, per_stratum, rng):
    label = stratum_label(country, page)

    with db_session:
        s = Stratum.get(label=label)

        if s is not None and s.fetched:
            # Already have the page's membership stored, so a raised
            # --per-stratum tops up from it rather than refetching.
            added = select_from_stratum(s, per_stratum, rng)
            selected = sum(1 for p in s.players if p.selected)

            print(
                f"STRATUM {label:<12} (cached)"
                f"{'':<30}{selected} selected (+{added})"
            )
            return

    params = {"page": page}
    if country:
        params["country"] = country

    payload = await api.get("/rankings/osu/performance", **params)
    ranking = payload.get("ranking", [])

    with db_session:
        s = Stratum.get(label=label)

        if s is None:
            s = Stratum(
                label=label,
                country=country or "",
                page=page,
                rank_low=(page - 1) * RANKING_PAGE_SIZE + 1,
                rank_high=page * RANKING_PAGE_SIZE,
            )

        entries = []

        for entry in ranking:
            user_data = entry.get("user")
            if not user_data:
                continue

            p = upsert_player(user_data)

            p.global_rank = entry.get("global_rank")
            p.pp = entry.get("pp")
            p.play_count = entry.get("play_count")
            p.hit_accuracy = entry.get("hit_accuracy")
            p.sampled_at = utcnow()

            # A player can surface in more than one ranking slice; keep the
            # first assignment so re-runs don't reshuffle the strata.
            if p.stratum is None:
                p.stratum = s

            entries.append(p)

        s.fetched = True
        n_chosen = select_from_stratum(s, per_stratum, rng)

        # For country strata the page number is a country rank; the global
        # ranks it actually covers are the interesting thing.
        ranks = [p.global_rank for p in entries if p.global_rank]
        span = f"global #{min(ranks)}-{max(ranks)}" if ranks else "empty"

    print(
        f"STRATUM {label:<12} page={page:<4} {span:<26} "
        f"{len(entries)} players, {n_chosen} selected"
    )


async def fetch_strata(api, pages, per_stratum, seed):
    # Sequential: strata are few, and each one's selection feeds the
    # expansion queue that follows.
    rng = random.Random(seed)

    for country, page in pages:
        await fetch_stratum(api, country, page, per_stratum, rng)


def widen_pages(have):
    """
    Pages next to one already sampled.

    A ranking page holds exactly 50 players, which caps how dense a single
    stratum can get. The page beside it covers nearly the same ability, so
    it raises the number of players available to co-play any given map,
    which is what a within-stratum correlation is short of.
    """
    for country, page in sorted(have):
        for step in (-1, 1):
            neighbour = (country, page + step)

            if 1 <= neighbour[1] <= MAX_RANKING_PAGE and neighbour not in have:
                yield neighbour


def bisect_pages(have):
    """
    Geometric midpoints of the widest gaps in each country's ladder.

    Adjacent strata have to share map vocabulary for the joint scale to be
    identifiable, and the widest gap in log rank is where that chain is
    thinnest. Widest gap first, so the weakest link is addressed first.
    """
    by_country = defaultdict(list)

    for country, page in have:
        by_country[country].append(page)

    gaps = []

    for country, pages in by_country.items():
        pages.sort()

        for low, high in zip(pages, pages[1:]):
            mid = int((low * high) ** 0.5)

            if low < mid < high and (country, mid) not in have:
                gaps.append((high / low, country, mid))

    for _, country, page in sorted(gaps, reverse=True):
        yield (country, page)


def deepen_pages(have):
    """
    The deepest page of a country not yet sampled.

    Country rankings are the only way below global rank ~10k. Biggest
    playerbase first, because page 200 of a small country is dead accounts
    rather than low-ranked players.
    """
    sampled = {country for country, _ in have}

    for country in DEEP_COUNTRIES:
        if country not in sampled:
            yield (country, MAX_RANKING_PAGE)


@db_session
def proposed_pages(n):
    """
    Ranking pages worth sampling that have not been sampled yet.

    The three policies buy different things, so they take turns rather
    than one running to exhaustion first. Every candidate is checked
    against what is already stored, so repeated runs keep proposing new
    pages instead of the same page forever.
    """
    have = {(s.country, s.page) for s in select(s for s in Stratum)}

    policies = [widen_pages(have), bisect_pages(have), deepen_pages(have)]

    chosen = []
    taken = set()

    while len(chosen) < n and policies:
        for policy in list(policies):
            if len(chosen) >= n:
                break

            for candidate in policy:
                if candidate not in have and candidate not in taken:
                    taken.add(candidate)
                    chosen.append(candidate)
                    break
            else:
                # Exhausted, so stop asking it.
                policies.remove(policy)

    # Global strata are stored with an empty country, the API wants none.
    return [(country or None, page) for country, page in chosen]


# ---------------------------------------------------------------------------
# Player expansion
# ---------------------------------------------------------------------------

@db_session
def pending_by_stratum():
    """Unexpanded selected players, grouped by stratum label."""
    pending = defaultdict(list)

    rows = select(
        p for p in Player
        if p.selected and not p.best_crawled
    )[:]

    for p in rows:
        label = p.stratum.label if p.stratum else "?"
        pending[label].append(p.id)

    for ids in pending.values():
        ids.sort()

    return dict(pending)


async def crawl_player_best(api, player_id):
    with db_session:
        player = Player[player_id]

        if player.best_crawled:
            return

        username = player.username or str(player.id)
        label = player.stratum.label if player.stratum else "?"
        rank = player.global_rank

    # Get User Scores includes beatmap + beatmapset for "best" scores.
    scores = await api.get(
        f"/users/{player_id}/scores/best",
        mode="osu",
        legacy_only=0,
        limit=100,
        offset=0,
    )

    with db_session:
        player = Player[player_id]
        seen_maps = set()

        for score in scores:
            beatmap_data = score.get("beatmap")

            if not beatmap_data:
                continue

            beatmap = upsert_beatmap(beatmap_data)

            ingest_score(
                score,
                beatmap,
                source="user_best",
                known_player=player,
            )

            seen_maps.add(beatmap.id)

        pps = [s.get("pp") for s in scores if s.get("pp") is not None]

        player.best_count = len(scores)
        player.best_cutoff_pp = min(pps) if pps else None
        player.best_crawled = True

        cutoff = player.best_cutoff_pp

    cutoff_note = f", cutoff={cutoff:.1f}pp" if cutoff else ""

    print(
        f"PLAYER  {label}  #{rank}  {username}  "
        f"{len(scores)} best, {len(seen_maps)} maps{cutoff_note}"
    )


async def expand_players(api):
    """
    Round-robin across strata.

    If the budget runs out mid-run the coverage stays balanced across the
    ability range instead of being depth-first in the top stratum. The
    pool preserves submission order closely enough for that to hold.
    """
    pending = pending_by_stratum()

    if not pending:
        print("No selected players left to expand.")
        return

    queues = [pending[label] for label in sorted(pending)]

    order = [
        player_id
        for player_id in itertools.chain.from_iterable(
            itertools.zip_longest(*queues)
        )
        if player_id is not None
    ]

    await run_pool(lambda pid: crawl_player_best(api, pid), order)


# ---------------------------------------------------------------------------
# Map leaderboards
# ---------------------------------------------------------------------------

async def crawl_map(api, beatmap_id):
    """
    Top-50 leaderboard for one map.

    Worth doing not for the extra players (they're all elite, so it adds
    little vertical spread) but for the 50th-place cutoff, which bounds
    everyone absent from the board.
    """
    with db_session:
        b = Beatmap.get(id=beatmap_id)

        if b is not None and b.leaderboard_crawled:
            return

        version = b.version if b is not None else None

    if version is None:
        data = await api.get(f"/beatmaps/{beatmap_id}")
        with db_session:
            upsert_beatmap(data)
            version = Beatmap[beatmap_id].version

    payload = await api.get(
        f"/beatmaps/{beatmap_id}/scores",
        mode="osu",
        legacy_only=0,
    )

    scores = payload.get("scores", [])

    with db_session:
        beatmap = Beatmap[beatmap_id]
        seen_players = set()

        for score in scores:
            player_id = int(score["user_id"])

            ingest_score(
                score,
                beatmap,
                source="leaderboard",
            )

            if player_id not in seen_players:
                seen_players.add(player_id)

                p = Player.get(id=player_id)
                if p is not None:
                    p.leaderboard_hits += 1

        pps = [s.get("pp") for s in scores if s.get("pp") is not None]

        beatmap.leaderboard_crawled = True
        beatmap.leaderboard_cutoff_pp = min(pps) if pps else None

    print(
        f"MAP     {beatmap_id}  {version}  "
        f"{len(scores)} scores"
    )


# ---------------------------------------------------------------------------
# Panel filling
# ---------------------------------------------------------------------------

@db_session
def stratum_panels(n_items, n_users, rng):
    """
    One rectangle per stratum: the maps that stratum plays, against its
    own players.

    Top-100 sampling spreads ~5k scores over ~3k maps per stratum, so
    almost no map pair shares enough players to correlate. Probing a
    chosen rectangle is what makes those pairs exist, and it reaches the
    plays below a player's top-100 cutoff that sampling cannot see.

    The rectangle is per stratum because map popularity is: the globally
    most-played maps are ones rank-400k players have never touched, so
    probing those against them mostly returns 404. Counting plays within
    the stratum picks maps its own members actually play, which is also
    the overlap a within-stratum correlation needs.
    """
    stratum_of = {}
    by_stratum = defaultdict(list)

    for player_id, stratum in select(
        (p.id, p.stratum) for p in Player if p.best_crawled
    ):
        label = stratum.label if stratum else "?"

        stratum_of[player_id] = label
        by_stratum[label].append(player_id)

    played = defaultdict(lambda: defaultdict(set))

    for beatmap_id, player_id in select(
        (s.beatmap.id, s.player.id) for s in Score
        if s.first_seen_via == "user_best"
    ):
        label = stratum_of.get(player_id)

        if label is not None:
            played[label][beatmap_id].add(player_id)

    panels = []

    for label in sorted(by_stratum):
        users = sorted(by_stratum[label])

        if len(users) > n_users:
            users = sorted(rng.sample(users, n_users))

        counts = played[label]

        # Play counts tie heavily down the tail, so break ties on id:
        # otherwise the panel changes between runs at the same seed.
        items = sorted(counts, key=lambda b: (-len(counts[b]), b))[:n_items]

        if items and users:
            panels.append((label, items, users))

    return panels


@db_session
def known_cells(items):
    """
    Cells that already have an answer: an observed score, or a probe that
    came back empty. Re-requesting either buys nothing, and a probe that
    found no score is as much an answer as one that did.
    """
    wanted = set(items)
    known = set()

    for beatmap_id, player_id in select(
        (s.beatmap.id, s.player.id) for s in Score
    ):
        if beatmap_id in wanted:
            known.add((beatmap_id, player_id))

    for beatmap_id, player_id in select(
        (pr.beatmap.id, pr.player.id) for pr in Probe
    ):
        if beatmap_id in wanted:
            known.add((beatmap_id, player_id))

    return known


def core_columns(panels, known):
    """
    Each stratum's rectangle as a list of complete columns, taken
    round-robin across strata.

    Item-major, so a run that stops early leaves whole columns behind:
    two complete columns share every player and so pair with each other,
    whereas a user-major order leaves every column partial and no pair
    usable. Round-robin across strata so that truncation costs each
    stratum equally rather than emptying the last one.
    """
    per_stratum = [
        [
            [
                (beatmap_id, user_id)
                for user_id in users
                if (beatmap_id, user_id) not in known
            ]
            for beatmap_id in items
        ]
        for _, items, users in panels
    ]

    return [
        column
        for group in itertools.zip_longest(*per_stratum)
        for column in group
        if column
    ]


def explore_cells(panels, known, n, rng):
    """
    Random cells outside every stratum's own rectangle.

    These are the cross-stratum bridges: one stratum's maps against
    another's players. Nothing else ties the per-stratum rectangles into
    a single scale, since each rectangle on its own is only internally
    comparable. A 404 is a real observation here rather than a wasted
    request: it says the player has never submitted a play on the map.

    Drawn by rejection rather than by building the full map x player
    product, which is over an order of magnitude larger than the number
    of cells drawn from it.
    """
    items = sorted({b for _, block_items, _ in panels for b in block_items})
    users = sorted({u for _, _, block_users in panels for u in block_users})

    core = {
        (beatmap_id, user_id)
        for _, block_items, block_users in panels
        for beatmap_id in block_items
        for user_id in block_users
    }

    chosen = set()

    # Stop once the draw stops finding anything new, which is what
    # happens when the space outside the core is nearly used up.
    misses = 0

    while len(chosen) < n and misses < 10_000:
        cell = (rng.choice(items), rng.choice(users))

        if cell in core or cell in known or cell in chosen:
            misses += 1
            continue

        chosen.add(cell)
        misses = 0

    return sorted(chosen)


def fill_order(panels, known, explore_fraction, budget, rng):
    """
    Every cell to probe, tagged core or explore, in probing order.

    Exploration cells are spread through the run rather than appended, so
    a run that stops early still has the intended mix of both kinds.
    """
    columns = core_columns(panels, known)
    n_core = sum(len(column) for column in columns)

    # Exploration is a share of the work, and it is also what keeps a run
    # productive after the rectangles are complete: at that point random
    # cells are the only cells left that this command can reach, and a run
    # given a budget should spend it rather than report nothing to do.
    n_explore = max(
        round(n_core * explore_fraction / (1 - explore_fraction)),
        budget - n_core,
    )

    explore = explore_cells(panels, known, n_explore, rng)

    order = []
    taken = 0

    for i, column in enumerate(columns, start=1):
        order.extend(("core", b, u) for b, u in column)

        # Keep the running mix at explore_fraction rather than letting
        # exploration bunch at either end.
        want = round(len(explore) * i / len(columns))

        order.extend(("explore", b, u) for b, u in explore[taken:want])
        taken = want

    order.extend(("explore", b, u) for b, u in explore[taken:])

    return order


async def probe_cell(api, beatmap_id, user_id):
    try:
        payload = await api.get(
            f"/beatmaps/{beatmap_id}/scores/users/{user_id}",
            mode="osu",
            legacy_only=0,
        )
    except NotFound:
        with db_session:
            Probe(
                player=Player[user_id],
                beatmap=Beatmap[beatmap_id],
                found=False,
                probed_at=utcnow(),
            )
        return False

    score = payload.get("score")

    with db_session:
        beatmap = Beatmap[beatmap_id]

        if score:
            ingest_score(score, beatmap, source="probe")

        Probe(
            player=Player[user_id],
            beatmap=beatmap,
            found=bool(score),
            probed_at=utcnow(),
        )

    return bool(score)


def hit_rates(done, found):
    return ", ".join(
        f"{kind} {found[kind]}/{done[kind]}"
        f" ({100.0 * found[kind] / done[kind]:.0f}%)"
        for kind in sorted(done) if done[kind]
    )


async def fill_panel(api, n_items, n_users, explore_fraction, seed, limit=None):
    rng = random.Random(seed)

    panels = stratum_panels(n_items, n_users, rng)

    if not panels:
        print("Nothing to fill: expand some players first.")
        return

    known = known_cells(
        {b for _, block_items, _ in panels for b in block_items}
    )

    budget = api.max_requests - api.requests_used

    if limit is not None:
        budget = min(budget, limit)

    order = fill_order(panels, known, explore_fraction, budget, rng)

    if limit is not None:
        order = order[:limit]

    if not order:
        print("Nothing to fill: every panel cell already has a score or probe.")
        return

    n_core = sum(1 for kind, _, _ in order if kind == "core")

    print(
        f"panel {len(panels)} strata x {n_items} maps x {n_users} players, "
        f"{len(order)} cells to probe "
        f"({n_core} core, {len(order) - n_core} exploration, "
        f"~{len(order) / REQUESTS_PER_MINUTE / 60:.1f}h)"
    )

    done = defaultdict(int)
    found = defaultdict(int)

    async def one(cell):
        kind, beatmap_id, user_id = cell

        hit = await probe_cell(api, beatmap_id, user_id)

        done[kind] += 1
        found[kind] += hit

        probed = sum(done.values())

        if probed % 100 == 0:
            print(
                f"        {probed}/{len(order)} probed, "
                f"{hit_rates(done, found)}"
            )

    await run_pool(one, order)

    print(f"        {sum(done.values())} probed, {hit_rates(done, found)}")


# ---------------------------------------------------------------------------
# Growing the graph
# ---------------------------------------------------------------------------

async def grow(api, args):
    """
    Expand the space and densify it, alternating, until the budget runs out.

    One process rather than two, because the rate limiter is per-process:
    a sampler and a filler running side by side would each pace themselves
    to 57 requests/minute and put the pair over the 60/minute cap.

    Expansion comes first in each cycle. New strata bring new players, and
    those players widen the panel that the filling stage then works on, so
    running it the other way round would fill a panel that is about to
    change.
    """
    cycle = 0

    while True:
        cycle += 1
        before = api.requests_used

        pages = proposed_pages(args.grow_pages)

        if pages:
            print(f"\n=== cycle {cycle}: {len(pages)} new pages ===")

            await fetch_strata(api, pages, args.per_stratum, args.seed)
            await expand_players(api)
        else:
            print(f"\n=== cycle {cycle}: no unsampled pages left ===")

        print(f"=== cycle {cycle}: filling ===")

        await fill_panel(
            api,
            args.fill_items,
            args.fill_users,
            args.fill_explore,
            args.seed,
            limit=args.grow_cycle,
        )

        # Neither stage found anything to request, so another cycle would
        # do the same nothing again.
        if api.requests_used == before:
            print("\nNothing left to expand or fill.")
            return


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


@db_session
def report():
    strata = select(s for s in Stratum)[:]

    if not strata:
        print("No strata sampled yet.")
        return

    # Map vocabulary per stratum, restricted to expanded players. The item
    # is (beatmap, mod_key), so difficulty-changing mods split the map.
    maps_by_stratum = {}
    rows = []

    for s in strata:
        players = [p for p in s.players if p.selected]
        expanded = [p for p in players if p.best_crawled]

        items = set()
        n_scores = 0

        for p in expanded:
            for score in p.scores:
                if score.first_seen_via == "user_best":
                    items.add((score.beatmap.id, score.mod_key))
                    n_scores += 1

        maps_by_stratum[s.label] = items

        rows.append({
            "stratum": s,
            "selected": len(players),
            "expanded": len(expanded),
            "items": items,
            "scores": n_scores,
            "cutoff": median(
                [p.best_cutoff_pp for p in expanded if p.best_cutoff_pp]
            ),
            # Country pages aren't comparable to global ones; order by the
            # global ranks the stratum actually landed on.
            "rank": median([p.global_rank for p in players if p.global_rank]),
        })

    rows.sort(key=lambda r: r["rank"] or 0)

    print()
    print(
        f"{'stratum':<13}{'median rank':>12}{'sel':>5}{'exp':>5}"
        f"{'items':>8}{'scores':>8}{'cutoff pp':>11}"
    )
    print("-" * 62)

    for r in rows:
        print(
            f"{r['stratum'].label:<13}"
            f"{r['rank'] or 0:>12,}"
            f"{r['selected']:>5}{r['expanded']:>5}"
            f"{len(r['items']):>8}{r['scores']:>8}"
            f"{r['cutoff'] or 0.0:>11.1f}"
        )

    # The identifiability question: do adjacent strata share any items?
    labels = [r["stratum"].label for r in rows if r["items"]]

    if len(labels) < 2:
        return

    print()
    print("item-set overlap between strata (shared / jaccard %)")
    print()

    print(" " * 13 + "".join(f"{lab:>14}" for lab in labels))

    for a in labels:
        row = f"{a:<13}"
        for b in labels:
            if a == b:
                row += f"{'-':>14}"
                continue

            ma, mb = maps_by_stratum[a], maps_by_stratum[b]
            shared = len(ma & mb)
            union = len(ma | mb)
            jac = 100.0 * shared / union if union else 0.0
            row += f"{f'{shared}/{jac:.1f}%':>14}"

        print(row)

    print()
    print("adjacent-stratum connectivity (the chain that has to hold):")

    for a, b in zip(labels, labels[1:]):
        ma, mb = maps_by_stratum[a], maps_by_stratum[b]
        shared = len(ma & mb)
        smaller = min(len(ma), len(mb)) or 1
        print(
            f"  {a} <-> {b}: {shared} shared items "
            f"({100.0 * shared / smaller:.1f}% of the smaller vocabulary)"
        )


@db_session
def print_stats(api):
    n_maps = select(b for b in Beatmap).count()
    n_players = select(p for p in Player).count()
    n_selected = select(p for p in Player if p.selected).count()
    n_expanded = select(p for p in Player if p.best_crawled).count()
    n_scores = select(s for s in Score).count()
    n_strata = select(s for s in Stratum if s.fetched).count()

    print()
    print(f"requests:          {api.requests_used}/{api.max_requests}")
    print(f"strata:            {n_strata}")
    print(f"maps:              {n_maps}")
    print(f"players:           {n_players}")
    print(f"players selected:  {n_selected}")
    print(f"players expanded:  {n_expanded}")
    print(f"scores:            {n_scores}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_pages(spec):
    """
    --pages accepts ranking pages (`1,2,4`), ranks (`r1,r1000`), and a
    country prefix for either (`EE:200`, `EE:r5000`).

    Pages past MAX_RANKING_PAGE are clamped rather than requested: the API
    returns page 200's contents for them without any error.
    """
    pages = []

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        country = None
        if ":" in part:
            country, part = part.split(":", 1)
            country = country.strip().upper()

        part = part.strip()

        if part.lower().startswith("r"):
            page = page_for_rank(int(part[1:]))
        else:
            page = int(part)

        if page > MAX_RANKING_PAGE:
            print(
                f"warning: page {page} clamped to {MAX_RANKING_PAGE} "
                f"(the API silently repeats page {MAX_RANKING_PAGE} past it)"
            )
            page = MAX_RANKING_PAGE

        pages.append((country, max(1, page)))

    seen = set()
    return [p for p in pages if not (p in seen or seen.add(p))]


def fraction(value):
    """A share of the run, so 1.0 is excluded: it would leave no core."""
    x = float(value)

    if not 0.0 <= x < 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not in [0, 1)")

    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "command",
        choices=["sample", "report", "maps", "fill", "grow"],
        nargs="?",
        default="sample",
        help="sample: fetch strata and expand players; "
             "report: stratum coverage and overlap; "
             "maps: crawl leaderboards for the given beatmap IDs; "
             "fill: densify the panel by probing (player, map) cells; "
             "grow: alternate sampling and filling until the budget ends",
    )

    parser.add_argument(
        "beatmaps",
        metavar="BEATMAP_ID",
        type=int,
        nargs="*",
        help="beatmap IDs for the `maps` command",
    )

    parser.add_argument("--db", default=os.environ.get("OSU_DB", "osu.sqlite"))

    parser.add_argument(
        "--pages",
        default=",".join(
            f"{c}:{p}" if c else str(p) for c, p in DEFAULT_PAGES
        ),
        help="ranking pages (4), ranks (r1000), or country-scoped "
             "versions of either (EE:200, EE:r5000)",
    )

    parser.add_argument(
        "--per-stratum",
        type=int,
        default=15,
        help="players to expand per stratum (page holds 50)",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=300,
        help="maximum API GET requests this run",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for within-stratum player selection",
    )

    parser.add_argument(
        "--fill-items",
        type=int,
        default=60,
        help="panel width for `fill`: maps per stratum, ranked by plays "
             "within that stratum",
    )

    parser.add_argument(
        "--fill-users",
        type=int,
        default=150,
        help="panel height for `fill`: players probed per map, drawn from "
             "the map's own stratum (a stratum holds at most 50)",
    )

    parser.add_argument(
        "--fill-explore",
        type=fraction,
        default=0.15,
        help="share of `fill` probes spent on random cells outside the "
             "per-stratum rectangles, which are what bridge strata",
    )

    parser.add_argument(
        "--grow-pages",
        type=int,
        default=4,
        help="new ranking pages `grow` adds per cycle",
    )

    parser.add_argument(
        "--grow-cycle",
        type=int,
        default=2000,
        help="cells `grow` probes per cycle before expanding again",
    )

    args = parser.parse_args()

    bind_db(args.db)

    if args.command == "report":
        report()
        return

    asyncio.run(run(args))


async def run(args):
    async with OsuAPI(args.requests) as api:
        try:
            if args.command == "maps":
                await run_pool(
                    lambda bid: crawl_map(api, bid),
                    args.beatmaps,
                )

            elif args.command == "fill":
                await fill_panel(
                    api,
                    args.fill_items,
                    args.fill_users,
                    args.fill_explore,
                    args.seed,
                )

            elif args.command == "grow":
                await grow(api, args)

            else:
                await fetch_strata(
                    api,
                    parse_pages(args.pages),
                    args.per_stratum,
                    args.seed,
                )
                await expand_players(api)

        except RequestBudgetExhausted:
            print("\nRequest budget exhausted.")

        finally:
            print_stats(api)


if __name__ == "__main__":
    main()
