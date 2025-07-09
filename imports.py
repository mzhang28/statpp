from math import ceil
import json
from pony.orm import db_session, select
from tqdm import tqdm
from db import Beatmap, BeatmapMod, Score, User, conn, db
from prefect import flow, task
from prefect.states import Completed

# Num of players/beatmaps/scores: 10K, 200K, 54M
BATCH_SIZE = 10_000

db.generate_mapping(create_tables=True)

def import_users():
    cur = conn.cursor()
    cur.execute(f"select user_id, username from sample_users")
    data = cur.fetchall()
    with db_session:
        for user_id, username in tqdm(data):
            User(id=user_id, username=username, total_pp=0.0)
    print("Done.")

def import_beatmaps():
    cur = conn.cursor()
    cur.execute(f"select beatmap_id, artist, title, version from osu_beatmaps as b join osu_beatmapsets as s on b.beatmapset_id = s.beatmapset_id")
    data = cur.fetchall()
    with db_session:
        for beatmap_id, artist, title, version in tqdm(data):
            Beatmap(id=beatmap_id, artist=artist, title=title, diffname=version)
    print("Done.")

# @task
def fetch_old_scores(last_seen_id, limit):
    cur = conn.cursor()
    cur.execute("select id, user_id, beatmap_id, total_score, data from scores where id > %s order by id limit %s", (last_seen_id, limit))
    return cur.fetchall()

# @task
def insert_new_scores(data):
    score_ids = {row[0] for row in data}
    user_ids = {row[1] for row in data}
    beatmap_ids = {row[2] for row in data}
    with db_session:
        existing_scores = {s.id for s in select(s for s in Score if s.id in score_ids)}
        users = {u.id: u for u in User.select(lambda u: u.id in user_ids)}
        beatmaps = {b.id: b for b in Beatmap.select(lambda b: b.id in beatmap_ids)}
        beatmap_mods = {(bm.beatmap.id, bm.mod): bm for bm in BeatmapMod.select(lambda bm: bm.beatmap.id in beatmap_ids)}

        for id, user_id, beatmap_id, total_score, metadata_s in data:
            if id in existing_scores: continue
            metadata = json.loads(metadata_s)
            mods = metadata.get('mods', [])
            mods_l = []
            for mod in mods:
                if mod != "NF":
                    mods_l.append(mod['acronym'])
            mods_l.sort()
            mods_s = "|".join(mods_l)
            user = users[user_id]
            beatmap = beatmaps[beatmap_id]
            bm = beatmap_mods.get((beatmap.id, mods_s), None)
            if not bm: bm = BeatmapMod(beatmap=beatmap, difficulty=5.0, mod=mods_s)
            beatmap_mods[bm.beatmap.id, bm.mod] = bm
            Score(id=id, user=user, beatmap_mod=bm, score=total_score, score_pp=1.0)


# @flow(log_prints=True)
def import_scores():
    # cur = conn.cursor()
    # cur.execute(f"select count(*) from scores")
    # total: int = cur.fetchone()[0]
    total = 54126306
    num_batches = ceil(total / BATCH_SIZE)
    print("Total:", total, "# batches", num_batches)

    last_seen_id = 0
    for i in tqdm(range(num_batches)):
        rows = fetch_old_scores(last_seen_id, BATCH_SIZE)
        insert_new_scores(rows)
        last_seen_id = rows[-1][0]
        print(last_seen_id)

import_users()
import_beatmaps()
import_scores()
