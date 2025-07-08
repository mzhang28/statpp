from math import ceil
import json
from pony.orm import db_session
from tqdm import tqdm
from db import Beatmap, Score, User, conn, db
from prefect import flow, task
from prefect.states import Completed

# Num of players/beatmaps/scores: 10K, 200K, 54M
BATCH_SIZE = 1_000

db.generate_mapping(create_tables=True)

def import_users():
    cur = conn.cursor()
    cur.execute(f"select user_id, username from sample_users")
    data = cur.fetchall()
    with db_session:
        for user_id, username in tqdm(data):
            User(id=user_id, username=username, total_pp=0.0, normalized_pp=0.0)
    print("Done.")

def import_beatmaps():
    cur = conn.cursor()
    cur.execute(f"select beatmap_id, artist, title, version from osu_beatmaps as b join osu_beatmapsets as s on b.beatmapset_id = s.beatmapset_id")
    data = cur.fetchall()
    with db_session:
        for beatmap_id, artist, title, version in tqdm(data):
            Beatmap(id=beatmap_id, difficulty=1.0, artist=artist, title=title, diffname=version)
    print("Done.")

# @task
def fetch_old_scores(offset, limit):
    cur = conn.cursor()
    cur.execute("select id, user_id, beatmap_id, total_score, data from scores limit %s offset %s", (limit, offset))
    return cur.fetchall()

# @task
def insert_new_scores(data):
    with db_session:
        for id, user_id, beatmap_id, total_score, metadata_s in data:
            s = Score.get(id=id)
            if s is not None: continue
            metadata = json.loads(metadata_s)
            mods = metadata.get('mods', [])
            mods_l = []
            for mod in mods:
                mods_l.append(mod['acronym'])
            mods_l.sort()
            mods_s = "|".join(mods_l)
            user = User.get(id=user_id)
            beatmap = Beatmap.get(id=beatmap_id)
            Score(id=id, user=user, beatmap=beatmap, score=total_score, score_pp=1.0, mods=mods_s)


# @flow(log_prints=True)
def import_scores():
    cur = conn.cursor()
    cur.execute(f"select count(*) from scores")
    total: int = cur.fetchone()[0]
    num_batches = ceil(total / BATCH_SIZE)

    for i in tqdm(range(num_batches)):
        offset = i * BATCH_SIZE
        rows = fetch_old_scores(offset, BATCH_SIZE)
        insert_new_scores(rows)

# import_users()
# import_beatmaps()
import_scores()
