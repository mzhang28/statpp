from pony.orm import db_session, desc, select
from math import exp, log
from prefect import task
from tqdm import tqdm
from db import Beatmap, Score, User, conn, db
from scipy.special import expit
import random
import json

with open("sample.json") as f:
    sample = json.load(f)
user_ids = sample["users"]
beatmap_ids = sample["beatmaps"]

db.generate_mapping(create_tables=False, check_tables=True)

def update_all_user_pp():
    with db_session:
        # First, update their pp
        db.execute("""
            UPDATE User u
            SET total_pp = COALESCE((
                WITH RankedScores AS (
                    SELECT s.user, s.score_pp, ROW_NUMBER() OVER
                    (PARTITION BY s.user ORDER BY s.score_pp DESC) - 1 AS rn FROM Score s
                )
                SELECT SUM(POWER(0.95, rn) * score_pp)
                FROM RankedScores rs
                WHERE rs.user = u.id
            ), 0)
        """)

def expected_performance(player_rating, beatmap_difficulty):
    diff = player_rating - beatmap_difficulty
    slope = 3  # tuned for [0,1] scale; reduce if needed
    return 1 / (1 + exp(-slope * diff))

def performance_function(actual_score_norm, player_rating, beatmap_difficulty):
    expected = expected_performance(player_rating, beatmap_difficulty)
    expected = max(expected, 1e-8)  # avoid log(0)
    actual_score_norm = max(actual_score_norm, 1e-8)
    delta_log = log(actual_score_norm) - log(expected)
    return exp(delta_log)  # multiplier on difficulty for pp

def update_beatmap_difficulty(beatmap_id: int, scale: float):
    with db_session:
        b = Beatmap.get(id=beatmap_id)
        norm_diff = b.difficulty / scale
        if len(b.scores) == 0: return
        # print(f"Crunching {len(b.scores)} scores...")
        weighted_log_expected = 0.0
        weighted_log_actual = 0.0
        weight_sum = 0.0
        for score in b.scores:
            if score.score == 0: continue
            player_pp = score.user.normalized_pp
            points_norm = score.score / 1_000_000.0
            player_weight = expit(player_pp)
            expected_score = expected_performance(player_pp, norm_diff)
            expected_score = max(expected_score, 1e-8)  # avoid log(0)

            weighted_log_expected += player_weight * log(expected_score)
            weighted_log_actual += player_weight * log(points_norm)
            weight_sum += player_weight

        if weight_sum == 0: return
        avg_log_expected = weighted_log_expected / weight_sum
        avg_log_actual = weighted_log_actual / weight_sum

        adjustment_log = avg_log_expected - avg_log_actual
        adjustment_ratio = exp(adjustment_log)
        b.difficulty = norm_diff * adjustment_ratio * scale


# update_all_user_pp()

def update_all_beatmaps_difficulty(max_diff:float, all_beatmap_ids=None):
    with db_session:
        if all_beatmap_ids is None:
            all_beatmap_ids = list(select(u.id for u in Beatmap)[:])
    random.shuffle(all_beatmap_ids)
    print(f"Computing {len(all_beatmap_ids)} beatmaps...")
    for id in tqdm(all_beatmap_ids):
        update_beatmap_difficulty(id, scale=max_diff)

def compute_score_pp(score_id: int, max_pp: float, scale):
    with db_session:
        s = Score.get(id=score_id)
        points_norm = s.score / 1_000_000
        norm_diff = s.beatmap.difficulty / scale
        norm_pp = s.user.total_pp / max_pp
        score_pp = s.beatmap.difficulty * performance_function(points_norm, norm_pp, norm_diff)
        s.score_pp = score_pp


def compute_all_score_pp(max_diff: float, max_pp: float, all_beatmap_ids=None, all_player_ids=None):
    with db_session:
        if all_player_ids is None:
                all_player_ids = list(select(u.id for u in User)[:])
        player_id_set = set(all_player_ids)
        if all_beatmap_ids is None:
                all_beatmap_ids = list(select(b.id for b in Beatmap)[:])
        beatmap_id_set = set(all_beatmap_ids)
        all_score_ids = []
        with db_session:
            for s in Score.select(lambda s: s.beatmap.id in beatmap_id_set and s.user.id in player_id_set):
                all_score_ids.append(s.id)
    random.shuffle(all_score_ids)
    print(f"Computing {len(all_score_ids)} scores...")
    for id in tqdm(all_score_ids):
        compute_score_pp(id, max_pp=max_pp, scale=max_diff)

while True:
    print("New loop ------------------------")
    with db_session:
        max_diff: float = max(select(b.difficulty for b in Beatmap).max(), 1e-8)
        max_pp: float = max(select(u.total_pp for u in User).max(), 1e-8)
        print("Max diff:", max_diff)
    update_all_beatmaps_difficulty(max_diff, beatmap_ids)
    compute_all_score_pp(max_diff, max_pp, beatmap_ids, user_ids)
    update_all_user_pp()
