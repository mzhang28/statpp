from pony.orm import db_session, desc, select
from math import exp, log, tanh, sqrt
from prefect import task
from tqdm import tqdm
from db import Beatmap, BeatmapMod, Score, User, conn, db
from scipy.special import expit
import random
import json
from tunable import collect_beatmap_score_influences, compute_beatmap_individual_score_influence, expected_performance, score_to_pp

db.generate_mapping(create_tables=False, check_tables=True)

# SAMPLE RELATED
with open("sample3.json") as f:
    sample = json.load(f)
user_ids = sample["users"]
beatmapmod_ids = sample["beatmapmods"]

def update_all_user_pp():
    with db_session:
        print(f"Updating user pp for {len(user_ids)} users...")
        db.execute(f"""
            UPDATE User u
            JOIN (
                SELECT
                    rs.user_id,
                    SUM(POWER(0.9, rs.rn) * rs.score_pp) AS total_pp
                FROM (
                    SELECT
                        s.user AS user_id,
                        s.score_pp,
                        @rownum := IF(@prev_user = s.user, @rownum + 1, 0) AS rn,
                        @prev_user := s.user
                    FROM (
                        SELECT *
                        FROM Score s
                        WHERE s.user IN ({", ".join(str(u) for u in user_ids)})
                        ORDER BY s.user, s.score_pp DESC
                    ) s
                    CROSS JOIN (SELECT @rownum := -1, @prev_user := NULL) vars
                ) rs
                GROUP BY rs.user_id
            ) us ON us.user_id = u.id
            SET u.total_pp = COALESCE(us.total_pp, 0)
        """)

def update_beatmap_mod_difficulty(bmid: int, max_pp: float, max_diff: float):
    with db_session:
        bm = BeatmapMod.get(id=bmid)
        norm_diff = bm.difficulty / max_diff
        if len(bm.scores) == 0: return
        # print(f"Crunching {len(b.scores)} scores...")
        accum = dict(count=0, sum_log_expected=0, sum_log_actual=0)
        for score in bm.scores:
            if score.score == 0: continue
            compute_beatmap_individual_score_influence(accum, bm, score)
            # player_pp = score.user.total_pp / max_pp
            # points_norm = normalize_score(score.score, bm.mod)
            # player_weight = 1.0 / (1 + exp(-8.0 * (player_pp - 0.7)))
            # combined_weight = player_weight * beatmap_weight

            # expected_score = expected_performance(player_pp, norm_diff)
            # expected_score = max(expected_score, 1e-8)  # avoid log(0)

            # weighted_log_expected += combined_weight * log(expected_score)

            # if player_pp <= norm_diff:
            #     adjusted_actual = points_norm
            # else:
            #     penalty = exp(-5.0 * (player_pp - norm_diff) * points_norm ** 5.0)
            #     adjusted_actual = points_norm * penalty
            # weighted_log_actual += combined_weight * log(adjusted_actual)
            # weight_sum += combined_weight

        # if weight_sum == 0: return
        # avg_log_expected = weighted_log_expected / weight_sum
        # avg_log_actual = weighted_log_actual / weight_sum
        # if bm.beatmap.id == 1173116:
        #     print(f"Shiet: {norm_diff = }, {beatmap_weight = }, {avg_log_expected = }, {avg_log_actual = }")
        #     print(f"Expected for 1.0, {expected_performance(1.0, norm_diff) = }, for 0.0, {expected_performance(0.0, norm_diff) = }")

        # adjustment_log = avg_log_expected - avg_log_actual
        # adjustment_log = max(min(adjustment_log, 0.1), -0.1) # Clamp the adjustment so this is no more than 0.1 on either side
        # adjustment_ratio = exp(adjustment_log)

        if bm.beatmap.id == 1173116: print(accum)
        adjustment_ratio = collect_beatmap_score_influences(accum, bm)
        training_alpha = 0.05
        bm.difficulty = (1 - training_alpha) * bm.difficulty + training_alpha * (norm_diff * adjustment_ratio * max_diff)

def update_all_beatmaps_difficulty(max_diff:float, max_pp:float, all_beatmap_ids=None):
    with db_session:
        if all_beatmap_ids is None:
            all_beatmap_ids = list(select(bm.id for bm in BeatmapMod)[:])
    random.shuffle(all_beatmap_ids)
    print(f"Computing {len(all_beatmap_ids)} beatmaps...")
    for id in tqdm(all_beatmap_ids):
        update_beatmap_mod_difficulty(id, max_pp=max_pp, max_diff=max_diff)

def normalize_score(score_i: int, mods: str) -> float:
    score = score_i * 1.0
    mods_l = mods.split("|")
    for mod in mods_l:
        if mod == "CL": score /= 0.96
        if mod == "DT": score /= 1.2
        if mod == "HR": score /= 1.1
        if mod == "HD": score /= 1.1
        if mod == "EZ": score /= 0.5
    return score / 1_000_000.0

def compute_score_pp(score_id: int, max_pp: float, scale):
    with db_session:
        s = Score.get(id=score_id)
        points_norm = normalize_score(s.score, s.beatmap_mod.mod)
        norm_diff = s.beatmap_mod.difficulty / scale
        norm_pp = s.user.total_pp / max_pp
        score_pp = score_to_pp(points_norm, norm_pp, norm_diff)
        s.score_pp = score_pp

def compute_all_score_pp(max_diff: float, max_pp: float, all_beatmap_ids=None, all_player_ids=None):
    with db_session:
        if all_player_ids is None:
                all_player_ids = list(select(u.id for u in User)[:])
        player_id_set = set(all_player_ids)
        if all_beatmap_ids is None:
            all_beatmap_ids = list(select(bm.id for bm in BeatmapMod)[:])
        beatmapmod_id_set = set(all_beatmap_ids)
        all_score_ids = []
        with db_session:
            scores = select(s for s in Score if s.beatmap_mod.id in beatmapmod_id_set and s.user.id in player_id_set)[:]
            for s in scores:
                all_score_ids.append(s.id)
    random.shuffle(all_score_ids)
    print(f"Computing {len(all_score_ids)} scores...")
    for id in tqdm(all_score_ids):
        compute_score_pp(id, max_pp=max_pp, scale=max_diff)

def compute_all_beatmap_success_rate(beatmapmod_ids=None):
    with db_session:
        if beatmapmod_ids is None:
            beatmapmod_ids = list(select(bm.id for bm in BeatmapMod)[:])

        beatmap_success_data = {}
        scores_to_process = select(s for s in Score if s.beatmap_mod.id in beatmapmod_ids)[:]

        print(f"Computing success rates for {len(beatmapmod_ids)} beatmap mods based on {len(scores_to_process)} scores...")

        for s in tqdm(scores_to_process, desc="Calculating score contributions to success rate"):
            bm_id = s.beatmap_mod.id
            if bm_id not in beatmap_success_data:
                beatmap_success_data[bm_id] = {'n_scores': 0, 'pass_sum': 0.0}

            points_norm = normalize_score(s.score, s.beatmap_mod.mod)
            points_norm = max(min(points_norm, 1.0), 0.0)
            pass_contribution = 1.0 / (1.0 + exp(-20 * (points_norm - 0.8)))

            beatmap_success_data[bm_id]['n_scores'] += 1
            beatmap_success_data[bm_id]['pass_sum'] += pass_contribution

        print("Updating beatmap mod success rates in database...")
        for bm_id, data in tqdm(beatmap_success_data.items(), desc="Updating database"):
            bm = BeatmapMod.get(id=bm_id)
            if bm:
                n_scores = data['n_scores']
                pass_sum = data['pass_sum']

                if n_scores == 0:
                    bm.success_rate = 0.0
                else:
                    success_rate_raw = (pass_sum / n_scores) / 0.7
                    bm.success_rate = sqrt(min(1.0, success_rate_raw))

max_diff = 8
max_pp = 300
while True:
    print("New loop ------------------------")
    compute_all_beatmap_success_rate(beatmapmod_ids=beatmapmod_ids)
    update_all_beatmaps_difficulty(max_diff, max_pp, beatmapmod_ids)
    compute_all_score_pp(max_diff, max_pp, beatmapmod_ids, user_ids)
    update_all_user_pp()
