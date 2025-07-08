import json
from pony.orm import db_session, select
from db import Score, User, Beatmap, db
import random

db.generate_mapping(create_tables=True)

def sample_connected_subgraph(target_score_count=1000, initial_user_count=30):
    with db_session:
        # 1. Pick initial random users
        initial_users = select(u for u in User).order_by(lambda: random.random())[:initial_user_count]
        user_ids = set(u.id for u in initial_users)

        beatmap_ids = set()
        score_count = 0

        prev_user_count = 0
        prev_beatmap_count = 0

        while score_count < target_score_count:
            # 2. Find beatmaps played by current users
            new_beatmaps = select(s.beatmap.id for s in Score if s.user.id in user_ids)
            new_beatmap_ids = set(new_beatmaps) - beatmap_ids
            if not new_beatmap_ids:
                break
            beatmap_ids.update(new_beatmap_ids)

            # 3. Find all users who played those beatmaps
            new_users = select(s.user.id for s in Score if s.beatmap.id in beatmap_ids)
            new_user_ids = set(new_users) - user_ids
            user_ids.update(new_user_ids)

            # 4. Count total scores in this subgraph
            score_count = select(s for s in Score if s.user.id in user_ids and s.beatmap.id in beatmap_ids).count()

            # Stop if no growth or enough scores
            if (len(user_ids) == prev_user_count and len(beatmap_ids) == prev_beatmap_count) or score_count >= target_score_count:
                break

            prev_user_count = len(user_ids)
            prev_beatmap_count = len(beatmap_ids)

        return list(user_ids), list(beatmap_ids), score_count

# Usage example:
with db_session:
    users_sample, beatmaps_sample, num_scores = sample_connected_subgraph()
    with open("sample.json", "w") as f:
        json.dump(dict(users=users_sample, beatmaps=beatmaps_sample), f)
    print(f"Sampled {len(users_sample)} users, {len(beatmaps_sample)} beatmaps, {num_scores} scores")
