import json
import random
from pony.orm import count, db_session, select
from tqdm import tqdm
from db import BeatmapMod, Score, User, Beatmap, db
import click

db.generate_mapping(create_tables=True)

def sample_connected_subgraph(min_scores_per_user=20, min_scores_per_beatmapmod=20):
    # Get beatmapmods with their score counts using aggregation
    beatmapmod_data = select((bm, count(bm.scores)) for bm in BeatmapMod)[:]
    eligible_beatmapmods = [(bm, cnt) for bm, cnt in beatmapmod_data
                           if cnt >= min_scores_per_beatmapmod]
    print(f"Step 1 Done: {len(eligible_beatmapmods)} eligible beatmapmods")

    # Sort by score count descending
    eligible_beatmapmods.sort(key=lambda x: x[1], reverse=True)

    # Get users with their score counts using aggregation
    user_data = select((u, count(u.scores)) for u in User)[:]
    eligible_users = [u for u, cnt in user_data if cnt >= min_scores_per_user]
    print(f"Step 2 Done: {len(eligible_users)} eligible users")

    # Build connected subgraph
    sampled_users = set()
    sampled_beatmapmods_serial = set()
    sampled_beatmapmods = set()

    # Process beatmapmods in order of popularity
    for beatmapmod, _ in tqdm(random.sample(eligible_beatmapmods, 2000)):
        # Get all scores for this beatmapmod from eligible users in one query
        scores_for_bm = Score.select(lambda s: s.beatmap_mod == beatmapmod
                                     and s.user in eligible_users)[:]

        if scores_for_bm:
            # Add this beatmapmod and its beatmap
            sampled_beatmapmods.add(beatmapmod)
            sampled_beatmapmods_serial.add(beatmapmod.id)

            # Add all users who have scores on this beatmapmod
            for score in scores_for_bm:
                sampled_users.add(score.user.id)

    # Count total scores for sampled beatmapmods and users
    num_scores = count(s for s in Score
                      if s.beatmap_mod in sampled_beatmapmods
                      and s.user.id in sampled_users)

    sampled_users = list(sampled_users)
    sampled_users.sort()
    sampled_beatmapmods_serial = list(sampled_beatmapmods_serial)
    sampled_users.sort()
    return sampled_users, sampled_beatmapmods_serial, num_scores


@click.command()
@click.argument('filename',type=click.Path())
def main(filename):
    with db_session:
        users_sample, beatmapmods_sample, num_scores = sample_connected_subgraph()
        with open(filename, "w") as f:
            json.dump(dict(users=users_sample, beatmapmods=beatmapmods_sample), f)
        print(f"Sampled {len(users_sample)} users, {len(beatmapmods_sample)} beatmapmods, {num_scores} scores")

if __name__ == "__main__":
    main()
