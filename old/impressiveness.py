import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from tqdm import tqdm

OUTPUT = "./training-data"
MODS = "DT,HD"
MIN_PLAYS = 100
TOP_K = 50
N_ITERS = 20

df = pd.read_parquet(f"{OUTPUT}/scores.parquet")
df = df[(df["mods"] == MODS) & (df["pp"] > 0)]

counts = df.groupby("map_idx").size()
keep = set(counts[counts >= MIN_PLAYS].index)
df = df[df["map_idx"].isin(keep)].copy()
print(f"{df['user_idx'].nunique()} users, {df['map_idx'].nunique()} maps, {len(df)} scores")

# fit log-normal per map (once, distributions don't change)
print("fitting log-normal per map")
fits = {}
for map_idx, group in tqdm(df.groupby("map_idx"), unit="map"):
    pp = group["pp"].values.astype(np.float64)
    try:
        shape, _, scale = lognorm.fit(pp, floc=0)
    except Exception:
        continue
    fits[map_idx] = {"shape": shape, "scale": scale, "n": len(pp)}

valid = df["map_idx"].isin(fits)
df = df[valid].copy()
print(f"{len(fits)} maps fitted")

# precompute impressiveness (static)
shape_arr = df["map_idx"].map(lambda m: fits[m]["shape"]).values
scale_arr = df["map_idx"].map(lambda m: fits[m]["scale"]).values
df["impressiveness"] = lognorm.sf(
    df["pp"].values, shape_arr, loc=0, scale=scale_arr)

# initial uniform map weights
map_idxs = np.array(list(fits.keys()))
map_weight = pd.Series(1.0, index=map_idxs)

# precompute per-user sorted impressiveness for top-k selection
print("precomputing per-user groups")
user_groups = {}
for uidx, group in df.groupby("user_idx"):
    user_groups[uidx] = group[["map_idx", "impressiveness"]].copy()

user_lookup = df[["user_idx", "user_id"]
                 ].drop_duplicates().set_index("user_idx")
if "username" in df.columns:
    name_lookup = df[["user_id", "username"]
                     ].drop_duplicates().set_index("user_id")
    def get_name(
        uid): return name_lookup.loc[uid, "username"] if uid in name_lookup.index else str(uid)
else:
    def get_name(uid): return str(uid)

KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}


def compute_skills(map_weight):
    skills = {}
    for uidx, g in user_groups.items():
        top = g.nlargest(TOP_K, "impressiveness")
        w = top["map_idx"].map(map_weight).fillna(1.0).values
        skills[uidx] = np.average(top["impressiveness"].values, weights=w)
    return pd.Series(skills)


# iterate
prev_skills = None
pbar = tqdm(range(N_ITERS), unit="iter")
for it in pbar:
    skills = compute_skills(map_weight)

    # convergence check
    if prev_skills is not None:
        delta = (skills - prev_skills.reindex(skills.index,
                 fill_value=0)).abs().mean()
    else:
        delta = float("inf")

    # reweight maps: mean skill of players who played each map
    player_skill_lookup = skills
    new_weights = {}
    for mid in map_idxs:
        sub = df[df["map_idx"] == mid]
        player_skills = sub["user_idx"].map(player_skill_lookup).dropna()
        if len(player_skills) == 0:
            new_weights[mid] = 1.0
        else:
            new_weights[mid] = player_skills.mean()
    map_weight = pd.Series(new_weights)

    # print known players
    ranked = skills.sort_values(ascending=False)
    known_str = []
    for uid, name in KNOWN.items():
        matches = user_lookup[user_lookup["user_id"] == uid]
        if matches.empty or matches.index[0] not in skills.index:
            known_str.append(f"{name}=N/A")
            continue
        uidx = matches.index[0]
        r = int((skills > skills[uidx]).sum()) + 1
        known_str.append(f"{name}=#{r}")

    pbar.set_postfix_str(f"delta={delta:.6f} | {' '.join(known_str)}")
    prev_skills = skills

# final output
skills = skills.sort_values(ascending=False)
print("\ntop 30:")
for r, (uidx, sk) in enumerate(skills.head(30).items()):
    uid = int(user_lookup.loc[uidx, "user_id"]
              ) if uidx in user_lookup.index else -1
    print(f"  #{r+1} {get_name(uid)} skill={sk:.4f}")

print("\nknown players (final):")
for uid, name in KNOWN.items():
    matches = user_lookup[user_lookup["user_id"] == uid]
    if matches.empty or matches.index[0] not in skills.index:
        print(f"  {name}: not in pool")
        continue
    uidx = matches.index[0]
    r = int((skills > skills[uidx]).sum()) + 1
    print(f"  {name}: #{r}/{len(skills)} skill={skills[uidx]:.4f}")

# convergence plot
skills.to_frame("skill").to_parquet(
    f"{OUTPUT}/impressiveness_iterated.parquet")
print("\nsaved impressiveness_iterated.parquet")
