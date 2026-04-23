import numpy as np
import pandas as pd

OUTPUT = "./training_data"

data = np.load(f"{OUTPUT}/pmf_hddt.npz", allow_pickle=True)
pu, qi = data["pu"], data["qi"]
bu, bi, mu = data["bu"], data["bi"], float(data["mu"])
uids, mids = data["uids"], data["mids"]
n_users, n_maps = len(uids), len(mids)

df = pd.read_parquet(f"{OUTPUT}/scores_indexed.parquet")
# df = df[df["mods"] == "DT,HD"]
play_counts = df.groupby("map_idx").size()

# factor-only variance per map
proj = pu @ qi.T  # (n_users, n_maps)
map_var = proj.var(axis=0)

# full predicted score per map (for player ranking later)
pred_full = mu + bu[:, None] + bi[None, :] + proj

results = pd.DataFrame({
    "map_idx": mids,
    "variance": map_var,
    "play_count": [play_counts.get(mid, 0) for mid in mids],
    "mean_bi": bi,
})

# map_idx -> beatmap_id
map_lookup = df[["map_idx", "beatmap_id", "mods"]
                ].drop_duplicates().set_index("map_idx")
results["beatmap_id"] = results["map_idx"].map(lambda m: int(
    map_lookup.loc[m, "beatmap_id"]) if m in map_lookup.index else -1)
results["mods"] = results["map_idx"].map(
    lambda m: map_lookup.loc[m, "mods"] if m in map_lookup.index else "?")

results = results.sort_values("variance", ascending=False)

print("top 20 discriminating maps (all):")
for _, row in results.head(20).iterrows():
    print(
        f"  beatmap_id={int(row['beatmap_id'])} mods={row['mods']} var={row['variance']:.6f} plays={int(row['play_count'])} bi={row['mean_bi']:.4f}")

print("\ntop 20 discriminating maps (plays <= 150):")
filtered = results[results["play_count"] <= 150]
for _, row in filtered.head(20).iterrows():
    print(
        f"  beatmap_id={int(row['beatmap_id'])} mods={row['mods']} var={row['variance']:.6f} plays={int(row['play_count'])} bi={row['mean_bi']:.4f}")

print("\ntop 20 discriminating maps (plays >= 500):")
filtered = results[results["play_count"] >= 500]
for _, row in filtered.head(20).iterrows():
    print(
        f"  beatmap_id={int(row['beatmap_id'])} mods={row['mods']} var={row['variance']:.6f} plays={int(row['play_count'])} bi={row['mean_bi']:.4f}")

# score known players on top-k discriminating maps
K = 100
top_k_local = [np.where(mids == row["map_idx"])[0][0]
               for _, row in filtered.head(K).iterrows()]
player_disc_score = pred_full[:, top_k_local].mean(axis=1)
ranking = np.argsort(-player_disc_score)

user_lookup = df[["user_idx", "user_id"]
                 ].drop_duplicates().set_index("user_idx")
uid_to_local = {uid: i for i, uid in enumerate(uids)}

print(
    f"\ntop 20 players by mean predicted score on top {K} discriminating maps:")
for r, li in enumerate(ranking[:20]):
    uid = int(user_lookup.loc[uids[li], "user_id"]
              ) if uids[li] in user_lookup.index else "?"
    print(f"  #{r+1} user_id={uid} score={player_disc_score[li]:.4f}")

KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
print(f"\nknown players (disc score, top {K}):")
for uid, name in KNOWN.items():
    matches = user_lookup[user_lookup["user_id"] == uid]
    if matches.empty or matches.index[0] not in uid_to_local:
        print(f"  {name}: not in pool")
        continue
    li = uid_to_local[matches.index[0]]
    r = int((player_disc_score > player_disc_score[li]).sum()) + 1
    print(f"  {name}: #{r}/{n_users} score={player_disc_score[li]:.4f}")

results.to_parquet(f"{OUTPUT}/map_discrimination.parquet")
print("\nsaved map_discrimination.parquet")
