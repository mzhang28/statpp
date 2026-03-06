import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

OUTPUT = "./training-data"
MODS = "DT,HD"
MIN_PLAYS = 100
TOP_K = 50
N_ITERS = 200

df = pd.read_parquet(f"{OUTPUT}/scores.parquet")
df = df[(df["mods"] == MODS) & (df["pp"] > 0)]
counts = df.groupby("map_idx").size()
keep = set(counts[counts >= MIN_PLAYS].index)
df = df[df["map_idx"].isin(keep)].copy()

uids, u_inv = np.unique(df["user_idx"].values, return_inverse=True)
mids, m_inv = np.unique(df["map_idx"].values, return_inverse=True)
n_users, n_maps = len(uids), len(mids)
print(f"{n_users} users, {n_maps} maps, {len(df)} scores")

log_pp = np.log(df["pp"].values).astype(np.float64)

# per-map: arrays of (local_user_idx, log_pp, play_index)
map_plays = [[] for _ in range(n_maps)]
for k, (mi, ui, lp) in enumerate(zip(m_inv, u_inv, log_pp)):
    map_plays[mi].append((ui, lp, k))
map_plays = [(np.array([x[0] for x in lst], dtype=np.int32),
              np.array([x[1] for x in lst], dtype=np.float64),
              np.array([x[2] for x in lst], dtype=np.int64)) for lst in map_plays]

# per-user: arrays of (local_map_idx, play_index)
user_plays = [[] for _ in range(n_users)]
for k, (ui, mi) in enumerate(zip(u_inv, m_inv)):
    user_plays[ui].append((mi, k))
user_plays = [(np.array([x[0] for x in lst], dtype=np.int32),
               np.array([x[1] for x in lst], dtype=np.int64)) for lst in user_plays]

skill = np.ones(n_users, dtype=np.float64)
logsurv = np.zeros(len(df), dtype=np.float64)
map_mu = np.zeros(n_maps, dtype=np.float64)
map_sigma = np.zeros(n_maps, dtype=np.float64)
map_q = np.zeros(n_maps, dtype=np.float64)

user_lookup = df[["user_idx", "user_id"]
                 ].drop_duplicates().set_index("user_idx")
if "username" in df.columns:
    name_lookup = df[["user_id", "username"]
                     ].drop_duplicates().set_index("user_id")
    def get_name(
        uid): return name_lookup.loc[uid, "username"] if uid in name_lookup.index else str(uid)
else:
    def get_name(uid): return str(uid)

uid_to_local = {uid: i for i, uid in enumerate(uids)}
KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}

pbar = tqdm(range(N_ITERS), unit="iter")
for it in pbar:
    prev_skill = skill.copy()

    # A: map quality
    for j in range(n_maps):
        u_idxs, _, _ = map_plays[j]
        map_q[j] = skill[u_idxs].mean()

    # B: weighted distribution fit
    for j in range(n_maps):
        u_idxs, lps, _ = map_plays[j]
        w = skill[u_idxs]
        wsum = w.sum()
        mu = (w * lps).sum() / wsum
        var = (w * (lps - mu) ** 2).sum() / wsum
        map_mu[j] = mu
        map_sigma[j] = max(np.sqrt(var), 0.01)

    # C: log-survival per play
    for j in range(n_maps):
        _, lps, pidxs = map_plays[j]
        z = (lps - map_mu[j]) / map_sigma[j]
        logsurv[pidxs] = -norm.logsf(z)

    # D: aggregate per player
    for i in range(n_users):
        m_idxs, pidxs = user_plays[i]
        ls = logsurv[pidxs]
        q = map_q[m_idxs]
        if len(ls) > TOP_K:
            top = np.argpartition(-ls, TOP_K)[:TOP_K]
            ls, q = ls[top], q[top]
        skill[i] = (q * ls).sum() / q.sum()

    # E: normalize
    skill /= skill.mean()

    delta = np.abs(skill - prev_skill).max()

    known_str = []
    for uid, name in KNOWN.items():
        matches = user_lookup[user_lookup["user_id"] == uid]
        if matches.empty or matches.index[0] not in uid_to_local:
            known_str.append(f"{name}=N/A")
            continue
        li = uid_to_local[matches.index[0]]
        r = int((skill > skill[li]).sum()) + 1
        known_str.append(f"{name}=#{r}({skill[li]:.2f})")

    pbar.set_postfix_str(f"delta={delta:.4f} | {' '.join(known_str)}")

# final output
ranking = np.argsort(-skill)
print("\ntop 30:")
for r, li in enumerate(ranking[:30]):
    uid = int(user_lookup.loc[uids[li], "user_id"]
              ) if uids[li] in user_lookup.index else -1
    print(f"  #{r+1} {get_name(uid)} skill={skill[li]:.4f}")

print("\nknown players:")
for uid, name in KNOWN.items():
    matches = user_lookup[user_lookup["user_id"] == uid]
    if matches.empty or matches.index[0] not in uid_to_local:
        print(f"  {name}: not in pool")
        continue
    li = uid_to_local[matches.index[0]]
    r = int((skill > skill[li]).sum()) + 1
    print(f"  {name}: #{r}/{n_users} skill={skill[li]:.4f}")

np.savez_compressed(
    f"{OUTPUT}/iterative_hddt.npz",
    skill=skill, map_mu=map_mu, map_sigma=map_sigma, map_q=map_q,
    logsurv=logsurv, uids=uids, mids=mids,
)
print("\nsaved iterative_hddt.npz")
