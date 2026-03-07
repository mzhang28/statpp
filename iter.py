import polars as pl
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

OUTPUT = "./training-data"
MODS = "DT,HD"
MIN_PLAYS = 100
TOP_K = 50
N_ITERS = 100
ALPHA = 5.0
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


df = (
    pl.scan_parquet(f"{OUTPUT}/scores.parquet")
    .filter((pl.col("mods") == MODS) & (pl.col("pp") > 0))
    .collect()
)
counts = df.group_by("map_idx").len()
keep = set(counts.filter(pl.col("len") >= MIN_PLAYS)["map_idx"].to_list())
df = df.filter(pl.col("map_idx").is_in(keep)).to_pandas()

uids, u_inv = np.unique(df["user_idx"].values, return_inverse=True)
mids, m_inv = np.unique(df["map_idx"].values, return_inverse=True)
n_users, n_maps = len(uids), len(mids)
print(f"{n_users} users, {n_maps} maps, {len(df)} scores, device={DEVICE}")

u_inv_t = torch.tensor(u_inv, dtype=torch.long, device=DEVICE)
m_inv_t = torch.tensor(m_inv, dtype=torch.long, device=DEVICE)
log_pp_t = torch.log(torch.tensor(
    df["pp"].values, dtype=torch.float32, device=DEVICE))

# precompute per-map and per-user play counts
map_counts = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
map_counts.scatter_add_(0, m_inv_t, torch.ones(
    len(df), dtype=torch.float32, device=DEVICE))
user_counts = torch.zeros(n_users, dtype=torch.float32, device=DEVICE)
user_counts.scatter_add_(0, u_inv_t, torch.ones(
    len(df), dtype=torch.float32, device=DEVICE))

# precompute per-user play offsets for top-k
# sort plays by user, then we can slice per user
user_sort = torch.argsort(u_inv_t)
u_inv_sorted = u_inv_t[user_sort]
# boundaries per user
user_boundaries = torch.zeros(n_users + 1, dtype=torch.long, device=DEVICE)
user_boundaries[1:] = torch.cumsum(user_counts.long(), 0)

skill = torch.ones(n_users, dtype=torch.float32, device=DEVICE)

SQRT2 = math.sqrt(2.0)

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
    prev_skill = skill.clone()

    # A: map quality = mean skill per map
    skill_per_play = skill[u_inv_t]
    map_q = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
    map_q.scatter_add_(0, m_inv_t, skill_per_play)
    map_q /= map_counts

    # B: weighted mu and sigma per map (sharpened weights)
    w = skill_per_play ** ALPHA
    w_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
    w_sum.scatter_add_(0, m_inv_t, w)
    wlp_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
    wlp_sum.scatter_add_(0, m_inv_t, w * log_pp_t)
    map_mu = wlp_sum / w_sum

    resid = log_pp_t - map_mu[m_inv_t]
    wvar_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
    wvar_sum.scatter_add_(0, m_inv_t, w * resid ** 2)
    map_sigma = (wvar_sum / w_sum).sqrt().clamp(min=0.01)

    # C: log-survival = -log(sf(z)) = -log(erfc(z/sqrt2)/2)
    z = resid / map_sigma[m_inv_t]
    erfc_val = torch.erfc(z / SQRT2).clamp(min=1e-30)
    logsurv = -(erfc_val / 2.0).log()

    # D: top-k aggregation per user (vectorized via padding)
    logsurv_sorted = logsurv[user_sort]
    q_sorted = map_q[m_inv_t[user_sort]]

    max_plays = user_counts.max().long().item()
    k = min(TOP_K, max_plays)

    # pad into (n_users, max_plays)
    ls_pad = torch.full((n_users, max_plays), -1e30,
                        dtype=torch.float32, device=DEVICE)
    q_pad = torch.zeros((n_users, max_plays),
                        dtype=torch.float32, device=DEVICE)

    # fill via arange offsets
    play_idx = torch.arange(len(df), device=DEVICE)
    user_offsets = play_idx - user_boundaries[u_inv_sorted]
    ls_pad[u_inv_sorted, user_offsets] = logsurv_sorted
    q_pad[u_inv_sorted, user_offsets] = q_sorted

    top_ls, top_idx = torch.topk(ls_pad, k, dim=1)
    top_q = torch.gather(q_pad, 1, top_idx)

    # mask out padding
    mask = top_ls > -1e29
    top_ls = top_ls * mask
    top_q = top_q * mask

    q_sum = top_q.sum(dim=1).clamp(min=1e-30)
    new_skill = (top_q * top_ls).sum(dim=1) / q_sum

    skill = new_skill / new_skill.mean()

    delta = (skill - prev_skill).abs().max().item()

    known_str = []
    for uid, name in KNOWN.items():
        matches = user_lookup[user_lookup["user_id"] == uid]
        if matches.empty or matches.index[0] not in uid_to_local:
            known_str.append(f"{name}=N/A")
            continue
        li = uid_to_local[matches.index[0]]
        r = int((skill > skill[li]).sum().item()) + 1
        known_str.append(f"{name}=#{r}({skill[li]:.2f})")

    pbar.set_postfix_str(
        f"d={delta:.4f} range=[{skill.min():.2f},{skill.max():.2f}] std={skill.std():.3f} | {' '.join(known_str)}")

# final output
skill_np = skill.cpu().numpy()
ranking = np.argsort(-skill_np)

print("\ntop 30:")
for r, li in enumerate(ranking[:30]):
    uid = int(user_lookup.loc[uids[li], "user_id"]
              ) if uids[li] in user_lookup.index else -1
    print(f"  #{r+1} {get_name(uid)} skill={skill_np[li]:.4f}")

print("\nknown players:")
for uid, name in KNOWN.items():
    matches = user_lookup[user_lookup["user_id"] == uid]
    if matches.empty or matches.index[0] not in uid_to_local:
        print(f"  {name}: not in pool")
        continue
    li = uid_to_local[matches.index[0]]
    r = int((skill_np > skill_np[li]).sum()) + 1
    print(f"  {name}: #{r}/{n_users} skill={skill_np[li]:.4f}")

np.savez_compressed(
    f"{OUTPUT}/iterative_hddt.npz",
    skill=skill_np,
    map_mu=map_mu.cpu().numpy(), map_sigma=map_sigma.cpu().numpy(), map_q=map_q.cpu().numpy(),
    logsurv=logsurv.cpu().numpy(), uids=uids, mids=mids,
)
print("\nsaved iterative_hddt.npz")
