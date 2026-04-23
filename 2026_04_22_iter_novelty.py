import math
import numpy as np
import pandas as pd
import polars as pl
import torch
import sqlite3
import json
import subprocess
from datetime import datetime
from osu_shit import get_beatmap_info
from torch import nn
from tqdm import tqdm

OUTPUT = "./training-data"
MIN_PLAYS = 30
N_FACTORS = 10
OUTER_ITERS = 15
INNER_EPOCHS = 5
INIT_EPOCHS = 20
LR_INIT = 0.005
LR_REFINE = 0.001
REG = 0.02
BATCH = 65536
W_CLAMP = 100.0
ALPHA_COV = 1.0
MAX_PLAYS_NOVELTY = 200
NOVELTY_INTERVAL = 3
SHRINKAGE_N0 = 50.0
RELEVANCE_TEMP = 5.0
TOP_K = 50
SQRT2 = math.sqrt(2.0)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    raise Exception("not running this shit on cpu")


# load
print("loading")
df = (
    pl.scan_parquet(f"{OUTPUT}/scores.parquet")
    .filter(pl.col("pp") > 0)
    .collect()
)
counts = df.group_by("map_idx").len()
keep = set(counts.filter(pl.col("len") >= MIN_PLAYS)["map_idx"].to_list())
df = df.filter(pl.col("map_idx").is_in(keep))

mod_stats = df.group_by("mods").agg(
    pl.col("map_idx").n_unique().alias("n_items"),
    pl.len().alias("n_plays"),
)
print("items per mod pool:")
for row in mod_stats.sort("n_plays", descending=True).iter_rows(named=True):
    print(
        f"  {row['mods'] or 'nomod':<12} {row['n_items']:>7} items  {row['n_plays']:>10} plays")

df = df.to_pandas()

uids, u_inv = np.unique(df["user_idx"].values, return_inverse=True)
mids, m_inv = np.unique(df["map_idx"].values, return_inverse=True)
n_users, n_maps, n_plays = len(uids), len(mids), len(df)
print(f"\n{n_users} users, {n_maps} items, {n_plays} plays, device={DEVICE}")

u_t = torch.tensor(u_inv, dtype=torch.long, device=DEVICE)
m_t = torch.tensor(m_inv, dtype=torch.long, device=DEVICE)
log_pp = torch.log(torch.tensor(
    df["pp"].values, dtype=torch.float32, device=DEVICE))

map_counts = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
map_counts.scatter_add_(0, m_t, torch.ones(
    n_plays, dtype=torch.float32, device=DEVICE))

user_counts = torch.zeros(n_users, dtype=torch.float32, device=DEVICE)
user_counts.scatter_add_(0, u_t, torch.ones(
    n_plays, dtype=torch.float32, device=DEVICE))

# per-user play lists (vectorized)
print("building per-user play lists")
sort_order = np.argsort(u_inv)
u_inv_sorted = u_inv[sort_order]
boundaries = np.searchsorted(u_inv_sorted, np.arange(n_users + 1))
user_play_indices = [sort_order[boundaries[i]:boundaries[i + 1]]
                     for i in range(n_users)]

# model
bu = nn.Embedding(n_users, 1).to(DEVICE).float()
bi = nn.Embedding(n_maps, 1).to(DEVICE).float()
pu = nn.Embedding(n_users, N_FACTORS).to(DEVICE).float()
qi = nn.Embedding(n_maps, N_FACTORS).to(DEVICE).float()
nn.init.zeros_(bu.weight)
nn.init.zeros_(bi.weight)
nn.init.normal_(pu.weight, 0, 0.1)
nn.init.normal_(qi.weight, 0, 0.1)
all_params = list(bu.parameters()) + list(bi.parameters()) + \
    list(pu.parameters()) + list(qi.parameters())

# lookups
user_info = df[["user_idx", "user_id", "username"]
               ].drop_duplicates().set_index("user_idx")
uid_to_local = {uid: i for i, uid in enumerate(uids)}


def get_name(li):
    uidx = uids[li]
    if uidx in user_info.index:
        row = user_info.loc[uidx]
        return row["username"] if pd.notna(row["username"]) else str(int(row["user_id"]))
    return "?"


def get_user_id(li):
    uidx = uids[li]
    if uidx in user_info.index:
        row = user_info.loc[uidx]
        return int(row["user_id"])
    return 0


KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
known_locals = {}
for uid, name in KNOWN.items():
    matches = user_info[user_info["user_id"] == uid]
    if not matches.empty and matches.index[0] in uid_to_local:
        known_locals[name] = uid_to_local[matches.index[0]]


def compute_novelty_weights(logsurv_np, qi_np, m_inv_np):
    novelty = np.ones(n_plays, dtype=np.float32)
    for i in tqdm(range(n_users), unit="user", desc="novelty", leave=False):
        pidxs = user_play_indices[i]
        if len(pidxs) == 0:
            continue
        ls = logsurv_np[pidxs]
        order = np.argsort(-ls)
        if len(order) > MAX_PLAYS_NOVELTY:
            order = order[:MAX_PLAYS_NOVELTY]

        global_idxs = pidxs[order]
        phi_mat = qi_np[m_inv_np[global_idxs]]          # (K, N_FACTORS)
        lk_vec = ls[order]                               # (K,)

        # exclusive prefix sum: c[rank] = sum_{j<rank} ALPHA_COV * lk_j * phi_j
        weighted = (ALPHA_COV * lk_vec[:, None] * phi_mat).astype(np.float32)
        c_mat = np.empty_like(weighted)
        c_mat[0] = 0.0
        np.cumsum(weighted[:-1], axis=0, out=c_mat[1:])

        c_norm = np.linalg.norm(c_mat, axis=1)          # (K,)
        phi_norm = np.linalg.norm(phi_mat, axis=1)      # (K,)

        nov = np.ones(len(order), dtype=np.float32)
        valid = (c_norm >= 1e-12) & (phi_norm >= 1e-12)
        if valid.any():
            dots = (c_mat[valid] * phi_mat[valid]).sum(axis=1)
            cos = np.clip(dots / (c_norm[valid] * phi_norm[valid]), -1.0, 1.0)
            nov[valid] = (1.0 - cos) / 2.0

        novelty[global_idxs] = nov
    return novelty


# volume-normalized weights for SGD
vol_inv = 1.0 / user_counts[u_t]
vol_inv = vol_inv / vol_inv.mean()


def run_sgd(targets, n_epochs, lr, weights=None):
    opt = torch.optim.SGD(all_params, lr=lr, weight_decay=REG)
    mu = targets.mean().detach()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_plays, device=DEVICE)
        total_loss = 0.0
        for start in range(0, n_plays, BATCH):
            idx = perm[start:start + BATCH]
            u_b, m_b, r_b = u_t[idx], m_t[idx], targets[idx]
            pred = mu + bu(u_b).squeeze() + bi(m_b).squeeze() + \
                (pu(u_b) * qi(m_b)).sum(dim=1)
            err = (pred - r_b) ** 2
            w = vol_inv[idx]
            if weights is not None:
                w = w * weights[idx]
            w = w / w.mean()
            loss = (w * err).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
    return mu, total_loss / n_plays


def print_rankings(label):
    bu_np = bu.weight.detach().cpu().numpy().flatten()
    ranking = np.argsort(-bu_np)
    print(f"\n{label} — top 10 by user bias:")
    for r, li in enumerate(ranking[:10]):
        print(f"  #{r+1} {get_name(li)} bu={bu_np[li]:.4f}")
    print("known players:")
    for name, li in known_locals.items():
        r = int((bu_np > bu_np[li]).sum()) + 1
        print(f"  {name}: #{r}/{n_users} bu={bu_np[li]:.4f}")


# step 0: initial SVD on log(pp)
print("\nstep 0: initial SVD on log(pp)")
mu_global = log_pp.mean().detach()
pbar = tqdm(range(INIT_EPOCHS), unit="epoch", desc="init")
opt = torch.optim.SGD(all_params, lr=LR_INIT, weight_decay=REG)
for epoch in pbar:
    perm = torch.randperm(n_plays, device=DEVICE)
    total_loss = 0.0
    for start in range(0, n_plays, BATCH):
        idx = perm[start:start + BATCH]
        u_b, m_b, r_b = u_t[idx], m_t[idx], log_pp[idx]
        pred = mu_global + bu(u_b).squeeze() + \
            bi(m_b).squeeze() + (pu(u_b) * qi(m_b)).sum(dim=1)
        w = vol_inv[idx]
        w = w / w.mean()
        loss = (w * (pred - r_b) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    pbar.set_postfix(mse=f"{total_loss / n_plays:.6f}")

print_rankings("after init")

# outer loop
logsurv = None
novelty_t = torch.ones(n_plays, dtype=torch.float32, device=DEVICE)
map_sigma = torch.ones(n_maps, dtype=torch.float32, device=DEVICE)
m_inv_np = m_inv

pbar = tqdm(range(OUTER_ITERS), unit="outer")
for outer in pbar:
    with torch.no_grad():
        # A: relevance weights (sharpened)
        w = (RELEVANCE_TEMP * (pu(u_t) * qi(m_t)).sum(dim=1)
             ).exp().clamp(max=W_CLAMP)

        # B: weighted sums per map (full, not leave-one-out yet)
        w_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
        w_sum.scatter_add_(0, m_t, w)
        wx_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
        wx_sum.scatter_add_(0, m_t, w * log_pp)
        wxx_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
        wxx_sum.scatter_add_(0, m_t, w * log_pp * log_pp)

        # full-pool versions (kept for saving / logging / map_q computation)
        map_mu = wx_sum / w_sum
        map_var_full = wxx_sum / w_sum - map_mu ** 2
        map_sigma = map_var_full.clamp(min=1e-6).sqrt().clamp(min=0.01)

        # C: leave-one-out mu, sigma per play
        w_k = w
        # Heavy weights guard: cap per-play weight at 50% of map sum to prevent LOO blowup
        w_k = torch.min(w_k, 0.5 * w_sum[m_t])

        x_k = log_pp
        denom_loo = (w_sum[m_t] - w_k).clamp(min=1e-6)
        mu_loo = (wx_sum[m_t] - w_k * x_k) / denom_loo
        var_loo = (wxx_sum[m_t] - w_k * x_k * x_k) / denom_loo - mu_loo ** 2
        sigma_loo = var_loo.clamp(min=1e-6).sqrt().clamp(min=0.01)

        # z-score and logsurv using LOO moments
        resid_loo = x_k - mu_loo
        z = resid_loo / sigma_loo
        erfc_val = torch.erfc(z / SQRT2).clamp(min=1e-30)
        new_logsurv = -(erfc_val / 2.0).log()

        # confidence shrinkage
        alpha = map_counts[m_t] / (map_counts[m_t] + SHRINKAGE_N0)
        global_mean_ls = new_logsurv.mean()
        new_logsurv = alpha * new_logsurv + (1.0 - alpha) * global_mean_ls

        if logsurv is not None:
            delta = (new_logsurv - logsurv).abs().max().item()
        else:
            delta = float("inf")
        logsurv = new_logsurv

    # novelty weights (recompute every NOVELTY_INTERVAL iterations)
    if outer % NOVELTY_INTERVAL == 0:
        logsurv_np = logsurv.cpu().numpy()
        qi_np = qi.weight.detach().cpu().numpy()
        novelty_np = compute_novelty_weights(logsurv_np, qi_np, m_inv_np)
        novelty_t = torch.tensor(
            novelty_np, dtype=torch.float32, device=DEVICE)

    # combined weights: sigma * novelty, normalized to mean 1
    play_w = map_sigma[m_t] * novelty_t
    play_w = play_w / play_w.mean()

    # D: weighted SGD refit
    mu_ls, final_mse = run_sgd(
        logsurv, INNER_EPOCHS, LR_REFINE, weights=play_w)

    sigma_np = map_sigma.cpu().numpy()

    known_str = []
    bu_np = bu.weight.detach().cpu().numpy().flatten()
    for name, li in known_locals.items():
        r = int((bu_np > bu_np[li]).sum()) + 1
        known_str.append(f"{name}=#{r}({bu_np[li]:.2f})")

    nov_mean = novelty_t.mean().item()
    pbar.set_postfix_str(
        f"d={delta:.2f} mse={final_mse:.4f} σ={sigma_np.mean():.3f} nov={nov_mean:.3f} z_max={z.max().item():.1f} | {' '.join(known_str)}"
    )

print_rankings("final")

# --- post-hoc analysis ---
bu_np = bu.weight.detach().cpu().numpy().flatten()
pu_np = pu.weight.detach().cpu().numpy()
qi_np = qi.weight.detach().cpu().numpy()

# 4. Mahalanobis ranking
print("\n--- Mahalanobis ranking ---")
Sigma = np.cov(pu_np.T)
Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(N_FACTORS))
mahal_scores = np.einsum('ij,jk,ik->i', pu_np, Sigma_inv, pu_np)
mahal_sqrt = np.sqrt(mahal_scores)
bu_z = (bu_np - bu_np.mean()) / (bu_np.std() + 1e-9)
mahal_z = (mahal_sqrt - mahal_sqrt.mean()) / (mahal_sqrt.std() + 1e-9)
combined = bu_z + mahal_z
ranking_m = np.argsort(-combined)

print("top 10 by bu + mahal:")
for r, li in enumerate(ranking_m[:10]):
    print(
        f"  #{r+1} {get_name(li)} combined={combined[li]:.4f} bu={bu_np[li]:.4f} mahal={mahal_scores[li]:.4f}")
print("known players:")
for name, li in known_locals.items():
    r = int((combined > combined[li]).sum()) + 1
    print(
        f"  {name}: #{r}/{n_users} combined={combined[li]:.4f} bu={bu_np[li]:.4f} mahal={mahal_scores[li]:.4f}")

# 5. Dimension inspection via PCA of user factors
print("\n--- Dimension inspection ---")
eigvals, eigvecs = np.linalg.eigh(Sigma)
order = np.argsort(-eigvals)
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

map_info = df[["map_idx", "beatmap_id", "mods"]
              ].drop_duplicates().set_index("map_idx")
mid_to_local = {mid: i for i, mid in enumerate(mids)}

for d in range(min(5, N_FACTORS)):
    ev = eigvecs[:, d]
    map_proj = qi_np @ ev
    top_maps = np.argsort(-map_proj)[:10]
    bot_maps = np.argsort(map_proj)[:10]
    print(f"\neigenvector {d} (variance={eigvals[d]:.4f}):")
    print("  top maps:", end="")
    for mi in top_maps:
        midx = mids[mi]
        bid = int(map_info.loc[midx, "beatmap_id"]
                  ) if midx in map_info.index else "?"
        mods = map_info.loc[midx, "mods"] if midx in map_info.index else ""
        print(f" {bid}+{mods}({map_proj[mi]:.3f})", end="")
    print()
    print("  bot maps:", end="")
    for mi in bot_maps:
        midx = mids[mi]
        bid = int(map_info.loc[midx, "beatmap_id"]
                  ) if midx in map_info.index else "?"
        mods = map_info.loc[midx, "mods"] if midx in map_info.index else ""
        print(f" {bid}+{mods}({map_proj[mi]:.3f})", end="")
    print()

print("\nknown player projections onto eigenvectors:")
header = f"{'name':<12}" + \
    "".join(f"{'pc'+str(d):>8}" for d in range(min(5, N_FACTORS)))
print(header)
for name, li in known_locals.items():
    proj = pu_np[li] @ eigvecs[:, :5]
    vals = "".join(f"{p:>8.3f}" for p in proj)
    print(f"{name:<12}{vals}")

np.savez_compressed(
    f"{OUTPUT}/itersvd_novelty.npz",
    bu=bu_np, bi=bi.weight.detach().cpu().numpy().flatten(),
    pu=pu_np, qi=qi_np,
    map_mu=map_mu.cpu().numpy(), map_sigma=map_sigma.cpu().numpy(),
    uids=uids, mids=mids,
)
print(f"\nsaved {OUTPUT}/itersvd_novelty.npz")

# --- Additional analysis for DB ---
print("\n--- calculating map quality and player skill ---")
logsurv_np = logsurv.cpu().numpy()
map_mu_np = map_mu.cpu().numpy()
map_sigma_np = map_sigma.cpu().numpy()

df["local_u"] = u_inv
df["local_m"] = m_inv
df["logsurv"] = logsurv_np

# map quality = mean bu of players who played each map
df["bu"] = bu_np[u_inv]
map_q = df.groupby("local_m")["bu"].mean().reindex(range(n_maps), fill_value=0.0).values
df["map_q"] = map_q[m_inv]
df["contribution"] = df["map_q"] * df["logsurv"]

# player skill: weighted mean of top-k contributions
player_skills = {}
for lu, group in tqdm(df.groupby("local_u"), desc="skill"):
    top = group.nlargest(TOP_K, "contribution")
    player_skills[lu] = top["contribution"].mean()
skill = np.zeros(n_users)
for lu, s in player_skills.items():
    skill[lu] = s

# --- SQLite output ---
print("\n--- exporting to statpp.db ---")
conn = sqlite3.connect("statpp.db")
c = conn.cursor()

c.execute("DROP TABLE IF EXISTS meta")
c.execute("DROP TABLE IF EXISTS players")
c.execute("DROP TABLE IF EXISTS beatmaps")
c.execute("DROP TABLE IF EXISTS scores")

c.execute("CREATE TABLE meta (key TEXT, value TEXT)")
c.execute("INSERT INTO meta VALUES (?, ?)", ("dimensions", str(N_FACTORS)))

# metadata for provenance
try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    git_diff = subprocess.check_output(["git", "diff", "HEAD"]).decode("utf-8")
except Exception as e:
    git_hash = "unknown"
    git_diff = str(e)

c.execute("INSERT INTO meta VALUES (?, ?)",
          ("timestamp", datetime.now().isoformat()))
c.execute("INSERT INTO meta VALUES (?, ?)", ("git_hash", git_hash))
c.execute("INSERT INTO meta VALUES (?, ?)", ("git_diff", git_diff))

c.execute(
    "CREATE TABLE players (id INTEGER PRIMARY KEY, username TEXT, metadata TEXT)")
player_rows = [
    (i, get_name(i), json.dumps({
        "user_id": get_user_id(i),
        "ratings": pu_np[i].tolist(),
        "bu": float(bu_np[i]),
        "skill": float(skill[i]),
        "combined": float(combined[i]),
        "mahal": float(mahal_scores[i]),
    }))
    for i in tqdm(range(n_users), desc="players")
]
c.executemany("INSERT INTO players VALUES (?, ?, ?)", player_rows)

c.execute("CREATE TABLE beatmaps (id INTEGER PRIMARY KEY, artist TEXT, title TEXT, version TEXT, metadata TEXT)")
bi_np = bi.weight.detach().cpu().numpy().flatten()
beatmap_rows = []
for i in tqdm(range(n_maps), desc="maps"):
    midx = mids[i]
    row = map_info.loc[midx] if midx in map_info.index else None
    bid = int(row["beatmap_id"]) if row is not None else 0
    mods_str = row["mods"] if row is not None else ""

    osu_path = f"source-data/2026_01_01_osu_files/{bid}.osu"
    try:
        info = get_beatmap_info(osu_path, mods_str)
        metadata = info.get("metadata", {})
        artist = metadata.get("artist", "")
        title = metadata.get("title", "")
        version = metadata.get("version", mods_str)
    except Exception:
        info = {}
        artist = ""
        title = str(bid)
        version = mods_str

    beatmap_rows.append((i, artist, title, version, json.dumps({
        "difficulties": qi_np[i].tolist(),
        "bi": float(bi_np[i]),
        "mu": float(map_mu_np[i]),
        "sigma": float(map_sigma_np[i]),
        "q": float(map_q[i]),
        "osu_id": bid,
        "mods": mods_str,
        "info": info,
    })))
c.executemany("INSERT INTO beatmaps VALUES (?, ?, ?, ?, ?)", beatmap_rows)


c.execute("CREATE TABLE scores (player_id INTEGER, beatmap_id INTEGER, mod TEXT, metadata TEXT)")
score_rows = []
for lu, group in tqdm(df.groupby("local_u"), desc="scores"):
    top = group.nlargest(TOP_K, "contribution")
    for _, row in top.iterrows():
        score_rows.append((int(lu), int(row["local_m"]), row["mods"], json.dumps({
            "scores": [float(row["logsurv"]), float(row["contribution"])],
            "accuracy": float(row["accuracy"]),
            "pp": float(row["pp"]),
        })))
c.executemany("INSERT INTO scores VALUES (?, ?, ?, ?)", score_rows)

conn.commit()
conn.close()
print("done: statpp.db generated")
