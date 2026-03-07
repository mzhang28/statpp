import math
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from tqdm import tqdm

OUTPUT = "./training-data"
MODS = "DT,HD"
MIN_PLAYS = 100
N_FACTORS = 10
OUTER_ITERS = 50
INNER_EPOCHS = 5
INIT_EPOCHS = 20
LR_INIT = 0.005
LR_REFINE = 0.001
REG = 0.02
BATCH = 65536
W_CLAMP = 100.0
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
SQRT2 = math.sqrt(2.0)

# load and filter
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
n_users, n_maps, n_plays = len(uids), len(mids), len(df)
print(f"{n_users} users, {n_maps} maps, {n_plays} scores, device={DEVICE}")

u_t = torch.tensor(u_inv, dtype=torch.long, device=DEVICE)
m_t = torch.tensor(m_inv, dtype=torch.long, device=DEVICE)
log_pp = torch.log(torch.tensor(
    df["pp"].values, dtype=torch.float32, device=DEVICE))

map_counts = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
map_counts.scatter_add_(0, m_t, torch.ones(
    n_plays, dtype=torch.float32, device=DEVICE))

# SVD parameters
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

# lookups for printing
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
KNOWN = {7562902: "mrekk", 10549880: "ninerik",
         9269034: "accolibed",  9823042: "heilia"}


def run_sgd(targets, n_epochs, lr):
    opt = torch.optim.SGD(all_params, lr=lr, weight_decay=REG)
    mu = targets.mean()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_plays, device=DEVICE)
        total_loss = 0.0
        for start in range(0, n_plays, BATCH):
            idx = perm[start:start + BATCH]
            u_b, m_b, r_b = u_t[idx], m_t[idx], targets[idx]
            pred = mu + bu(u_b).squeeze() + bi(m_b).squeeze() + \
                (pu(u_b) * qi(m_b)).sum(dim=1)
            loss = ((pred - r_b) ** 2).mean()
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
        uid = int(user_lookup.loc[uids[li], "user_id"]
                  ) if uids[li] in user_lookup.index else -1
        print(f"  #{r+1} {get_name(uid)} bu={bu_np[li]:.4f}")
    print(f"known players:")
    for uid, name in KNOWN.items():
        matches = user_lookup[user_lookup["user_id"] == uid]
        if matches.empty or matches.index[0] not in uid_to_local:
            print(f"  {name}: not in pool")
            continue
        li = uid_to_local[matches.index[0]]
        r = int((bu_np > bu_np[li]).sum()) + 1
        print(f"  {name}: #{r}/{n_users} bu={bu_np[li]:.4f}")


# step 0: initial SVD on log(pp)
print("step 0: initial SVD on log(pp)")
pbar = tqdm(range(INIT_EPOCHS), unit="epoch", desc="init")
opt = torch.optim.SGD(all_params, lr=LR_INIT, weight_decay=REG)
mu_global = log_pp.mean()
for epoch in pbar:
    perm = torch.randperm(n_plays, device=DEVICE)
    total_loss = 0.0
    for start in range(0, n_plays, BATCH):
        idx = perm[start:start + BATCH]
        u_b, m_b, r_b = u_t[idx], m_t[idx], log_pp[idx]
        pred = mu_global + bu(u_b).squeeze() + \
            bi(m_b).squeeze() + (pu(u_b) * qi(m_b)).sum(dim=1)
        loss = ((pred - r_b) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    pbar.set_postfix(mse=f"{total_loss / n_plays:.6f}")

print_rankings("after init")

# outer loop
logsurv = torch.zeros(n_plays, dtype=torch.float32, device=DEVICE)

pbar = tqdm(range(OUTER_ITERS), unit="outer")
for outer in pbar:
    with torch.no_grad():
        # A: relevance weights
        pu_play = pu(u_t)  # (n_plays, k)
        qi_play = qi(m_t)
        w = (pu_play * qi_play).sum(dim=1).exp().clamp(max=W_CLAMP)

        # B: weighted distribution fit per map
        w_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
        w_sum.scatter_add_(0, m_t, w)
        wlp_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
        wlp_sum.scatter_add_(0, m_t, w * log_pp)
        map_mu = wlp_sum / w_sum

        resid = log_pp - map_mu[m_t]
        wvar_sum = torch.zeros(n_maps, dtype=torch.float32, device=DEVICE)
        wvar_sum.scatter_add_(0, m_t, w * resid ** 2)
        map_sigma = (wvar_sum / w_sum).sqrt().clamp(min=0.01)

        # C: log-survival
        z = resid / map_sigma[m_t]
        erfc_val = torch.erfc(z / SQRT2).clamp(min=1e-30)
        prev_logsurv = logsurv.clone()
        logsurv = -(erfc_val / 2.0).log()

    # D: re-fit SVD on log-survivals (warm start)
    mu_ls, final_mse = run_sgd(logsurv, INNER_EPOCHS, LR_REFINE)

    delta = (logsurv - prev_logsurv).abs().max().item()

    known_str = []
    bu_np = bu.weight.detach().cpu().numpy().flatten()
    for uid, name in KNOWN.items():
        matches = user_lookup[user_lookup["user_id"] == uid]
        if matches.empty or matches.index[0] not in uid_to_local:
            known_str.append(f"{name}=N/A")
            continue
        li = uid_to_local[matches.index[0]]
        r = int((bu_np > bu_np[li]).sum()) + 1
        known_str.append(f"{name}=#{r}({bu_np[li]:.2f})")

    pbar.set_postfix_str(
        f"d={delta:.2f} mse={final_mse:.4f} | {' '.join(known_str)}")

print_rankings("final")

# save
np.savez_compressed(
    f"{OUTPUT}/itersvd_hddt.npz",
    bu=bu.weight.detach().cpu().numpy().flatten(),
    bi=bi.weight.detach().cpu().numpy().flatten(),
    pu=pu.weight.detach().cpu().numpy(),
    qi=qi.weight.detach().cpu().numpy(),
    map_mu=map_mu.cpu().numpy(),
    map_sigma=map_sigma.cpu().numpy(),
    uids=uids, mids=mids,
)
print("saved itersvd_hddt.npz")
