from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn

OUTPUT = "./training_data"
MODS = "DT,HD"
MIN_PLAYS = 100
N_FACTORS = 10
N_EPOCHS = 50
LR = 0.005
REG = 0.001
BATCH = 65536
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_parquet(f"{OUTPUT}/scores_indexed.parquet")
df = df[df["mods"] == MODS]
counts = df.groupby("map_idx").size()
keep = set(counts[counts >= MIN_PLAYS].index)
df = df[df["map_idx"].isin(keep)]

uids, u_inv = np.unique(df["user_idx"].values, return_inverse=True)
mids, m_inv = np.unique(df["map_idx"].values, return_inverse=True)
n_users, n_maps = len(uids), len(mids)
print(f"{n_users} users, {n_maps} maps, {len(df)} scores, device={DEVICE}")

mu = df["score_norm"].mean()
u_t = torch.tensor(u_inv, dtype=torch.long, device=DEVICE)
m_t = torch.tensor(m_inv, dtype=torch.long, device=DEVICE)
r_t = torch.tensor(df["score_norm"].values, dtype=torch.float32, device=DEVICE)

bu = nn.Embedding(n_users, 1).to(DEVICE)
bi = nn.Embedding(n_maps, 1).to(DEVICE)
pu = nn.Embedding(n_users, N_FACTORS).to(DEVICE)
qi = nn.Embedding(n_maps, N_FACTORS).to(DEVICE)
nn.init.zeros_(bu.weight)
nn.init.zeros_(bi.weight)
nn.init.normal_(pu.weight, 0, 0.1)
nn.init.normal_(qi.weight, 0, 0.1)

opt = torch.optim.SGD(list(bu.parameters()) + list(bi.parameters()) +
                      list(pu.parameters()) + list(qi.parameters()), lr=LR, weight_decay=REG)


n = len(r_t)
pbar = tqdm(range(N_EPOCHS), unit="epoch")
for epoch in pbar:
    perm = torch.randperm(n, device=DEVICE)
    total_loss = 0.0
    for start in range(0, n, BATCH):
        idx = perm[start:start + BATCH]
        u_b, m_b, r_b = u_t[idx], m_t[idx], r_t[idx]
        pred = mu + bu(u_b).squeeze() + bi(m_b).squeeze() + \
            (pu(u_b) * qi(m_b)).sum(dim=1)
        loss = ((pred - r_b) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    pbar.set_postfix(mse=f"{total_loss / n:.6f}")

bu_np = bu.weight.detach().cpu().numpy().flatten()
bi_np = bi.weight.detach().cpu().numpy().flatten()
pu_np = pu.weight.detach().cpu().numpy()
qi_np = qi.weight.detach().cpu().numpy()

scores_bias = bu_np.copy()
scores_combined = bu_np + np.linalg.norm(pu_np, axis=1)

user_lookup = df[["user_idx", "user_id"]
                 ].drop_duplicates().set_index("user_idx")


def print_ranking(scores, label):
    ranking = np.argsort(-scores)
    print(f"\ntop 20 by {label}:")
    for r, li in enumerate(ranking[:20]):
        uid = int(user_lookup.loc[uids[li], "user_id"]
                  ) if uids[li] in user_lookup.index else "?"
        print(f"  #{r+1} user_id={uid} score={scores[li]:.4f}")

    KNOWN = {7562902: "mrekk", 14715160: "toro",
             10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
    uid_to_local = {uid: i for i, uid in enumerate(uids)}
    print(f"\nknown players ({label}):")
    for uid, name in KNOWN.items():
        matches = user_lookup[user_lookup["user_id"] == uid]
        if matches.empty or matches.index[0] not in uid_to_local:
            print(f"  {name}: not in pool")
            continue
        li = uid_to_local[matches.index[0]]
        r = int((scores > scores[li]).sum()) + 1
        factors = ", ".join(f"f{d}={pu_np[li, d]:.4f}" for d in range(10))
        print(
            f"  {name}: #{r}/{n_users} bias={bu_np[li]:.4f} norm={np.linalg.norm(pu_np[li]):.4f} | {factors}")


print_ranking(scores_bias, "bias only")
print_ranking(scores_combined, "bias + factor norm")

play_counts = df.groupby("user_idx").size()
uid_to_local = {uid: i for i, uid in enumerate(uids)}
pc_arr = np.array([play_counts.get(uids[i], 0) for i in range(n_users)])
print(
    f"\ncorr(bias, log_playcount) = {np.corrcoef(bu_np, np.log1p(pc_arr))[0, 1]:.4f}")
print(
    f"corr(combined, log_playcount) = {np.corrcoef(scores_combined, np.log1p(pc_arr))[0, 1]:.4f}")

np.savez_compressed(f"{OUTPUT}/pmf_hddt.npz", mu=mu, bu=bu_np,
                    bi=bi_np, pu=pu_np, qi=qi_np, uids=uids, mids=mids)
print("\nsaved pmf_hddt.npz")
