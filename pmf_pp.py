import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

OUTPUT = "./training-data"
MODS = "DT,HD"
MIN_PLAYS = 100
N_FACTORS = 10
N_EPOCHS = 20
LR = 0.005
REG = 0.02
BATCH = 65536
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_parquet(f"{OUTPUT}/scores.parquet")
df = df[df["mods"] == MODS]
df = df[df["pp"] > 0]  # drop scores with no pp

counts = df.groupby("map_idx").size()
keep = set(counts[counts >= MIN_PLAYS].index)
df = df[df["map_idx"].isin(keep)]

uids, u_inv = np.unique(df["user_idx"].values, return_inverse=True)
mids, m_inv = np.unique(df["map_idx"].values, return_inverse=True)
n_users, n_maps = len(uids), len(mids)

ratings = np.log1p(df["pp"].values).astype(np.float32)
print(f"{n_users} users, {n_maps} maps, {len(df)} scores, device={DEVICE}")
print(
    f"log(pp+1) range: [{ratings.min():.2f}, {ratings.max():.2f}] mean={ratings.mean():.2f} std={ratings.std():.2f}")

mu = float(ratings.mean())
u_t = torch.tensor(u_inv, dtype=torch.long, device=DEVICE)
m_t = torch.tensor(m_inv, dtype=torch.long, device=DEVICE)
r_t = torch.tensor(ratings, dtype=torch.float32, device=DEVICE)

bu = nn.Embedding(n_users, 1).to(DEVICE)
bi = nn.Embedding(n_maps, 1).to(DEVICE)
pu = nn.Embedding(n_users, N_FACTORS).to(DEVICE)
qi = nn.Embedding(n_maps, N_FACTORS).to(DEVICE)
nn.init.zeros_(bu.weight)
nn.init.zeros_(bi.weight)
nn.init.normal_(pu.weight, 0, 0.1)
nn.init.normal_(qi.weight, 0, 0.1)

opt = torch.optim.SGD(
    list(bu.parameters()) + list(bi.parameters()) +
    list(pu.parameters()) + list(qi.parameters()),
    lr=LR, weight_decay=REG,
)

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

# endogenous skill: u_i . v_bar, no biases
norms = np.linalg.norm(qi_np, axis=1)
v_bar = (qi_np * norms[:, None]).sum(axis=0)
skill = pu_np @ v_bar

ranking = np.argsort(-skill)
ranking_bias = np.argsort(-bu_np)

user_lookup = df[["user_idx", "user_id"]
                 ].drop_duplicates().set_index("user_idx")
uid_to_local = {uid: i for i, uid in enumerate(uids)}

# try to get usernames
if "username" in df.columns:
    name_lookup = df[["user_id", "username"]
                     ].drop_duplicates().set_index("user_id")
    def get_name(
        uid): return name_lookup.loc[uid, "username"] if uid in name_lookup.index else str(uid)
else:
    def get_name(uid): return str(uid)


def uid_for(li):
    return int(user_lookup.loc[uids[li], "user_id"]) if uids[li] in user_lookup.index else -1


print("\ntop 20 by endogenous skill (bias-free):")
for r, li in enumerate(ranking[:20]):
    uid = uid_for(li)
    print(
        f"  #{r+1} {get_name(uid)} skill={skill[li]:.4f} bias={bu_np[li]:.4f}")

print("\ntop 20 by user bias (farm index):")
for r, li in enumerate(ranking_bias[:20]):
    uid = uid_for(li)
    print(
        f"  #{r+1} {get_name(uid)} bias={bu_np[li]:.4f} skill={skill[li]:.4f}")

print("\nbottom 20 by user bias (underrated):")
for r, li in enumerate(ranking_bias[::-1][:20]):
    uid = uid_for(li)
    print(
        f"  #{r+1} {get_name(uid)} bias={bu_np[li]:.4f} skill={skill[li]:.4f}")

# top/bottom item biases (farm maps vs underrated maps)
map_lookup = df[["map_idx", "beatmap_id"]
                ].drop_duplicates().set_index("map_idx")
mid_to_bid = {mid: int(map_lookup.loc[mid, "beatmap_id"])
              if mid in map_lookup.index else -1 for mid in mids}

map_rank = np.argsort(-bi_np)
print("\ntop 20 farm maps (highest item bias):")
for r, li in enumerate(map_rank[:20]):
    print(
        f"  #{r+1} beatmap_id={mid_to_bid[mids[li]]} bi={bi_np[li]:.4f} ||v||={norms[li]:.4f}")

print("\ntop 20 underrated maps (lowest item bias):")
for r, li in enumerate(map_rank[::-1][:20]):
    print(
        f"  #{r+1} beatmap_id={mid_to_bid[mids[li]]} bi={bi_np[li]:.4f} ||v||={norms[li]:.4f}")

KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
print("\nknown players:")
for uid, name in KNOWN.items():
    matches = user_lookup[user_lookup["user_id"] == uid]
    if matches.empty or matches.index[0] not in uid_to_local:
        print(f"  {name}: not in pool")
        continue
    li = uid_to_local[matches.index[0]]
    r_s = int((skill > skill[li]).sum()) + 1
    r_b = int((bu_np > bu_np[li]).sum()) + 1
    print(
        f"  {name}: skill #{r_s} | bias #{r_b} | skill={skill[li]:.4f} bias={bu_np[li]:.4f}")

np.savez_compressed(f"{OUTPUT}/pmf_pp_hddt.npz", mu=mu, bu=bu_np,
                    bi=bi_np, pu=pu_np, qi=qi_np, uids=uids, mids=mids)
print("\nsaved pmf_pp_hddt.npz")
