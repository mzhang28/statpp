import numpy as np
import pandas as pd
import implicit
from scipy.sparse import csr_matrix

OUTPUT = "./training_data"
MODS = "DT,HD"
MIN_PLAYS = 100

df = pd.read_parquet(f"{OUTPUT}/scores_indexed.parquet")
df = df[df["mods"] == MODS]

counts = df.groupby("map_idx").size()
keep = set(counts[counts >= MIN_PLAYS].index)
df = df[df["map_idx"].isin(keep)]

uids, u_inv = np.unique(df["user_idx"].values, return_inverse=True)
mids, m_inv = np.unique(df["map_idx"].values, return_inverse=True)
n_users, n_maps = len(uids), len(mids)
print(f"{n_users} users, {n_maps} maps, {len(df)} scores")

X = csr_matrix(
    (df["score_norm"].values.astype(np.float32), (u_inv, m_inv)),
    shape=(n_users, n_maps),
)

model = implicit.als.AlternatingLeastSquares(
    factors=10, iterations=15, regularization=0.1)
model.fit(X.T)

uf = model.item_factors
norms = np.linalg.norm(uf, axis=1)
ranking = np.argsort(-norms)

user_lookup = df[["user_idx", "user_id"]
                 ].drop_duplicates().set_index("user_idx")
uid_to_local = {uid: i for i, uid in enumerate(uids)}

print("\ntop 20 by factor norm:")
for r, li in enumerate(ranking[:20]):
    uid = int(user_lookup.loc[uids[li], "user_id"]
              ) if uids[li] in user_lookup.index else "?"
    print(f"  #{r+1} user_id={uid} norm={norms[li]:.4f}")

KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
print("\nknown players:")
for uid, name in KNOWN.items():
    matches = user_lookup[user_lookup["user_id"] == uid]
    if matches.empty:
        print(f"  {name}: not in pool")
        continue
    uidx = matches.index[0]
    if uidx not in uid_to_local:
        print(f"  {name}: not in pool")
        continue
    li = uid_to_local[uidx]
    r = int((norms > norms[li]).sum()) + 1
    factors = ", ".join(f"f{d}={uf[li, d]:.4f}" for d in range(10))
    print(f"  {name}: #{r}/{n_users} norm={norms[li]:.4f} | {factors}")

np.savez_compressed(f"{OUTPUT}/pmf_hddt.npz", user_factors=uf,
                    item_factors=model.user_factors, uids=uids, mids=mids)
print("\nsaved pmf_hddt.npz")
