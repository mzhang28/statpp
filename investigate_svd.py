import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

OUTPUT = "./training_data"
MOD_POOLS = {"nomod": "", "DT": "DT",
             "HDDT": "DT,HD", "HR": "HR", "HDHR": "HD,HR"}
K = 50
MIN_PLAYS = 40

df_all = pd.read_parquet(f"{OUTPUT}/scores_indexed.parquet")

for pool_name, mods_str in MOD_POOLS.items():
    print(f"\n{'='*40}\n{pool_name} (mods='{mods_str}')\n{'='*40}")
    df = df_all[df_all["mods"] == mods_str]

    # filter maps with < MIN_PLAYS
    counts = df.groupby("map_idx").size()
    keep_maps = set(counts[counts >= MIN_PLAYS].index)
    df = df[df["map_idx"].isin(keep_maps)]

    # reindex to contiguous
    uids, u_inv = np.unique(df["user_idx"].values, return_inverse=True)
    mids, m_inv = np.unique(df["map_idx"].values, return_inverse=True)
    n_users, n_maps = len(uids), len(mids)
    print(f"{n_users} users, {n_maps} maps, {len(df)} scores")

    k = min(K, n_users - 1, n_maps - 1)

    # build + center
    X = csr_matrix(
        (df["score_norm"].values.astype(np.float64), (u_inv, m_inv)),
        shape=(n_users, n_maps),
    )
    Xc = X.tocsc()
    for j in range(n_maps):
        s, e = Xc.indptr[j], Xc.indptr[j + 1]
        if s == e:
            continue
        Xc.data[s:e] -= Xc.data[s:e].mean()
    Xc = Xc.tocsr()

    # SVD
    print(f"running svds k={k}")
    U, S, Vt = svds(Xc, k=k)
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], Vt[order, :]

    # save factors
    np.savez_compressed(f"{OUTPUT}/svd_{pool_name}.npz",
                        U=U, S=S, Vt=Vt, uids=uids, mids=mids)

    # spectrum
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, k + 1), S, "o-", ms=4)
    ax.set_xlabel("component")
    ax.set_ylabel("singular value")
    ax.set_title(pool_name)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/sv_spectrum_{pool_name}.png", dpi=150)
    plt.close()

    # map loadings
    map_lookup = df[["map_idx", "beatmap_id"]
                    ].drop_duplicates().set_index("map_idx")
    mid_to_bid = {mid: int(map_lookup.loc[mid, "beatmap_id"])
                  if mid in map_lookup.index else "?" for mid in mids}

    for c in range(min(5, k)):
        v = Vt[c]
        pos = np.argsort(-v)[:20]
        neg = np.argsort(v)[:20]
        print(f"\n--- {pool_name} component {c} (sv={S[c]:.2f}) ---")
        print("positive:", [mid_to_bid[mids[i]] for i in pos])
        print("negative:", [mid_to_bid[mids[i]] for i in neg])

    # player scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sub = np.random.choice(n_users, min(50000, n_users), replace=False)
    axes[0].scatter(U[sub, 0], U[sub, 1], s=1, alpha=0.1)
    axes[0].set_xlabel("dim 0")
    axes[0].set_ylabel("dim 1")
    axes[1].scatter(U[sub, 0], U[sub, 2], s=1, alpha=0.1)
    axes[1].set_xlabel("dim 0")
    axes[1].set_ylabel("dim 2")
    fig.suptitle(pool_name)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/player_scatter_{pool_name}.png", dpi=150)
    plt.close()

    # rank-1 top players
    rank1 = U[:, 0] * S[0]
    ranking = np.argsort(-rank1)
    user_lookup = df[["user_idx", "user_id"]
                     ].drop_duplicates().set_index("user_idx")
    print(f"\n{pool_name} top 10 by rank-1:")
    for r, li in enumerate(ranking[:10]):
        uid = int(user_lookup.loc[uids[li], "user_id"]
                  ) if uids[li] in user_lookup.index else "?"
        print(f"  #{r+1} user_id={uid} score={rank1[li]:.4f}")

    # known players
    KNOWN = {7562902: "mrekk", 14715160: "toro",
             10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
    uid_to_local = {uid: i for i, uid in enumerate(uids)}
    print(f"\n{pool_name} known players:")
    for uid, name in KNOWN.items():
        if uid not in user_lookup["user_id"].values:
            print(f"  {name}: not in pool")
            continue
        uidx = user_lookup[user_lookup["user_id"] == uid].index[0]
        if uidx not in uid_to_local:
            print(f"  {name}: not in pool")
            continue
        li = uid_to_local[uidx]
        r1_rank = int((rank1 > rank1[li]).sum()) + 1
        dims = ", ".join(f"d{d}={U[li, d]:.4f}" for d in range(min(5, k)))
        print(f"  {name}: rank1=#{r1_rank} ({rank1[li]:.4f}) | {dims}")
