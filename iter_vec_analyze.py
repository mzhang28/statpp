import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUTPUT = "./training-data"
MODS = "DT,HD"

data = np.load(f"{OUTPUT}/itersvd_hddt.npz", allow_pickle=True)
bu, bi, pu, qi = data["bu"], data["bi"], data["pu"], data["qi"]
map_mu, map_sigma = data["map_mu"], data["map_sigma"]
uids, mids = data["uids"], data["mids"]
n_users, n_maps, n_factors = len(uids), len(mids), pu.shape[1]

df = (
    pl.scan_parquet(f"{OUTPUT}/scores.parquet")
    .filter((pl.col("mods") == MODS) & (pl.col("pp") > 0))
    .collect()
)
counts_map = df.group_by("map_idx").len()
keep = set(counts_map.filter(pl.col("len") >= 100)["map_idx"].to_list())
df = df.filter(pl.col("map_idx").is_in(keep)).to_pandas()

uid_to_local = {uid: i for i, uid in enumerate(uids)}
mid_to_local = {mid: i for i, mid in enumerate(mids)}

user_info = df[["user_idx", "user_id", "username"]
               ].drop_duplicates().set_index("user_idx")


def get_name(li):
    uidx = uids[li]
    if uidx in user_info.index:
        row = user_info.loc[uidx]
        return row["username"] if pd.notna(row["username"]) else str(int(row["user_id"]))
    return "?"


# per-user play counts and mean pp
user_stats = df.groupby("user_idx").agg(
    n_plays=("pp", "size"), mean_pp=("pp", "mean"), total_pp=("pp", "sum"))

KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa"}
known_locals = {}
for uid, name in KNOWN.items():
    matches = user_info[user_info["user_id"] == uid]
    if not matches.empty and matches.index[0] in uid_to_local:
        known_locals[name] = uid_to_local[matches.index[0]]

pdf = PdfPages(f"{OUTPUT}/diagnostics.pdf")

# --- 1. Bias vs play count ---
fig, ax = plt.subplots(figsize=(10, 7))
pc = np.array([user_stats.loc[uids[i], "n_plays"] if uids[i]
              in user_stats.index else 1 for i in range(n_users)])
ax.scatter(np.log10(pc), bu, s=2, alpha=0.15)
for name, li in known_locals.items():
    ax.annotate(name, (np.log10(pc[li]), bu[li]), fontsize=8, color="red")
corr = np.corrcoef(np.log10(pc), bu)[0, 1]
ax.set_xlabel("log10(play count)")
ax.set_ylabel("user bias (bu)")
ax.set_title(f"bias vs play count (corr={corr:.3f})")
pdf.savefig(fig)
plt.close()

# --- 2. Bias rank vs pp rank ---
# approximate pp rank: sum of top 100 pp scores per user, weighted


def weighted_pp(group):
    top = group.nlargest(100, "pp")["pp"].values
    weights = 0.95 ** np.arange(len(top))
    return (top * weights).sum()


pp_rank_series = df.groupby("user_idx").apply(weighted_pp)
pp_scores = np.array([pp_rank_series.get(uids[i], 0) for i in range(n_users)])
pp_ranking = np.argsort(-pp_scores)
bu_ranking = np.argsort(-bu)
pp_rank_of = np.empty(n_users, dtype=int)
bu_rank_of = np.empty(n_users, dtype=int)
pp_rank_of[pp_ranking] = np.arange(n_users)
bu_rank_of[bu_ranking] = np.arange(n_users)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(pp_rank_of, bu_rank_of, s=2, alpha=0.1)
ax.plot([0, n_users], [0, n_users], "r--", lw=0.5)
for name, li in known_locals.items():
    ax.annotate(
        name, (pp_rank_of[li], bu_rank_of[li]), fontsize=8, color="red")
ax.set_xlabel("pp rank (weighted top 100)")
ax.set_ylabel("model bias rank")
ax.set_title("pp rank vs model bias rank")
ax.invert_xaxis()
ax.invert_yaxis()
pdf.savefig(fig)
plt.close()

# biggest disagreements
# positive = model ranks higher (underrated by pp)
rank_diff = pp_rank_of - bu_rank_of
overrated = np.argsort(rank_diff)[:20]  # model ranks lower (pp inflated)
underrated = np.argsort(-rank_diff)[:20]  # model ranks higher (pp deflated)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(20), rank_diff[overrated], color="salmon",
        label="overrated (high pp, low bias)")
ax.barh(range(20, 40), rank_diff[underrated],
        color="steelblue", label="underrated (low pp, high bias)")
labels = [f"{get_name(i)} pp#{pp_rank_of[i]} bu#{bu_rank_of[i]}" for i in overrated] + \
         [f"{get_name(i)} pp#{pp_rank_of[i]} bu#{bu_rank_of[i]}" for i in underrated]
ax.set_yticks(range(40))
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel("rank difference (pp_rank - bias_rank)")
ax.set_title("biggest rank disagreements")
ax.legend(fontsize=8)
plt.tight_layout()
pdf.savefig(fig)
plt.close()

# --- 5. Dimension profiles of known players ---
if known_locals:
    names = list(known_locals.keys())
    factors = np.array([pu[known_locals[n]] for n in names])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_factors)
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        ax.bar(x + k * width, factors[k], width, label=name)
    ax.set_xticks(x + width * len(names) / 2)
    ax.set_xticklabels([f"d{i}" for i in range(n_factors)])
    ax.set_ylabel("factor loading")
    ax.set_title("player factor profiles")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # cosine similarity matrix
    norms = np.linalg.norm(factors, axis=1, keepdims=True)
    cos_sim = (factors @ factors.T) / (norms @ norms.T + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cos_sim[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("player profile similarity")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # print table
    print("\nknown player profiles:")
    header = f"{'name':<12}" + "".join(f"{'d'+str(d):>8}" for d in range(
        n_factors)) + f"{'bu':>8}" + f"{'||pu||':>8}"
    print(header)
    for name in names:
        li = known_locals[name]
        vals = "".join(f"{pu[li, d]:>8.3f}" for d in range(n_factors))
        print(f"{name:<12}{vals}{bu[li]:>8.3f}{np.linalg.norm(pu[li]):>8.3f}")

pdf.close()
print(f"\nsaved {OUTPUT}/diagnostics.pdf")
