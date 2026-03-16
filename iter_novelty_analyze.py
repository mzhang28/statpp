import math
import numpy as np
import pandas as pd
import polars as pl
import webbrowser
from pathlib import Path
from scipy.stats import norm as sp_norm

OUTPUT = "./training-data"
MIN_PLAYS = 30
TOP_K = 50
SHRINKAGE_N0 = 50.0

# load model
data = np.load(f"{OUTPUT}/itersvd_novelty.npz", allow_pickle=True)
bu, bi, pu, qi = data["bu"], data["bi"], data["pu"], data["qi"]
map_mu, map_sigma = data["map_mu"], data["map_sigma"]
uids, mids = data["uids"], data["mids"]
n_users, n_maps = len(uids), len(mids)
n_factors = pu.shape[1]

# load plays
df = (
    pl.scan_parquet(f"{OUTPUT}/scores.parquet")
    .filter(pl.col("pp") > 0)
    .collect()
)
counts = df.group_by("map_idx").len()
keep = set(counts.filter(pl.col("len") >= MIN_PLAYS)["map_idx"].to_list())
df = df.filter(pl.col("map_idx").is_in(keep)).to_pandas()

uid_to_local = {uid: i for i, uid in enumerate(uids)}
mid_to_local = {mid: i for i, mid in enumerate(mids)}

df["local_u"] = df["user_idx"].map(uid_to_local)
df["local_m"] = df["map_idx"].map(mid_to_local)
df = df.dropna(subset=["local_u", "local_m"])
df["local_u"] = df["local_u"].astype(int)
df["local_m"] = df["local_m"].astype(int)

# recompute log-survival with shrinkage
log_pp = np.log(df["pp"].values)
mu_play = map_mu[df["local_m"].values]
sigma_play = map_sigma[df["local_m"].values]
z = (log_pp - mu_play) / sigma_play
raw_logsurv = -sp_norm.logsf(z)

map_play_counts = df.groupby("map_idx").size().to_dict()
map_counts_arr = np.zeros(n_maps)
for midx, c in map_play_counts.items():
    if midx in mid_to_local:
        map_counts_arr[mid_to_local[midx]] = c
alpha = map_counts_arr[df["local_m"].values] / \
    (map_counts_arr[df["local_m"].values] + SHRINKAGE_N0)
global_mean_ls = raw_logsurv.mean()
df["logsurv"] = alpha * raw_logsurv + (1.0 - alpha) * global_mean_ls
df["map_sigma"] = sigma_play

# map quality = mean bu of players who played each map
df["bu"] = bu[df["local_u"].values]
map_q_series = df.groupby("local_m")["bu"].mean()
map_q = np.zeros(n_maps)
for lm, v in map_q_series.items():
    map_q[lm] = v
df["map_q"] = map_q[df["local_m"].values]
df["contribution"] = df["map_q"] * df["logsurv"]
df["map_plays"] = df["map_idx"].map(map_play_counts)

# player skill: weighted mean of top-k contributions
player_skills = {}
for lu, group in df.groupby("local_u"):
    top = group.nlargest(TOP_K, "contribution")
    player_skills[lu] = top["contribution"].mean()
skill = np.zeros(n_users)
for lu, s in player_skills.items():
    skill[lu] = s

# mahalanobis combined ranking (z-score normalized)
Sigma = np.cov(pu.T)
Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(n_factors))
mahal_scores = np.array([pu[i] @ Sigma_inv @ pu[i] for i in range(n_users)])
mahal_sqrt = np.sqrt(mahal_scores)

bu_z = (bu - bu.mean()) / (bu.std() + 1e-9)
mahal_z = (mahal_sqrt - mahal_sqrt.mean()) / (mahal_sqrt.std() + 1e-9)
combined = bu_z + mahal_z

# PCA of user factors
eigvals, eigvecs = np.linalg.eigh(Sigma)
eig_order = np.argsort(-eigvals)
eigvals, eigvecs = eigvals[eig_order], eigvecs[:, eig_order]
user_pca = pu @ eigvecs  # (n_users, n_factors)
map_pca = qi @ eigvecs

# user info
user_info = df[["user_idx", "user_id", "username"]
               ].drop_duplicates().set_index("user_idx")


def uname(uidx):
    if uidx in user_info.index:
        row = user_info.loc[uidx]
        return row["username"] if pd.notna(row["username"]) else str(int(row["user_id"]))
    return "?"


def uid_for(uidx):
    return int(user_info.loc[uidx, "user_id"]) if uidx in user_info.index else 0


KNOWN = {7562902: "mrekk", 14715160: "toro",
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa",
         3792472: "Woey", 12408961: "MALISZEWSKI"}
known_locals = {}
for uid, name in KNOWN.items():
    matches = user_info[user_info["user_id"] == uid]
    if not matches.empty and matches.index[0] in uid_to_local:
        known_locals[name] = uid_to_local[matches.index[0]]

ranking_skill = np.argsort(-skill)
ranking_bu = np.argsort(-bu)
ranking_combined = np.argsort(-combined)
top10 = ranking_skill[:10]

map_q_ranking = np.argsort(-map_q)
top100_maps = set(mids[map_q_ranking[:100]])

map_info = df[["map_idx", "beatmap_id", "mods"]
              ].drop_duplicates().set_index("map_idx")

# --- HTML ---
html = []
h = html.append


def osu_map_link(bid, mods=""):
    label = f"{int(bid)}"
    if mods:
        label += f" +{mods}"
    return f'<a href="https://osu.ppy.sh/b/{int(bid)}" target="_blank">{label}</a>'


def osu_user_link(uid, name):
    return f'<a href="https://osu.ppy.sh/users/{int(uid)}" target="_blank">{name}</a>'


def player_cell(li):
    uidx = uids[li]
    return f"{osu_user_link(uid_for(uidx), uname(uidx))}"


h("<!DOCTYPE html><html><head><meta charset='utf-8'>")
h("<title>Skill Report — All Mods (Novelty v2)</title>")
h("""<style>
body { font-family: -apple-system, sans-serif; margin: 2em; background: #0d1117; color: #c9d1d9; max-width: 1400px; }
h1, h2, h3 { color: #58a6ff; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #30363d; padding: 4px 10px; text-align: right; font-size: 13px; }
th { background: #161b22; position: sticky; top: 0; }
tr:nth-child(even) { background: #161b22; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.section { margin: 2em 0; }
td.left { text-align: left; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
.note { color: #8b949e; font-size: 12px; margin-top: -0.5em; }
</style>""")
h("</head><body>")
h("<h1>Iterative SVD + Novelty — All Mods Report (v2)</h1>")
h(f"<p>{n_users} users, {n_maps} items, {len(df)} plays | shrinkage n₀={SHRINKAGE_N0} | top-k={TOP_K}</p>")

# --- Rankings comparison ---
h('<div class="section"><h2>Top 30 Players — Three Rankings</h2>')
h('<p class="note">Skill = mean contribution of top-50 plays | bu = user bias from SVD | Combined = bu + √mahal</p>')
h("<table><tr><th>#</th><th>By Skill</th><th>Skill</th><th>By bu</th><th>bu</th><th>By Combined</th><th>Combined</th></tr>")
for r in range(30):
    li_s, li_b, li_c = ranking_skill[r], ranking_bu[r], ranking_combined[r]
    h(f"<tr><td>{r+1}</td>"
      f"<td class='left'>{player_cell(li_s)}</td><td>{skill[li_s]:.4f}</td>"
      f"<td class='left'>{player_cell(li_b)}</td><td>{bu[li_b]:.4f}</td>"
      f"<td class='left'>{player_cell(li_c)}</td><td>{combined[li_c]:.4f}</td></tr>")
h("</table></div>")

# --- Known players summary ---
h('<div class="section"><h2>Known Players Summary</h2>')
h("<table><tr><th>Player</th><th>Skill Rank</th><th>Skill</th><th>bu Rank</th><th>bu</th>"
  "<th>Combined Rank</th><th>Combined</th><th>Mahal</th></tr>")
for pname, li in known_locals.items():
    uidx = uids[li]
    uid = uid_for(uidx)
    rs = int((skill > skill[li]).sum()) + 1
    rb = int((bu > bu[li]).sum()) + 1
    rc = int((combined > combined[li]).sum()) + 1
    h(f"<tr><td class='left'>{osu_user_link(uid, pname)}</td>"
      f"<td>{rs}</td><td>{skill[li]:.4f}</td>"
      f"<td>{rb}</td><td>{bu[li]:.4f}</td>"
      f"<td>{rc}</td><td>{combined[li]:.4f}</td>"
      f"<td>{mahal_scores[li]:.4f}</td></tr>")
h("</table></div>")

# --- PCA profiles ---
h('<div class="section"><h2>Known Players — PCA Profiles</h2>')
h(f'<p class="note">Eigenvalues: {", ".join(f"PC{d}={eigvals[d]:.4f}" for d in range(
    min(5, n_factors)))}</p>')
h("<table><tr><th>Player</th><th>bu</th><th>‖pu‖</th>")
for d in range(min(5, n_factors)):
    h(f"<th>PC{d}</th>")
h("</tr>")
for pname, li in known_locals.items():
    uidx = uids[li]
    uid = uid_for(uidx)
    h(f"<tr><td class='left'>{osu_user_link(uid, pname)}</td>"
      f"<td>{bu[li]:.3f}</td><td>{np.linalg.norm(pu[li]):.3f}</td>")
    for d in range(min(5, n_factors)):
        h(f"<td>{user_pca[li, d]:.3f}</td>")
    h("</tr>")
h("</table></div>")

# --- PCA dimension interpretation ---
h('<div class="section"><h2>PCA Dimensions — Top/Bottom Maps</h2>')
for d in range(min(5, n_factors)):
    proj = map_pca[:, d]
    top_m = np.argsort(-proj)[:15]
    bot_m = np.argsort(proj)[:15]
    h(f"<h3>PC{d} (variance={eigvals[d]:.4f})</h3>")
    h('<div class="grid">')
    # positive
    h("<div><h4>Positive loading</h4><table><tr><th>#</th><th>Beatmap</th><th>Loading</th></tr>")
    for i, mi in enumerate(top_m):
        midx = mids[mi]
        bid = int(map_info.loc[midx, "beatmap_id"]
                  ) if midx in map_info.index else 0
        mods = map_info.loc[midx, "mods"] if midx in map_info.index else ""
        h(f"<tr><td>{i+1}</td><td class='left'>{osu_map_link(bid, mods)}</td><td>{proj[mi]:.4f}</td></tr>")
    h("</table></div>")
    # negative
    h("<div><h4>Negative loading</h4><table><tr><th>#</th><th>Beatmap</th><th>Loading</th></tr>")
    for i, mi in enumerate(bot_m):
        midx = mids[mi]
        bid = int(map_info.loc[midx, "beatmap_id"]
                  ) if midx in map_info.index else 0
        mods = map_info.loc[midx, "mods"] if midx in map_info.index else ""
        h(f"<tr><td>{i+1}</td><td class='left'>{osu_map_link(bid, mods)}</td><td>{proj[mi]:.4f}</td></tr>")
    h("</table></div>")
    h("</div>")
h("</div>")

# --- Known player breakdowns ---
for pname, li in known_locals.items():
    uidx = uids[li]
    uid = uid_for(uidx)
    r = int((skill > skill[li]).sum()) + 1
    player_df = df[df["local_u"] == li].sort_values(
        "contribution", ascending=False)
    top50 = player_df.head(TOP_K)

    h(f'<div class="section"><h2>{osu_user_link(uid, pname)} — skill #{r}/{n_users} '
      f'(skill={skill[li]:.4f}, bu={bu[li]:.4f}, combined={combined[li]:.4f})</h2>')

    h("<h3>Top 20 Plays by Contribution</h3>")
    h("<table><tr><th>#</th><th>Beatmap</th><th>PP</th><th>Log-surv</th><th>Map Q</th>"
      "<th>Contrib</th><th>σ</th><th>Plays</th><th>Shrink α</th></tr>")
    for i, (_, row) in enumerate(top50.head(20).iterrows()):
        mods = map_info.loc[row["map_idx"],
                            "mods"] if row["map_idx"] in map_info.index else ""
        mc = map_play_counts.get(row["map_idx"], 0)
        a = mc / (mc + SHRINKAGE_N0)
        h(f"<tr><td>{i+1}</td><td class='left'>{osu_map_link(row['beatmap_id'], mods)}</td>"
          f"<td>{row['pp']:.1f}</td><td>{row['logsurv']:.3f}</td><td>{row['map_q']:.3f}</td>"
          f"<td>{row['contribution']:.3f}</td><td>{row['map_sigma']:.3f}</td>"
          f"<td>{int(row['map_plays'])}</td><td>{a:.2f}</td></tr>")
    h("</table>")

    if len(top50) >= 10:
        bottom5 = top50.tail(5)
        h("<h3>Bottom 5 of Top 50</h3>")
        h("<table><tr><th>#</th><th>Beatmap</th><th>PP</th><th>Log-surv</th><th>Map Q</th>"
          "<th>Contrib</th><th>σ</th><th>Plays</th></tr>")
        for i, (_, row) in enumerate(bottom5.iterrows()):
            mods = map_info.loc[row["map_idx"],
                                "mods"] if row["map_idx"] in map_info.index else ""
            h(f"<tr><td>{TOP_K - 4 + i}</td><td class='left'>{osu_map_link(row['beatmap_id'], mods)}</td>"
              f"<td>{row['pp']:.1f}</td><td>{row['logsurv']:.3f}</td><td>{row['map_q']:.3f}</td>"
              f"<td>{row['contribution']:.3f}</td><td>{row['map_sigma']:.3f}</td>"
              f"<td>{int(row['map_plays'])}</td></tr>")
        h("</table>")

    n_on_top = top50["map_idx"].isin(top100_maps).sum()
    h(f"<p>Plays on top-100 competitive maps: <b>{n_on_top}/{len(top50)}</b></p>")
    h("</div>")

# --- Top 10 competitive fraction ---
h('<div class="section"><h2>Top 10 — Competitive Map Fraction</h2>')
h("<table><tr><th>#</th><th>Player</th><th>Skill</th><th>bu</th><th>Combined</th><th>Top-50 on Top-100 Maps</th></tr>")
for r, li in enumerate(top10):
    uidx = uids[li]
    uid = uid_for(uidx)
    name = uname(uidx)
    player_df = df[df["local_u"] == li].sort_values(
        "contribution", ascending=False).head(TOP_K)
    n_on = player_df["map_idx"].isin(top100_maps).sum()
    h(f"<tr><td>{r+1}</td><td class='left'>{osu_user_link(uid, name)}</td>"
      f"<td>{skill[li]:.4f}</td><td>{bu[li]:.4f}</td><td>{combined[li]:.4f}</td>"
      f"<td>{n_on}/{len(player_df)}</td></tr>")
h("</table></div>")

# --- Top/bottom maps by quality ---
h('<div class="section"><h2>Top 20 Maps by Quality</h2>')
h("<table><tr><th>#</th><th>Beatmap</th><th>q</th><th>σ</th><th>μ</th><th>Plays</th></tr>")
for r, mi in enumerate(map_q_ranking[:20]):
    midx = mids[mi]
    bid = int(map_info.loc[midx, "beatmap_id"]
              ) if midx in map_info.index else 0
    mods = map_info.loc[midx, "mods"] if midx in map_info.index else ""
    pc = map_play_counts.get(midx, 0)
    h(f"<tr><td>{r+1}</td><td class='left'>{osu_map_link(bid, mods)}</td>"
      f"<td>{map_q[mi]:.4f}</td><td>{map_sigma[mi]:.3f}</td><td>{map_mu[mi]:.3f}</td><td>{pc}</td></tr>")
h("</table></div>")

h('<div class="section"><h2>Bottom 20 Maps by Quality</h2>')
h("<table><tr><th>#</th><th>Beatmap</th><th>q</th><th>σ</th><th>μ</th><th>Plays</th></tr>")
for r, mi in enumerate(map_q_ranking[::-1][:20]):
    midx = mids[mi]
    bid = int(map_info.loc[midx, "beatmap_id"]
              ) if midx in map_info.index else 0
    mods = map_info.loc[midx, "mods"] if midx in map_info.index else ""
    pc = map_play_counts.get(midx, 0)
    h(f"<tr><td>{r+1}</td><td class='left'>{osu_map_link(bid, mods)}</td>"
      f"<td>{map_q[mi]:.4f}</td><td>{map_sigma[mi]:.3f}</td><td>{map_mu[mi]:.3f}</td><td>{pc}</td></tr>")
h("</table></div>")

h("</body></html>")

out_path = Path(OUTPUT) / "report.html"
out_path.write_text("\n".join(html))
print(f"saved {out_path}")
webbrowser.open(f"file://{out_path.resolve()}")
