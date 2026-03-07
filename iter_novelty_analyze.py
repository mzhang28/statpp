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

# load model
data = np.load(f"{OUTPUT}/itersvd_novelty.npz", allow_pickle=True)
bu, bi, pu, qi = data["bu"], data["bi"], data["pu"], data["qi"]
map_mu, map_sigma = data["map_mu"], data["map_sigma"]
uids, mids = data["uids"], data["mids"]
n_users, n_maps = len(uids), len(mids)

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

# recompute log-survival from saved model
log_pp = np.log(df["pp"].values)
mu_play = map_mu[df["local_m"].values]
sigma_play = map_sigma[df["local_m"].values]
z = (log_pp - mu_play) / sigma_play
df["logsurv"] = -sp_norm.logsf(z)
df["map_sigma"] = sigma_play

# map quality = mean bu of players who played each map
df["bu"] = bu[df["local_u"].values]
map_q_series = df.groupby("local_m")["bu"].mean()
map_q = np.zeros(n_maps)
for lm, v in map_q_series.items():
    map_q[lm] = v
df["map_q"] = map_q[df["local_m"].values]

df["contribution"] = df["map_q"] * df["logsurv"]

map_play_counts = df.groupby("map_idx").size().to_dict()
df["map_plays"] = df["map_idx"].map(map_play_counts)

# player skill: weighted mean of top-k contributions
player_skills = {}
for lu, group in df.groupby("local_u"):
    top = group.nlargest(TOP_K, "contribution")
    player_skills[lu] = top["contribution"].mean()
skill = np.zeros(n_users)
for lu, s in player_skills.items():
    skill[lu] = s

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
         10549880: "ninerik", 9269034: "accolibed", 15406985: "ivaxa", 3792472: "Woey", 12408961: "MALISZEWSKI"}
known_locals = {}
for uid, name in KNOWN.items():
    matches = user_info[user_info["user_id"] == uid]
    if not matches.empty and matches.index[0] in uid_to_local:
        known_locals[name] = uid_to_local[matches.index[0]]

ranking = np.argsort(-skill)
top10 = ranking[:10]

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


h("<!DOCTYPE html><html><head><meta charset='utf-8'>")
h("<title>Skill Report — All Mods (Novelty)</title>")
h("""<style>
body { font-family: -apple-system, sans-serif; margin: 2em; background: #0d1117; color: #c9d1d9; }
h1, h2, h3 { color: #58a6ff; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #30363d; padding: 4px 10px; text-align: right; font-size: 13px; }
th { background: #161b22; }
tr:nth-child(even) { background: #161b22; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.section { margin: 2em 0; }
td.left { text-align: left; }
</style>""")
h("</head><body>")
h("<h1>Iterative SVD + Novelty — All Mods Report</h1>")
h(f"<p>{n_users} users, {n_maps} items, {len(df)} plays</p>")

# top 30
h('<div class="section"><h2>Top 30 Players by Skill</h2>')
h("<table><tr><th>#</th><th>Player</th><th>Skill</th><th>bu</th></tr>")
for r, li in enumerate(ranking[:30]):
    uidx = uids[li]
    uid = uid_for(uidx)
    name = uname(uidx)
    h(f"<tr><td>{r+1}</td><td class='left'>{osu_user_link(uid, name)}</td>"
      f"<td>{skill[li]:.4f}</td><td>{bu[li]:.4f}</td></tr>")
h("</table></div>")

# known player breakdowns
for pname, li in known_locals.items():
    uidx = uids[li]
    uid = uid_for(uidx)
    r = int((skill > skill[li]).sum()) + 1
    player_df = df[df["local_u"] == li].sort_values(
        "contribution", ascending=False)
    top50 = player_df.head(TOP_K)

    h(f'<div class="section"><h2>{osu_user_link(uid, pname)} — #{r}/{n_users} (skill={skill[li]:.4f}, bu={bu[li]:.4f})</h2>')

    h("<h3>Top 20 Plays by Contribution</h3>")
    h("<table><tr><th>#</th><th>Beatmap</th><th>PP</th><th>Log-surv</th><th>Map Q</th><th>Contrib</th><th>σ</th><th>Plays</th></tr>")
    for i, (_, row) in enumerate(top50.head(20).iterrows()):
        mods = map_info.loc[row["map_idx"],
                            "mods"] if row["map_idx"] in map_info.index else ""
        h(f"<tr><td>{i+1}</td><td class='left'>{osu_map_link(row['beatmap_id'], mods)}</td>"
          f"<td>{row['pp']:.1f}</td><td>{row['logsurv']:.3f}</td><td>{row['map_q']:.3f}</td>"
          f"<td>{row['contribution']:.3f}</td><td>{row['map_sigma']:.3f}</td><td>{int(row['map_plays'])}</td></tr>")
    h("</table>")

    if len(top50) >= 10:
        bottom5 = top50.tail(5)
        h("<h3>Bottom 5 of Top 50</h3>")
        h("<table><tr><th>#</th><th>Beatmap</th><th>PP</th><th>Log-surv</th><th>Map Q</th><th>Contrib</th><th>σ</th><th>Plays</th></tr>")
        for i, (_, row) in enumerate(bottom5.iterrows()):
            mods = map_info.loc[row["map_idx"],
                                "mods"] if row["map_idx"] in map_info.index else ""
            h(f"<tr><td>{TOP_K - 4 + i}</td><td class='left'>{osu_map_link(row['beatmap_id'], mods)}</td>"
              f"<td>{row['pp']:.1f}</td><td>{row['logsurv']:.3f}</td><td>{row['map_q']:.3f}</td>"
              f"<td>{row['contribution']:.3f}</td><td>{row['map_sigma']:.3f}</td><td>{int(row['map_plays'])}</td></tr>")
        h("</table>")

    n_on_top = top50["map_idx"].isin(top100_maps).sum()
    h(f"<p>Plays on top-100 competitive maps: <b>{n_on_top}/{len(top50)}</b></p>")
    h("</div>")

# top 10 competitive fraction
h('<div class="section"><h2>Top 10 — Competitive Map Fraction</h2>')
h("<table><tr><th>#</th><th>Player</th><th>Skill</th><th>Top-50 on Top-100 Maps</th></tr>")
for r, li in enumerate(top10):
    uidx = uids[li]
    uid = uid_for(uidx)
    name = uname(uidx)
    player_df = df[df["local_u"] == li].sort_values(
        "contribution", ascending=False).head(TOP_K)
    n_on = player_df["map_idx"].isin(top100_maps).sum()
    h(f"<tr><td>{r+1}</td><td class='left'>{osu_user_link(uid, name)}</td>"
      f"<td>{skill[li]:.4f}</td><td>{n_on}/{len(player_df)}</td></tr>")
h("</table></div>")

# top 20 maps by q
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

# bottom 20 maps by q
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
