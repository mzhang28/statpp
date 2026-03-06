import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from tqdm import tqdm

df = pd.read_parquet("./training_data/scores_indexed.parquet")
EPS = 1e-6

counts = df.groupby("map_idx").size()
keep = set(counts[counts >= 100].index)
df = df[df["map_idx"].isin(keep)]
print(f"kept {len(keep)} maps with >= 100 scores")

print("fitting beta per map")
results = []
for map_idx, group in tqdm(df.groupby("map_idx"), unit="map"):
    acc = np.clip(group["score_norm"].values, EPS, 1 - EPS)
    try:
        a, b, _, _ = beta_dist.fit(acc, floc=0, fscale=1)
    except Exception:
        continue
    results.append({
        "map_idx": map_idx,
        "alpha": a,
        "beta": b,
        "n_scores": len(acc),
        "mean_score": acc.mean(),
    })

fits = pd.DataFrame(results)
print(f"{len(fits)} maps fitted, {len(keep) - len(fits)} failed to converge")
fits["fitted_mean"] = fits["alpha"] / (fits["alpha"] + fits["beta"])
fits["precision"] = fits["alpha"] + fits["beta"]
fits["mean_div"] = (fits["fitted_mean"] - fits["mean_score"]).abs()
fits.to_parquet("./training_data/beta_fits.parquet")

print(f"{len(fits)} maps fitted")
print(f"mean divergence: {fits['mean_div'].mean():.6f}")
print(f"max divergence:  {fits['mean_div'].max():.6f}")
print(f"maps with div > 0.01: {(fits['mean_div'] > 0.01).sum()}")

# scatter: fitted_mean vs precision, colored by log play count
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    fits["fitted_mean"], fits["precision"],
    c=np.log10(fits["n_scores"]), s=3, alpha=0.3, cmap="viridis",
)
ax.set_xlabel("fitted_mean")
ax.set_ylabel("precision (alpha + beta)")
ax.set_yscale("log")
plt.colorbar(sc, label="log10(n_scores)")
plt.tight_layout()
plt.savefig("./training_data/beta_scatter.png", dpi=150)
plt.show()

# diagnostic: worst fits
worst = fits.nlargest(6, "mean_div")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (_, row) in zip(axes.flat, worst.iterrows()):
    sub = df[df["map_idx"] == row["map_idx"]]
    acc = np.clip(sub["score_norm"].values, EPS, 1 - EPS)
    ax.hist(acc, bins=60, density=True, alpha=0.6, edgecolor="none")
    x = np.linspace(EPS, 1 - EPS, 200)
    ax.plot(x, beta_dist.pdf(x, row["alpha"], row["beta"]), "r-", lw=2)
    ax.set_title(
        f"map {int(row['map_idx'])} n={int(row['n_scores'])} div={row['mean_div']:.4f}")

plt.tight_layout()
plt.savefig("./training_data/beta_worst.png", dpi=150)
plt.show()
