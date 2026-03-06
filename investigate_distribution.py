import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("./training_data/scores.parquet")
MAP_IDS = [1537566, 4869656, 667021, 4755346]

fig, axes = plt.subplots(len(MAP_IDS), 4, figsize=(20, 4 * len(MAP_IDS)))

for row, bid in enumerate(MAP_IDS):
    sub = df[df["beatmap_id"] == bid]
    n = len(sub)

    axes[row, 0].hist(sub["accuracy"], bins=80, edgecolor="none")
    axes[row, 0].set_xlabel("accuracy")
    axes[row, 0].set_ylabel(f"{bid}\n({n} scores)")

    axes[row, 1].hist(sub["score_norm"], bins=80, edgecolor="none")
    axes[row, 1].set_xlabel("score_norm")

    axes[row, 2].scatter(sub["accuracy"], sub["miss_count"], alpha=0.15, s=4)
    axes[row, 2].set_xlabel("accuracy")
    axes[row, 2].set_ylabel("miss_count")

    axes[row, 3].scatter(sub["accuracy"], sub["score_norm"], alpha=0.15, s=4)
    axes[row, 3].set_xlabel("accuracy")
    axes[row, 3].set_ylabel("score_norm")

for j, title in enumerate(["accuracy dist", "score_norm dist", "acc vs miss", "acc vs score_norm"]):
    axes[0, j].set_title(title)

plt.tight_layout()
plt.savefig("./training_data/distributions.png", dpi=150)
plt.show()
