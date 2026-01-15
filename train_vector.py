from pathlib import Path
import torch
import torch.nn as nn
import polars as pl
import numpy as np
from tqdm import tqdm
import json
import os
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DATA_DIR = Path("./training_data")
PARQUET_PATH = DATA_DIR / "train_final.parquet"
MAP_MAPPINGS_PATH = DATA_DIR / "mappings_maps.json"
USER_MAPPINGS_PATH = DATA_DIR / "mappings_users.json"

BATCH_SIZE = 65536
LEARNING_RATE = 0.001
EPOCHS = 5
EMBEDDING_DIM = 16
DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------


class OsuVectorDataset:
    def __init__(self, path, device):
        print("Loading parquet file...")
        q = (
            pl.scan_parquet(path)
            # CHANGE: Removed the mod filter!
            # We now WANT to train on different mod combinations as distinct items.
            .filter(pl.col("score_norm").is_not_nan())
            .select(["user_idx", "map_idx", "score_norm"])
        )
        self.df = q.collect()

        print("Calculating sample weights...")
        user_counts = self.df.group_by("user_idx").count()
        self.df = self.df.join(user_counts, on="user_idx")

        # Clip counts to min 1 to avoid divide by zero
        counts = np.maximum(self.df["count"].to_numpy(), 1)
        weights_numpy = 1.0 / np.sqrt(counts)

        self.n_users = self.df["user_idx"].max() + 1
        self.n_maps = self.df["map_idx"].max() + 1
        self.dataset_len = len(self.df)
        print(
            f"Dimensions: {self.n_users} Users x {self.n_maps} Items (Map+Mods)")

        print(f"Moving data to {device}...")
        self.users = torch.tensor(
            self.df["user_idx"].to_numpy(), dtype=torch.int32, device=device)
        self.maps = torch.tensor(
            self.df["map_idx"].to_numpy(), dtype=torch.int32, device=device)
        self.targets = torch.tensor(
            self.df["score_norm"].to_numpy(), dtype=torch.float32, device=device)
        self.weights = torch.tensor(
            weights_numpy, dtype=torch.float32, device=device)

        if torch.isnan(self.targets).any() or torch.isnan(self.weights).any():
            raise ValueError(
                "NaNs found in input data! Check your parquet generation.")

        del self.df, weights_numpy

# ------------------------------------------------------------------
# 2. MODEL DEFINITION
# ------------------------------------------------------------------


class VectorModel(nn.Module):
    def __init__(self, num_users, num_maps, dim=16):
        super().__init__()
        self.user_vectors = nn.Embedding(num_users, dim)
        self.map_vectors = nn.Embedding(num_maps, dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.map_diff = nn.Embedding(num_maps, 1)

        # Initialize small to prevent explosion
        nn.init.normal_(self.user_vectors.weight, std=0.01)
        nn.init.normal_(self.map_vectors.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.map_diff.weight)

    def forward(self, user_idx, map_idx):
        u_vec = self.user_vectors(user_idx)
        m_vec = self.map_vectors(map_idx)
        u_b = self.user_bias(user_idx).squeeze()
        m_d = self.map_diff(map_idx).squeeze()

        interaction = (u_vec * m_vec).sum(dim=1)
        logits = interaction + u_b - m_d

        return torch.clamp(logits, min=-15.0, max=15.0)

# ------------------------------------------------------------------
# 3. ANALYSIS TOOLS (Updated for String Keys)
# ------------------------------------------------------------------


def run_pca_analysis(model):
    print("\n" + "="*60)
    print("  PCA INTERPRETABILITY ANALYSIS")
    print("="*60)

    # 1. Load Mappings
    try:
        with open(MAP_MAPPINGS_PATH, "r") as f:
            beatmap_map = json.load(f)
            # CHANGE: Values are now strings like "1234|HD,DT". Do NOT cast to int.
            idx_to_map_id = {v: k for k, v in beatmap_map.items()}
    except FileNotFoundError:
        print("Warning: mappings_maps.json not found.")
        idx_to_map_id = {}

    try:
        with open(USER_MAPPINGS_PATH, "r") as f:
            user_map = json.load(f)
            idx_to_user_id = {v: k for k, v in user_map.items()}
    except FileNotFoundError:
        print("Warning: mappings_users.json not found.")
        idx_to_user_id = {}

    # 2. Extract Vectors
    map_vecs = model.map_vectors.weight.detach().cpu().numpy()
    user_vecs = model.user_vectors.weight.detach().cpu().numpy()

    if np.isnan(map_vecs).any():
        print("Error: Map vectors contain NaN.")
        return

    # 3. Fit PCA
    pca = PCA(n_components=EMBEDDING_DIM)
    map_pcs = pca.fit_transform(map_vecs)
    user_pcs = pca.transform(user_vecs)
    var_ratios = pca.explained_variance_ratio_

    # 4. Print Results (Widened columns for mod strings)
    print(f"\n{'PC':<3} | {'Var %':<6} | {'High Map (+)':<25} | {'Low Map (-)':<25} | {'High User (+)':<15}")
    print("-" * 90)

    for dim in range(min(5, EMBEDDING_DIM)):
        top_map_idx = np.argmax(map_pcs[:, dim])
        bot_map_idx = np.argmin(map_pcs[:, dim])

        # Keys are now strings like "1005|DT"
        top_map_real = idx_to_map_id.get(top_map_idx, str(top_map_idx))
        bot_map_real = idx_to_map_id.get(bot_map_idx, str(bot_map_idx))

        top_user_idx = np.argmax(user_pcs[:, dim])
        top_user_real = idx_to_user_id.get(top_user_idx, str(top_user_idx))

        print(
            f"{dim:<3} | {var_ratios[dim]*100:5.1f}% | {top_map_real:<25} | {bot_map_real:<25} | {top_user_real:<15}")

# ------------------------------------------------------------------
# 4. TRAINING LOOP
# ------------------------------------------------------------------


def train():
    dataset = OsuVectorDataset(PARQUET_PATH, DEVICE)
    model = VectorModel(dataset.n_users, dataset.n_maps,
                        dim=EMBEDDING_DIM).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    print(f"\n--- Starting Stable Training ---")

    indices = torch.arange(dataset.dataset_len,
                           device=DEVICE, dtype=torch.int64)
    steps_per_epoch = (dataset.dataset_len + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        shuffled_indices = indices[torch.randperm(
            dataset.dataset_len, device=DEVICE)]
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i in pbar:
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            batch_idx = shuffled_indices[start:end]

            logits = model(dataset.users[batch_idx], dataset.maps[batch_idx])

            raw_loss = criterion(logits, dataset.targets[batch_idx])
            weighted_loss = (raw_loss * dataset.weights[batch_idx]).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            curr_loss = weighted_loss.item()
            if np.isnan(curr_loss):
                print("LOSS IS NAN - STOPPING")
                return

            total_loss += curr_loss
            if i % 100 == 0:
                pbar.set_postfix({'loss': f"{curr_loss:.4f}"})

        print(f"Epoch {epoch+1} Avg Loss: {total_loss / steps_per_epoch:.4f}")

    run_pca_analysis(model)
    torch.save(model.state_dict(), "osu_vector_model.pth")


if __name__ == "__main__":
    train()
