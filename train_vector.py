from pathlib import Path
import torch
import torch.nn as nn
import polars as pl
import numpy as np
from tqdm import tqdm
import json
import os
import random

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DATA_DIR = Path("./training_data")
PARQUET_PATH = DATA_DIR / "train_final.parquet"
MAP_MAPPINGS_PATH = DATA_DIR / "mappings_maps.json"
USER_MAPPINGS_PATH = DATA_DIR / "mappings_users.json"

BATCH_SIZE = 8192           # NCF prefers slightly smaller batches than pure MF
LEARNING_RATE = 0.001       # Standard Adam LR
EPOCHS = 15                 # NCF converges/overfits faster
MF_DIM = 32                 # Dimension for Linear Physics
MLP_DIMS = [64, 32, 16]     # Dimensions for Deep Logic
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
            .filter(pl.col("score_norm").is_not_nan())
            .select(["user_idx", "map_idx", "score_norm"])
        )
        self.df = q.collect()

        print("Calculating sample weights...")
        user_counts = self.df.group_by("user_idx").count()
        self.df = self.df.join(user_counts, on="user_idx")

        counts = np.maximum(self.df["count"].to_numpy(), 1)
        weights_numpy = 1.0 / np.sqrt(counts)

        self.n_users = self.df["user_idx"].max() + 1
        self.n_items = self.df["map_idx"].max() + 1
        self.dataset_len = len(self.df)
        print(f"Dimensions: {self.n_users} Users x {self.n_items} Items")

        print(f"Moving data to {device}...")
        self.users = torch.tensor(
            self.df["user_idx"].to_numpy(), dtype=torch.int32, device=device)
        self.items = torch.tensor(
            self.df["map_idx"].to_numpy(), dtype=torch.int32, device=device)
        self.targets = torch.tensor(
            self.df["score_norm"].to_numpy(), dtype=torch.float32, device=device)
        self.weights = torch.tensor(
            weights_numpy, dtype=torch.float32, device=device)

        del self.df, weights_numpy

# ------------------------------------------------------------------
# 2. NEUMF MODEL DEFINITION
# ------------------------------------------------------------------


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=32, mlp_dims=[64, 32, 16], dropout=0.1):
        super().__init__()

        # --- 1. GMF Branch (Linear Physics) ---
        self.user_embedding_mf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_mf = nn.Embedding(num_items, mf_dim)

        # --- 2. MLP Branch (Non-Linear Nuance) ---
        # Note: Input to MLP is User_Vec + Item_Vec (Concatenated)
        self.user_embedding_mlp = nn.Embedding(num_users, mlp_dims[0] // 2)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_dims[0] // 2)

        layers = []
        in_dim = mlp_dims[0]

        for out_dim in mlp_dims[1:]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp_layers = nn.Sequential(*layers)

        # --- 3. Fusion ---
        # Input: MF_Output (mf_dim) + MLP_Output (last_dim)
        self.predict_layer = nn.Linear(mf_dim + mlp_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        # Small random init for embeddings
        nn.init.normal_(self.user_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std=0.01)
        nn.init.xavier_uniform_(self.user_embedding_mlp.weight)
        nn.init.xavier_uniform_(self.item_embedding_mlp.weight)

        # He Init for ReLU layers
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

        # LeCun Init for final layer (Linear)
        nn.init.kaiming_uniform_(
            self.predict_layer.weight, nonlinearity='linear')

    def forward(self, user_idx, item_idx):
        # GMF
        u_mf = self.user_embedding_mf(user_idx)
        i_mf = self.item_embedding_mf(item_idx)
        gmf_vec = u_mf * i_mf

        # MLP
        u_mlp = self.user_embedding_mlp(user_idx)
        i_mlp = self.item_embedding_mlp(item_idx)
        mlp_vec = torch.cat([u_mlp, i_mlp], dim=1)
        mlp_vec = self.mlp_layers(mlp_vec)

        # Fuse
        vector = torch.cat([gmf_vec, mlp_vec], dim=1)
        logits = self.predict_layer(vector)

        return torch.clamp(logits.squeeze(), min=-15.0, max=15.0)

# ------------------------------------------------------------------
# 3. ANALYSIS TOOLS (Discriminative Power)
# ------------------------------------------------------------------


def analyze_and_rate_users(model, dataset, device):
    print("\n" + "="*60)
    print("  ANALYZING SKILL WEIGHTS & USER RATINGS")
    print("="*60)

    # Load Mappings
    try:
        with open(MAP_MAPPINGS_PATH, "r") as f:
            idx_to_map = {v: k for k, v in json.load(f).items()}
        with open(USER_MAPPINGS_PATH, "r") as f:
            idx_to_user = {v: k for k, v in json.load(f).items()}
    except:
        idx_to_map = {}
        idx_to_user = {}

    # --- STEP 1: Find "High Weight" Maps ---
    # We define weight by Discrimination Variance.
    # If a map produces the same score for everyone, it has Low Weight.
    # If a map separates users (0.1 vs 0.9), it has High Weight.

    print("Calculating Discrimination Power (Map Weights)...")

    # 1. Sample 5000 random maps to analyze
    sample_size = 5000
    map_indices = torch.randint(
        0, dataset.n_items, (sample_size,), device=device)

    # 2. Sample 100 random users as a "Reference Population"
    ref_users = torch.randint(0, dataset.n_users, (100,), device=device)

    map_weights = {}  # map_idx -> variance

    with torch.no_grad():
        # Batch process maps to save time
        # We need to run (100 users) against (1 map) repeatedly
        # Efficient way: Broadcasitng is tricky with Embedding layers, so we loop batch.

        batch_maps = 100  # Process 100 maps at a time
        for i in range(0, sample_size, batch_maps):
            batch_m_idxs = map_indices[i: i+batch_maps]

            # Create pairs: (User1, Map1), (User2, Map1)... (User100, Map100)
            # Actually simplest is just loop the 100 maps
            for m_idx in batch_m_idxs:
                m_repeated = m_idx.repeat(100)
                logits = model(ref_users, m_repeated)
                scores = torch.sigmoid(logits)

                # Variance = Discrimination Power
                var = torch.var(scores).item()
                map_weights[m_idx.item()] = var

    # Sort maps by weight
    sorted_maps = sorted(map_weights.items(), key=lambda x: x[1], reverse=True)

    print(f"\n--- Top 5 'High Weight' Maps (Skill Checks) ---")
    for mid, var in sorted_maps[:5]:
        name = idx_to_map.get(mid, str(mid))
        print(f"Map {name:<20} | Weight: {var:.4f}")

    print(f"\n--- Top 5 'Low Weight' Maps (Farm/Generic) ---")
    for mid, var in sorted_maps[-5:]:
        name = idx_to_map.get(mid, str(mid))
        print(f"Map {name:<20} | Weight: {var:.4f}")

    # --- STEP 2: Rate Users based on High Weight Maps ---
    # We define "Underrated God" as someone who scores HIGH on High Weight maps.
    print("\nCalculating User Ratings...")

    # Get indices of top 100 high-weight maps
    top_map_indices = torch.tensor(
        [x[0] for x in sorted_maps[:100]], device=device)

    # Evaluate ALL users on these 100 specific maps
    # This simulates "How would you do on the hardest skill checks?"

    # Because N_Users is huge, we just sample top active users or randoms for demo
    # Let's do a sample of 2000 users
    sample_users = torch.randint(0, dataset.n_users, (2000,), device=device)

    user_ratings = []

    with torch.no_grad():
        for u_idx in tqdm(sample_users, desc="Rating Users"):
            # Repeat user 100 times to match the 100 maps
            u_repeated = u_idx.repeat(100)

            logits = model(u_repeated, top_map_indices)
            scores = torch.sigmoid(logits)

            # Rating = Mean Score on High Weight Maps
            # (You could also do weighted sum)
            rating = scores.mean().item() * 10000
            user_ratings.append((u_idx.item(), rating))

    # Sort
    user_ratings.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Top 5 'Underrated God' Players (Sampled) ---")
    for rank, (uid, rating) in enumerate(user_ratings[:5], 1):
        real_uid = idx_to_user.get(uid, str(uid))
        print(f"#{rank} | User {real_uid:<15} | Rating: {rating:.0f}")

# ------------------------------------------------------------------
# 4. TRAINING LOOP
# ------------------------------------------------------------------


def train():
    dataset = OsuVectorDataset(PARQUET_PATH, DEVICE)
    model = NeuMF(
        dataset.n_users,
        dataset.n_items,
        mf_dim=MF_DIM,
        mlp_dims=MLP_DIMS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    print(f"\n--- Starting NeuMF Training ---")

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
            batch_idx = shuffled_indices[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            # Forward
            logits = model(dataset.users[batch_idx], dataset.items[batch_idx])

            # Weighted Loss
            raw_loss = criterion(logits, dataset.targets[batch_idx])
            loss = (raw_loss * dataset.weights[batch_idx]).mean()

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients (Critical for NCF stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            curr_loss = loss.item()
            total_loss += curr_loss

            if i % 100 == 0:
                pbar.set_postfix({'loss': f"{curr_loss:.4f}"})

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/steps_per_epoch:.4f}")

    # Save
    torch.save(model.state_dict(), "osu_neumf_model.pth")

    # Run Analysis
    analyze_and_rate_users(model, dataset, DEVICE)


if __name__ == "__main__":
    train()
