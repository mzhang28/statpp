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

BATCH_SIZE = 2**16           # NCF prefers slightly smaller batches than pure MF
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

        # Inverse Sqrt Weighting to prevent power users from dominating the loss
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

        # Clamp logits to prevent infinity in loss (Sigmoid range -15 to 15 is sufficient)
        return torch.clamp(logits.squeeze(), min=-15.0, max=15.0)

# ------------------------------------------------------------------
# 3. ANALYSIS TOOLS (Discriminative Power)
# ------------------------------------------------------------------


def analyze_and_rate_users(model, dataset, device):
    print("\n" + "="*60)
    print("  ANALYZING SKILL WEIGHTS & USER RATINGS")
    print("="*60)

    # --- Load Nested Metadata JSONs ---
    print("Loading metadata mappings...")
    try:
        with open(MAP_MAPPINGS_PATH, "r") as f:
            raw_map_data = json.load(f)
        with open(USER_MAPPINGS_PATH, "r") as f:
            raw_user_data = json.load(f)

        # Invert User Map: Index -> "Name (ID)"
        # Structure: "123": {"idx": 0, "name": "Cookiezi"}
        idx_to_user = {}
        for real_id, data in raw_user_data.items():
            idx_to_user[data['idx']] = f"{data['name']} ({real_id})"

        # Invert Map Map: Index -> "Artist - Title [Diff] +Mods"
        # Structure: "1001|DT": {"idx": 0, "artist": "...", ...}
        idx_to_map = {}
        for key_str, data in raw_map_data.items():
            # key_str is "beatmap_id|mods" (e.g., "12345|HD,DT")
            parts = key_str.split("|")
            mods = parts[1] if len(parts) > 1 and parts[1] else ""
            mod_str = f"+{mods}" if mods else ""

            # Format: Artist - Title [Version] +MODS
            name = f"{data['artist']} - {data['title']} [{data['version']}] {mod_str}"
            idx_to_map[data['idx']] = name

    except Exception as e:
        print(
            f"Warning: Could not load metadata mappings ({e}). Using raw indices.")
        idx_to_map = {}
        idx_to_user = {}

    # --- STEP 1: Find "High Weight" Maps ---
    # We define weight by Discrimination Variance.

    #
    # We are looking for maps with steep slopes (High Variance in predictions).
    # High Variance means the map sharply distinguishes between "Good" and "Bad" players.

    print("Calculating Discrimination Power (Map Weights)...")

    sample_size = 5000
    map_indices = torch.randint(
        0, dataset.n_items, (sample_size,), device=device)
    ref_users = torch.randint(
        0, dataset.n_users, (100,), device=device)  # Reference population

    map_weights = {}  # map_idx -> variance

    model.eval()
    with torch.no_grad():
        # Loop over sample maps
        for m_idx in tqdm(map_indices, desc="Scanning Maps"):
            # Create pairs: (Ref_Users, Current_Map)
            m_repeated = m_idx.repeat(100)
            logits = model(ref_users, m_repeated)
            scores = torch.sigmoid(logits)

            # Variance = Discrimination Power
            var = torch.var(scores).item()
            map_weights[m_idx.item()] = var

    sorted_maps = sorted(map_weights.items(), key=lambda x: x[1], reverse=True)

    print(f"\n--- Top 5 'High Weight' Maps (Skill Checks) ---")
    for mid, var in sorted_maps[:5]:
        name = idx_to_map.get(mid, f"Map {mid}")
        # Truncate for display
        print(f"{name[:50]:<50} | Weight: {var:.4f}")

    print(f"\n--- Top 5 'Low Weight' Maps (Farm/Generic) ---")
    for mid, var in sorted_maps[-5:]:
        name = idx_to_map.get(mid, f"Map {mid}")
        print(f"{name[:50]:<50} | Weight: {var:.4f}")

    # --- STEP 2: Rate Users based on High Weight Maps ---
    print("\nCalculating User Ratings on Top 100 Skill Maps...")

    top_map_indices = torch.tensor(
        [x[0] for x in sorted_maps[:100]], device=device)

    # Sample 2000 users to rate (Rating everyone takes too long for a quick demo)
    sample_users = torch.randint(0, dataset.n_users, (2000,), device=device)

    user_ratings = []

    with torch.no_grad():
        for u_idx in tqdm(sample_users, desc="Rating Users"):
            u_repeated = u_idx.repeat(100)
            logits = model(u_repeated, top_map_indices)
            scores = torch.sigmoid(logits)

            # Rating = Sum of scores on Hard Maps * 100
            rating = scores.sum().item() * 100
            user_ratings.append((u_idx.item(), rating))

    user_ratings.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Top 10 'Underrated God' Players (Sampled) ---")
    for rank, (uid, rating) in enumerate(user_ratings[:10], 1):
        name = idx_to_user.get(uid, f"User {uid}")
        print(f"#{rank:<2} | {name:<30} | Rating: {rating:.0f}")

# ------------------------------------------------------------------
# 4. TRAINING LOOP
# ------------------------------------------------------------------


def train():
    dataset = OsuVectorDataset(PARQUET_PATH, DEVICE)
    model = NeuMF(dataset.n_users, dataset.n_items,
                  mf_dim=MF_DIM, mlp_dims=MLP_DIMS).to(DEVICE)

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

            logits = model(dataset.users[batch_idx], dataset.items[batch_idx])

            raw_loss = criterion(logits, dataset.targets[batch_idx])
            loss = (raw_loss * dataset.weights[batch_idx]).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            curr_loss = loss.item()
            total_loss += curr_loss

            if i % 100 == 0:
                pbar.set_postfix({'loss': f"{curr_loss:.4f}"})

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/steps_per_epoch:.4f}")

    torch.save(model.state_dict(), "osu_neumf_model.pth")
    analyze_and_rate_users(model, dataset, DEVICE)


if __name__ == "__main__":
    train()
