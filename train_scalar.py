import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from tqdm import tqdm
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
PARQUET_PATH = "./training_data/train_final.parquet"
BATCH_SIZE = 4096      # Big batch size for stable updates
LEARNING_RATE = 0.01
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------


class OsuScalarDataset(Dataset):
    def __init__(self, path):
        print("Loading parquet file...")
        # Lazy scan to filter before loading into RAM
        q = (
            pl.scan_parquet(path)
            .filter(
                (pl.col("mods") == "") | (pl.col("mods") == "CL")
            )
            .select(["user_idx", "map_idx", "score_norm"])
        )

        # Collect into RAM
        self.df = q.collect()
        print(f"Loaded {len(self.df)} NoMod/CL scores.")

        # Get dimensions for Embedding layers
        self.n_users = self.df["user_idx"].max() + 1
        self.n_maps = self.df["map_idx"].max() + 1
        print(f"Dimensions: {self.n_users} Users x {self.n_maps} Maps")

        # Convert to Torch Tensors
        # We use .to_numpy() first to speed up conversion
        self.users = torch.tensor(
            self.df["user_idx"].to_numpy(), dtype=torch.long)
        self.maps = torch.tensor(
            self.df["map_idx"].to_numpy(), dtype=torch.long)
        self.targets = torch.tensor(
            self.df["score_norm"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.users[idx], self.maps[idx], self.targets[idx]

# ------------------------------------------------------------------
# 2. MODEL DEFINITION
# ------------------------------------------------------------------


class ScalarModel(nn.Module):
    def __init__(self, num_users, num_maps):
        super().__init__()
        # The "Skill" and "Difficulty" (1 dimension each)
        self.user_embedding = nn.Embedding(num_users, 1)
        self.map_embedding = nn.Embedding(num_maps, 1)

        # Biases capture "Global Player Ability" (e.g. some players are just better generally)
        # and "Global Map Easiness" (e.g. some maps are just score farms)
        self.user_bias = nn.Embedding(num_users, 1)
        self.map_bias = nn.Embedding(num_maps, 1)

        # Initialize weights to small random values to break symmetry
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.map_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.map_bias.weight)

    def forward(self, user_idx, map_idx):
        # 1. Fetch parameters
        u_vec = self.user_embedding(user_idx).squeeze()  # Shape: [Batch]
        m_vec = self.map_embedding(map_idx).squeeze()   # Shape: [Batch]

        u_b = self.user_bias(user_idx).squeeze()
        m_b = self.map_bias(map_idx).squeeze()

        # 2. The Interaction (Dot Product of scalars is just multiplication)
        dot = u_vec * m_vec

        # 3. Combine with biases
        logits = dot + u_b + m_b

        # 4. Squish to 0-1 range
        return torch.sigmoid(logits)

# ------------------------------------------------------------------
# 3. TRAINING LOOP
# ------------------------------------------------------------------


def train():
    # Load Data
    dataset = OsuScalarDataset(PARQUET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

    # Init Model
    model = ScalarModel(dataset.n_users, dataset.n_maps).to(DEVICE)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # Simple Mean Squared Error

    print("\n--- Starting Training ---")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for user_ids, map_ids, targets in pbar:
            # Move to GPU
            user_ids = user_ids.to(DEVICE)
            map_ids = map_ids.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward
            predictions = model(user_ids, map_ids)
            loss = criterion(predictions, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f}")

    # ------------------------------------------------------------------
    # 4. INSPECTION (Sanity Check)
    # ------------------------------------------------------------------
    print("\n--- Sanity Check ---")
    # Let's peek at the top 5 "highest skill" players according to the model
    # (Note: This is just looking at the bias term + embedding magnitude)

    # We grab the user bias weights
    u_biases = model.user_bias.weight.detach().cpu().numpy().flatten()

    # Get indices of top 5
    top_5_idx = np.argsort(u_biases)[-5:][::-1]

    import json
    # Load mappings to print real names/ids
    try:
        with open("./training_data/mappings_users.json", "r") as f:
            user_map = json.load(f)
            # Invert map: idx -> real_id
            idx_to_user = {v: k for k, v in user_map.items()}

        print("Top 5 Players (by learned bias):")
        for idx in top_5_idx:
            real_id = idx_to_user.get(idx, "Unknown")
            val = u_biases[idx]
            print(f"User ID {real_id}: {val:.4f}")
    except FileNotFoundError:
        print("Could not load mappings_users.json for pretty printing.")

    # Save model
    torch.save(model.state_dict(), "osu_scalar_model.pth")
    print("Model saved to osu_scalar_model.pth")


if __name__ == "__main__":
    train()
