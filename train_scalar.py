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
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------


class OsuScalarDataset(Dataset):
    def __init__(self, path):
        print("Loading parquet file...")
        # 1. Load Data
        q = (
            pl.scan_parquet(path)
            .filter((pl.col("mods") == "") | (pl.col("mods") == "CL"))
            .select(["user_idx", "map_idx", "score_norm"])
        )
        self.df = q.collect()

        # 2. Calculate User Play Counts for Weighting
        # "Who are the farmers?"
        print("Calculating sample weights...")
        user_counts = self.df.group_by("user_idx").count()

        # Join counts back to the main dataframe
        self.df = self.df.join(user_counts, on="user_idx")

        # 3. Compute Weight: 1.0 / sqrt(play_count)
        # We use sqrt to dampen the effect.
        # (1/N is too aggressive, 1/1 is too biased. Sqrt is the sweet spot)
        weights_numpy = 1.0 / np.sqrt(self.df["count"].to_numpy())

        self.n_users = self.df["user_idx"].max() + 1
        self.n_maps = self.df["map_idx"].max() + 1

        # 4. Tensors
        self.users = torch.tensor(
            self.df["user_idx"].to_numpy(), dtype=torch.long)
        self.maps = torch.tensor(
            self.df["map_idx"].to_numpy(), dtype=torch.long)
        self.targets = torch.tensor(
            self.df["score_norm"].to_numpy(), dtype=torch.float32)
        # The new Weight Tensor
        self.weights = torch.tensor(weights_numpy, dtype=torch.float32)

    def __getitem__(self, idx):
        # Return the weight along with the data
        return self.users[idx], self.maps[idx], self.targets[idx], self.weights[idx]

    def __len__(self):
        return len(self.df)

# ------------------------------------------------------------------
# 2. MODEL DEFINITION
# ------------------------------------------------------------------


class ScalarModel(nn.Module):
    def __init__(self, num_users, num_maps):
        super().__init__()
        # PURE IRT MODEL: Skill - Difficulty
        # We remove the biases because they are mathematically redundant in this formulation
        self.user_skill = nn.Embedding(num_users, 1)
        self.map_diff = nn.Embedding(num_maps, 1)

        # Initialize Skill to 0 and Difficulty to 0
        nn.init.normal_(self.user_skill.weight, mean=0, std=0.1)
        nn.init.normal_(self.map_diff.weight, mean=0, std=0.1)

    def forward(self, user_idx, map_idx):
        s = self.user_skill(user_idx).squeeze()
        d = self.map_diff(map_idx).squeeze()

        # The Core Logic: Skill minus Difficulty
        # If Skill >> Diff, result is huge positive -> Score = 1.0
        # If Skill << Diff, result is huge negative -> Score = 0.0
        logits = s - d

        return torch.sigmoid(logits)


def inspect(model):
    # Let's peek at the top 5 "highest skill" players according to the model
    # (Note: This is just looking at the bias term + embedding magnitude)

    # We grab the user bias weights
    u_biases = model.user_skill.weight.detach().cpu().numpy().flatten()

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

    print("\n--- Map Difficulty Analysis ---")

    # 1. Load the map ID translation (Dense Index -> Real Beatmap ID)
    try:
        with open("./training_data/mappings_maps.json", "r") as f:
            beatmap_map = json.load(f)
            # Invert: {12345: 0} -> {0: 12345}
            idx_to_id = {v: int(k) for k, v in beatmap_map.items()}
    except FileNotFoundError:
        print("Error: Could not find mappings_maps.json")
        return

    # 2. Extract learned weights
    # Shape: (Num_Maps,)
    diffs = model.map_diff.weight.detach().cpu().numpy().flatten()

    # 3. Sort indices
    # argsort gives [Index of Lowest, ..., Index of Highest]
    sorted_idx = np.argsort(diffs)

    # --- Helper to print ---
    def print_maps(indices, label):
        print(f"\n{label}:")
        print(f"{'Diff Rating':<12} | {'Beatmap ID':<12} | {'Map Index'}")
        print("-" * 40)
        for idx in indices:
            real_id = idx_to_id.get(idx, -1)
            val = diffs[idx]
            print(f"{val:>11.4f}  | {real_id:<12} | {idx}")

    # 4. Show Extremes
    # Top 10 Hardest (End of the sorted list, reversed)
    hardest_indices = sorted_idx[-10:][::-1]
    print_maps(hardest_indices, "TOP 10 HARDEST MAPS (High Positive Diff)")

    # Top 10 Easiest (Start of the sorted list)
    easiest_indices = sorted_idx[:10]
    print_maps(easiest_indices, "TOP 10 EASIEST MAPS (Low Negative Diff)")

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
    criterion = nn.MSELoss(reduction='none')  # Simple Mean Squared Error

    print("\n--- Starting Training ---")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for user_ids, map_ids, targets, weights in pbar:
            # Move to GPU
            user_ids, map_ids = user_ids.to(DEVICE), map_ids.to(DEVICE)
            targets, weights = targets.to(DEVICE), weights.to(DEVICE)

            predictions = model(user_ids, map_ids)

            # Calculate raw squared error
            raw_loss = criterion(predictions, targets)

            # Apply the "Anti-Farmer" Weighting
            weighted_loss = (raw_loss * weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()
            pbar.set_postfix({'loss': f"{weighted_loss.item():.6f}"})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f}")
        inspect(model)

    # ------------------------------------------------------------------
    # 4. INSPECTION (Sanity Check)
    # ------------------------------------------------------------------
    # Save model
    torch.save(model.state_dict(), "osu_scalar_model.pth")
    print("Model saved to osu_scalar_model.pth")


if __name__ == "__main__":
    train()
