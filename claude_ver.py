import torch
import json
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from db import bits_to_mods, fetch_all_scores, mods_to_bits
from tqdm import tqdm

class OsuRatingSystem(nn.Module):
    def __init__(self, n_players, embedding_dim=64, device='mps'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        # Player embeddings
        self.player_embeddings = nn.Embedding(n_players, embedding_dim).to(device)

        # Map embeddings - use ParameterDict for (beatmap_id, mod_bits) pairs
        self.map_embeddings = nn.ParameterDict()

        # Initialize embeddings
        nn.init.normal_(self.player_embeddings.weight, std=0.1)

    def get_map_embedding(self, beatmap_id, mod_bits):
        key = f"{beatmap_id}_{mod_bits}"
        if key not in self.map_embeddings:
            self.map_embeddings[key] = nn.Parameter(
                torch.randn(self.embedding_dim, device=self.device) * 0.1
            )
        return self.map_embeddings[key]

    def forward(self, player_indices, beatmap_ids, mod_bits):
        player_emb = self.player_embeddings(player_indices)

        # Get map embeddings for each (beatmap_id, mod_bits) pair
        map_emb = torch.stack([
            self.get_map_embedding(int(bid), int(mb))
            for bid, mb in zip(beatmap_ids.cpu(), mod_bits.cpu())
        ])

        # Simple dot product for score prediction (higher = better performance)
        predicted_rating = (player_emb * map_emb).sum(dim=1)
        return predicted_rating

def train_rating_system(scores_data, epochs=100, lr=0.01, reg_weight=0.001):
    """
    scores_data: list of (player_id, beatmap_id, mod_bits, score) tuples
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Create player ID to index mapping
    unique_players = sorted(set(s[0] for s in scores_data))
    player_to_idx = {pid: idx for idx, pid in enumerate(unique_players)}
    idx_to_player = {idx: pid for pid, idx in player_to_idx.items()}

    # Initialize model
    model = OsuRatingSystem(len(unique_players), device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert data to tensors with mapped indices
    player_indices = torch.tensor([player_to_idx[s[0]] for s in scores_data], device=device)
    beatmap_ids = torch.tensor([s[1] for s in scores_data], device=device)
    mod_bits = torch.tensor([s[2] for s in scores_data], device=device)
    scores = torch.tensor([s[3] for s in scores_data], dtype=torch.float32, device=device)

    # Normalize scores
    scores = (scores - scores.mean()) / scores.std()

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Forward pass
        predictions = model(player_indices, beatmap_ids, mod_bits)

        # MSE loss
        loss = ((predictions - scores) ** 2).mean()

        # L2 regularization
        reg_loss = reg_weight * (
            model.player_embeddings.weight.pow(2).sum() +
            sum(emb.pow(2).sum() for emb in model.map_embeddings.values())
        )

        total_loss = loss + reg_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

    return model, idx_to_player

def get_rankings(model, idx_to_player):
    # Get player ratings
    player_ratings = model.player_embeddings.weight.norm(dim=1)
    top_player_indices = player_ratings.argsort(descending=True)[:5]

    print("\nTop 5 Players:")
    for i, idx in enumerate(top_player_indices):
        player_id = idx_to_player[idx.item()]
        rating = player_ratings[idx].item()
        print(f"{i+1}. Player {player_id}: {rating:.3f}")

    # Get map ratings
    map_ratings = {}
    for key, emb in model.map_embeddings.items():
        map_ratings[key] = emb.norm().item()

    # Sort maps by difficulty
    sorted_maps = sorted(map_ratings.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\nTop 5 Most Difficult Map/Mod Combinations:")
    for i, (key, rating) in enumerate(sorted_maps):
        beatmap_id, mod_bits = key.split('_')
        mods = bits_to_mods(int(mod_bits))
        print(f"{i+1}. Beatmap {beatmap_id} with mods {mods}: {rating:.3f}")

# Example usage
if __name__ == "__main__":
    scores = []
    for score in fetch_all_scores(10000):
        user_id, beatmap_id, accuracy, total_score, mods = score
        scores.append((
            user_id,
            beatmap_id,
            mods_to_bits(json.loads(mods)),
            total_score,
        ))

    model, idx_to_player = train_rating_system(scores)
    get_rankings(model, idx_to_player)
