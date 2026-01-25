from pathlib import Path
import torch
import torch.nn as nn
import polars as pl
from tqdm import tqdm

MODEL_PATH = Path("./osu_mf_model.pth")
PARQUET_PATH = Path("./training_data/train_final.parquet")

BATCH_SIZE = 2**18
LEARNING_RATE = 0.005
EPOCHS = 20
EMBEDDING_DIM = 32

DEVICE = "mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class SimpleMF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, dim)
        self.item_embedding = nn.Embedding(num_items, dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        interaction = (user_vec * item_vec).sum(dim=1)
        bias_user = self.user_bias(user_indices).squeeze()
        bias_item = self.item_bias(item_indices).squeeze()

        return interaction + bias_user + bias_item + self.global_bias


def load_data():
    df = pl.read_parquet(PARQUET_PATH).select(
        ["user_idx", "map_idx", "score_norm"])

    users = torch.tensor(df["user_idx"].to_numpy(),
                         dtype=torch.long, device=DEVICE)
    items = torch.tensor(df["map_idx"].to_numpy(),
                         dtype=torch.long, device=DEVICE)
    targets = torch.tensor(df["score_norm"].to_numpy(),
                           dtype=torch.float32, device=DEVICE)

    num_users = df["user_idx"].max() + 1
    num_items = df["map_idx"].max() + 1

    return users, items, targets, num_users, num_items


def train():
    users, items, targets, num_users, num_items = load_data()
    dataset_size = len(targets)

    model = SimpleMF(num_users, num_items, EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    indices = torch.arange(dataset_size, device=DEVICE)
    steps = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        shuffled_indices = indices[torch.randperm(dataset_size, device=DEVICE)]

        for i in tqdm(range(steps), desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_idx = shuffled_indices[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            predictions = model(users[batch_idx], items[batch_idx])
            loss = criterion(predictions, targets[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Avg Loss: {total_loss / steps:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    train()
