# src/ncf.py
# ============================================================
# Neural Collaborative Filtering (NCF) Implementation
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os


# ------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------
class NCFDataset(Dataset):
    def __init__(self, df, user_col, item_col, rating_col, user_map, item_map):
        self.users = df[user_col].map(user_map).values
        self.items = df[item_col].map(item_map).values
        self.ratings = df[rating_col].values.astype("float32")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
        )


# ------------------------------------------------------------
# NCF Model (MLP-based)
# ------------------------------------------------------------
class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32, hidden_layers=[64, 32, 16, 8], dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # predicted rating probability
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, item_idx):
        u = self.user_embedding(user_idx)
        i = self.item_embedding(item_idx)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze()


# ------------------------------------------------------------
# NCF Wrapper for recommendation
# ------------------------------------------------------------
class NCFWrapper:
    def __init__(self, model, user_map, item_map, device="cpu"):
        self.model = model
        self.user_map = user_map
        self.item_map = item_map
        self.rev_item_map = {v: k for k, v in item_map.items()}
        self.device = device

    def recommend(self, user_id, top_k=10, k=None):
        if k is not None:
            top_k = k
        if user_id not in self.user_map:
            # fallback an toàn: random items
            return random.sample(list(self.item_map.keys()), min(top_k, len(self.item_map)))

        u_idx = torch.tensor([self.user_map[user_id]], dtype=torch.long).to(self.device)
        item_indices = list(self.item_map.values())
        i_idx = torch.tensor(item_indices, dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model(u_idx.repeat(len(i_idx)), i_idx).cpu().numpy()

        ranked = sorted(zip(item_indices, scores), key=lambda x: x[1], reverse=True)
        return [self.rev_item_map[i] for i, _ in ranked[:top_k]]


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_ncf(
    train_df,
    val_df,
    user_col,
    item_col,
    rating_col,
    embedding_dim=32,
    hidden_layers=[64, 32, 16, 8],
    dropout=0.2,
    batch_size=256,
    lr=0.001,
    epochs=20,
    device="cpu",
    verbose=True,
    patience=5,
    save_best=True,
    save_path="../models/ncf_best.pt"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Encode user và item IDs
    user_ids = train_df[user_col].unique()
    item_ids = train_df[item_col].unique()
    user_map = {u: idx for idx, u in enumerate(user_ids)}
    item_map = {i: idx for idx, i in enumerate(item_ids)}

    train_loader = DataLoader(
        NCFDataset(train_df, user_col, item_col, rating_col, user_map, item_map),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        NCFDataset(val_df, user_col, item_col, rating_col, user_map, item_map),
        batch_size=batch_size, shuffle=False
    )

    # Model + optimizer + loss
    model = NCF(len(user_ids), len(item_ids), embedding_dim, hidden_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            preds = model(users, items)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                preds = model(users, items)
                loss = criterion(preds, ratings)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        if verbose:
            print(f"[NCF] Epoch {epoch+1}/{epochs} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        # --- Early stopping check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "state_dict": model.state_dict(),
                "user_map": user_map,
                "item_map": item_map,
                "embedding_dim": embedding_dim,
                "hidden_layers": hidden_layers,
                "dropout": dropout
            }
            patience_counter = 0
            if save_best:
                torch.save(best_state, save_path)
                if verbose:
                    print(f"[NCF] Saved best model → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"[EarlyStopping] No improvement for {patience} epochs. Stopping at epoch {epoch+1}.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])

    return NCFWrapper(model, user_map, item_map, device), best_val_loss


# ------------------------------------------------------------
# Load function (tự động lấy cả mapping + config)
# ------------------------------------------------------------
def load_ncf_model(save_path, device="cpu"):
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    user_map = checkpoint["user_map"]
    item_map = checkpoint["item_map"]
    embedding_dim = checkpoint["embedding_dim"]
    hidden_layers = checkpoint["hidden_layers"]
    dropout = checkpoint["dropout"]

    model = NCF(len(user_map), len(item_map), embedding_dim, hidden_layers, dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return NCFWrapper(model, user_map, item_map, device)
