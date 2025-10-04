# src/sasrec.py
# ============================================================
# SASRec: Self-Attentive Sequential Recommendation
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os

# ------------------------------------------------------------
# Utils: device selection
# ------------------------------------------------------------
def get_device(device=None):
    """Chọn thiết bị tự động (CUDA nếu có, không thì CPU)."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# ------------------------------------------------------------
# Dataset wrapper cho SASRec
# ------------------------------------------------------------
class SASRecDataset(Dataset):
    def __init__(self, user_sequences, item_map, max_len=50):
        self.samples = []
        self.item_map = item_map
        self.max_len = max_len

        for user, seq in user_sequences.items():
            mapped = [item_map[i] for i in seq if i in item_map]
            for i in range(1, len(mapped)):
                seq_input = mapped[max(0, i - max_len):i]
                target = mapped[i]
                self.samples.append((seq_input, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        seq = seq[-self.max_len:]
        padding = [0] * (self.max_len - len(seq))
        seq = padding + seq
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# ------------------------------------------------------------
# SASRec Model
# ------------------------------------------------------------
class SASRec(nn.Module):
    def __init__(self, n_items, embed_dim=64, n_layers=2, n_heads=2, dropout=0.2, max_len=50):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc_out = nn.Linear(embed_dim, n_items + 1)
        self.max_len = max_len

    def forward(self, seq):
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        x = self.item_embedding(seq) + self.pos_embedding(positions)
        x = self.transformer(x)
        out = self.fc_out(x[:, -1, :])  # chỉ lấy bước cuối
        return out

# ------------------------------------------------------------
# Wrapper để recommend
# ------------------------------------------------------------
class SASRecWrapper:
    def __init__(self, model, item_map, user_histories=None, device="cpu", max_len=50):
        self.model = model
        self.item_map = item_map
        self.rev_item_map = {v: k for k, v in item_map.items()}
        self.device = device
        self.max_len = max_len
        self.user_histories = user_histories if user_histories is not None else {}

    def recommend(self, user_id, top_k=10, k=None):
        if k is not None:
            top_k = k

        history = self.user_histories.get(user_id, [])
        if not history:
            return random.sample(list(self.item_map.keys()), min(top_k, len(self.item_map)))

        mapped = [self.item_map[i] for i in history if i in self.item_map]
        seq = mapped[-self.max_len:]
        padding = [0] * (self.max_len - len(seq))
        seq = padding + seq
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_tensor)
            scores = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        ranked = sorted(
            [(i, s) for i, s in enumerate(scores) if i != 0],
            key=lambda x: x[1], reverse=True
        )
        return [self.rev_item_map[i] for i, _ in ranked[:top_k]]

# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_sasrec(
    user_sequences,
    item_map,
    embed_dim=64,
    n_layers=2,
    n_heads=2,
    dropout=0.2,
    lr=0.001,
    batch_size=64,
    epochs=10,
    max_len=50,
    device=None,
    verbose=True,
    save_best=True,
    save_path="../models/sasrec_best.pt",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = get_device(device)

    dataset = SASRecDataset(user_sequences, item_map, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SASRec(len(item_map), embed_dim, n_layers, n_heads, dropout, max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, target in loader:
            seq, target = seq.to(device), target.to(device)
            logits = model(seq)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if verbose:
            print(f"[SASRec] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            if save_best:
                torch.save({
                    "state_dict": model.state_dict(),
                    "item_map": item_map,
                    "embed_dim": embed_dim,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "dropout": dropout,
                    "max_len": max_len,
                }, save_path)
                if verbose:
                    print(f"[SASRec] Saved best model → {save_path}")

    return SASRecWrapper(model, item_map, user_histories=user_sequences, device=device, max_len=max_len), best_val_loss

# ------------------------------------------------------------
# Load function
# ------------------------------------------------------------
def load_sasrec_model(save_path, item_map, user_histories, device=None):
    device = get_device(device)
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model = SASRec(
        n_items=len(item_map),
        embed_dim=checkpoint["embed_dim"],
        n_layers=checkpoint["n_layers"],
        n_heads=checkpoint["n_heads"],
        dropout=checkpoint["dropout"],
        max_len=checkpoint["max_len"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return SASRecWrapper(model, item_map, user_histories=user_histories, device=device, max_len=checkpoint["max_len"])
