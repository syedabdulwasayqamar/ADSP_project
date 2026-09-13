# adsp_project/train_transformer.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from adsp_project.transformer_edc import TransformerEDC

PROCESSED_DIR = os.path.join("data", "processed")
CHECKPOINT_DIR = os.path.join("adsp_project", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# EDC length is 96000 samples; 96000 / 240 = 400
DOWNSAMPLE_FACTOR = 240

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed(processed_dir: str, downsample_factor: int):
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))

    Y_train = np.load(os.path.join(processed_dir, "Y_train.npy"))
    Y_val = np.load(os.path.join(processed_dir, "Y_val.npy"))
    Y_test = np.load(os.path.join(processed_dir, "Y_test.npy"))

    # Downsample EDC along time axis
    Y_train_ds = Y_train[:, ::downsample_factor]
    Y_val_ds = Y_val[:, ::downsample_factor]
    Y_test_ds = Y_test[:, ::downsample_factor]

    print("Shapes after loading & downsampling:")
    print(f"  X_train: {X_train.shape}, Y_train_ds: {Y_train_ds.shape}")
    print(f"  X_val:   {X_val.shape},   Y_val_ds:   {Y_val_ds.shape}")
    print(f"  X_test:  {X_test.shape},  Y_test_ds:  {Y_test_ds.shape}")

    return (X_train, Y_train_ds), (X_val, Y_val_ds), (X_test, Y_test_ds)


def make_loaders(X_train, Y_train, X_val, Y_val, batch_size):
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train():
    set_seed(SEED)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_processed(
        PROCESSED_DIR, DOWNSAMPLE_FACTOR
    )

    feature_dim = X_train.shape[1]
    seq_len = Y_train.shape[1]

    train_loader, val_loader = make_loaders(
        X_train, Y_train, X_val, Y_val, BATCH_SIZE
    )

    model = TransformerEDC(
        feature_dim=feature_dim,
        seq_len=seq_len,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    print(f"Using device: {DEVICE}")
    print(f"Feature dim: {feature_dim}, seq_len: {seq_len}")
    print(model)

    best_val = float("inf")
    best_epoch = -1
    patience = 8
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss_sum, n_train = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

        train_loss = train_loss_sum / max(n_train, 1)

        # ---- Validate ----
        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                bs = xb.size(0)
                val_loss_sum += loss.item() * bs
                n_val += bs

        val_loss = val_loss_sum / max(n_val, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6e} | "
            f"Val Loss: {val_loss:.6e}"
        )

        # ---- Early stopping + save best ----
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0

            ckpt_path = os.path.join(CHECKPOINT_DIR, "transformer_edc_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": feature_dim,
                    "seq_len": seq_len,
                },
                ckpt_path,
            )
            print(f"  👉 New best model saved to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best val: {best_val:.6e})"
                )
                break

    print("Training finished.")
    print(f"Best epoch: {best_epoch}, Best val loss: {best_val:.6e}")


if __name__ == "__main__":
    train()
