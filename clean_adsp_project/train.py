# train_transformer_edc.py
#
# EDC Transformer training script
# Stores outputs in a separate folder and fully uses GPU + mixed precision

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from models.transformer_model import EDCTransformer
from utils.data_utils import load_dataset
from utils.preprocessing import scale_and_save

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CONFIG = {
    # Paths to your original data (keep large files where they are)
    "csv_path": r"D:/Stuff/TU Ilmenau/Masters In Media Engineering/ADSP/LSTM-Model-Energy-Decay-Curves-main/LSTM-Model-Energy-Decay-Curves-main/dataset/room_acoustic_largedataset/roomFeaturesDataset.csv",
    "edc_dir": r"D:/Stuff/TU Ilmenau/Masters In Media Engineering/ADSP/LSTM-Model-Energy-Decay-Curves-main/LSTM-Model-Energy-Decay-Curves-main/dataset/room_acoustic_largedataset/EDC",

    # Where to store outputs (checkpoints, scalers)
    "output_dir": r"D:/Stuff/TU Ilmenau/Masters In Media Engineering/ADSP/LSTM-Model-Energy-Decay-Curves-main/LSTM-Model-Energy-Decay-Curves-main/EDC_Training",

    # Target type
    "target_type": "edc",  # always EDC here

    # Training
    "test_size": 0.2,
    "random_state": 42,
    "batch_size": 32,
    "max_epochs": 50,
    "learning_rate": 1e-4,

    # DataLoader
    "num_workers": 4,  # increase if your PC can handle
    "pin_memory": True,
}


def main():
    cfg = CONFIG

    # -----------------------------------------------------
    # 1. Resolve paths
    # -----------------------------------------------------
    csv_path = Path(cfg["csv_path"])
    edc_dir = Path(cfg["edc_dir"])
    output_dir = Path(cfg["output_dir"])
    scaler_dir = output_dir / "scalers"
    checkpoint_dir = output_dir / "checkpoints"

    scaler_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 2. Load EDC dataset
    # -----------------------------------------------------
    print("▶ Loading EDC dataset...")
    X, y = load_dataset(
        csv_path=csv_path,
        edc_folder=edc_dir,
        target_type="edc",
    )
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # -----------------------------------------------------
    # 3. Scale features & targets
    # -----------------------------------------------------
    print("▶ Scaling and saving scalers...")
    X_scaled, y_scaled = scale_and_save(
        X=X,
        y=y,
        scaler_dir=scaler_dir,
        target_type="edc",
    )

    # -----------------------------------------------------
    # 4. Train/Validation split
    # -----------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        shuffle=True,
    )
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Val samples:   {X_val.shape[0]}")

    # -----------------------------------------------------
    # 5. DataLoaders
    # -----------------------------------------------------
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg["batch_size"],
                              shuffle=True,
                              num_workers=cfg["num_workers"],
                              pin_memory=cfg["pin_memory"])
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg["batch_size"],
                            shuffle=False,
                            num_workers=cfg["num_workers"],
                            pin_memory=cfg["pin_memory"])

    # -----------------------------------------------------
    # 6. Model
    # -----------------------------------------------------
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    print(f"▶ Creating Transformer | input_dim={input_dim} | output_dim={output_dim}")

    model = EDCTransformer(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        output_dim=output_dim,
        lr=cfg["learning_rate"],
    )

    # -----------------------------------------------------
    # 7. Trainer + checkpointing
    # -----------------------------------------------------
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"▶ Using accelerator: {accelerator}")

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="transformer-edc-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator=accelerator,
        devices=1,
        callbacks=[ckpt_callback],
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32,  # mixed precision on GPU
    )

    # -----------------------------------------------------
    # 8. Train
    # -----------------------------------------------------
    print("▶ Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("✔ Training finished.")
    print(f"✔ Best checkpoint saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
