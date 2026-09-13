import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Paths (EDIT ONLY IF NEEDED)
# -----------------------------
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "room_acoustic_largedataset")
EDC_DIR = os.path.join(DATASET_DIR, "EDC")
FEATURE_CSV = os.path.join(DATASET_DIR, "roomFeaturesDataset.csv")

EDC_LEN = 144000       # Original EDC length
DOWNSAMPLE = 2         # Optional downsampling factor
EDC_LEN_DS = EDC_LEN // DOWNSAMPLE  # Downsampled length

BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 50           # Reduced epochs for 3-hour run
MAX_FILES = 2500      # Use ~2500 files for faster computation

# -----------------------------
# Dataset
# -----------------------------
class EDCDataset(Dataset):
    def __init__(self, feature_csv, edc_dir, max_files=None, downsample=1):
        self.df = pd.read_csv(feature_csv)

        # Keep ONLY numeric feature columns
        self.feature_cols = (
            self.df.drop(columns=["ID"])
                   .select_dtypes(include=[np.number])
                   .columns
                   .tolist()
        )
        print(f"Using {len(self.feature_cols)} numeric features")

        self.edc_dir = edc_dir
        self.downsample = downsample

        # Randomly sample max_files rows if specified
        if max_files is not None and max_files < len(self.df):
            self.df = self.df.sample(n=max_files, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Input features
        x = torch.tensor(row[self.feature_cols].astype(np.float32).values, dtype=torch.float32)

        # ID already contains full EDC name
        edc_filename = f"{row['ID']}.npy"
        edc_path = os.path.join(self.edc_dir, edc_filename)
        if not os.path.exists(edc_path):
            raise FileNotFoundError(f"Missing EDC file: {edc_path}")

        y = np.load(edc_path).astype(np.float32)

        # Downsample if needed
        if self.downsample > 1:
            y = y[::self.downsample]

        return x, torch.from_numpy(y)

# -----------------------------
# TCN Block
# -----------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # causal cut
        return self.norm(self.relu(out))

# -----------------------------
# TCN Model
# -----------------------------
class TCN_EDC(pl.LightningModule):
    def __init__(self, input_dim, edc_len):
        super().__init__()
        self.save_hyperparameters()
        self.fc = nn.Linear(input_dim, 128)
        self.tcn = nn.Sequential(
            TCNBlock(128, 128, 7, 1),
            TCNBlock(128, 128, 7, 2),
            TCNBlock(128, 128, 7, 4),
            TCNBlock(128, 128, 7, 8),
            TCNBlock(128, 128, 7, 16),
        )
        self.out = nn.Conv1d(128, 1, kernel_size=1)
        self.loss_fn = nn.MSELoss()
        self.edc_len = edc_len

    def forward(self, x):
        x = self.fc(x)                        # (B, 128)
        x = x.unsqueeze(-1).repeat(1, 1, self.edc_len)  # (B, 128, T)
        x = self.tcn(x)
        y = self.out(x).squeeze(1)            # (B, T)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: loss = {loss.item():.6f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return {"y_hat": y_hat, "y": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    dataset = EDCDataset(FEATURE_CSV, EDC_DIR, max_files=MAX_FILES, downsample=DOWNSAMPLE)

    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    input_dim = dataset[0][0].shape[0]
    model = TCN_EDC(input_dim, EDC_LEN_DS)

    # TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="tcn_edc")

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,
        precision=16,
        default_root_dir="Models",
        log_every_n_steps=10,
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    # Save a few validation predictions
    x_val, y_val = next(iter(val_loader))
    y_hat = model(x_val)
    np.save("sample_predictions.npy", y_hat.detach().cpu().numpy())
    print("Sample predictions saved to 'sample_predictions.npy'")
