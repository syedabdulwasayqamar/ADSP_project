# adsp_project/prepare_features.py

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
DATASET_DIR = os.path.join("dataset", "room_acoustic_largedataset")
FEATURES_CSV = os.path.join(DATASET_DIR, "roomFeaturesDataset.csv")
EDC_DIR = os.path.join(DATASET_DIR, "EDC")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Your CSV has 6000 rows and 17 columns.
    The first column is a string ID (e.g. 'RIR_001_case0_edc'),
    the remaining columns are numeric.

    We:
    - keep only numeric columns (drop the string ID)
    - assume the 16 numeric cols are:
        0-2 : room dimensions [L, W, H]
        3-5 : source position  [sx, sy, sz]
        6-8 : receiver position[rx, ry, rz]
        9-15: absorption bands (7 values)
    """

    # 1) Keep only numeric columns (drops the string ID column)
    df_num = df.select_dtypes(include=["number"]).copy()
    print("Numeric columns (used for features):", df_num.columns.tolist())
    print("Numeric shape:", df_num.shape)

    if df_num.shape[1] < 16:
        raise ValueError(
            f"Expected at least 16 numeric columns (got {df_num.shape[1]}). "
            "Check roomFeaturesDataset.csv format."
        )

    # 2) Basic slices by numeric position
    dims = df_num.iloc[:, 0:3].to_numpy(dtype=np.float32)   # L, W, H
    src = df_num.iloc[:, 3:6].to_numpy(dtype=np.float32)    # sx, sy, sz
    rec = df_num.iloc[:, 6:9].to_numpy(dtype=np.float32)    # rx, ry, rz
    abs_vals = df_num.iloc[:, 9:16].to_numpy(dtype=np.float32)  # 7 bands

    L = dims[:, 0]
    W = dims[:, 1]
    H = dims[:, 2]

    volume = L * W * H
    surface_area = 2 * (L * W + L * H + W * H)

    eps = 1e-6
    ratio_L_W = L / (W + eps)
    ratio_L_H = L / (H + eps)
    ratio_W_H = W / (H + eps)

    # src–rec distance
    src_rec_dist = np.linalg.norm(src - rec, axis=1)

    # absorption stats
    abs_mean = abs_vals.mean(axis=1)
    abs_min = abs_vals.min(axis=1)
    abs_max = abs_vals.max(axis=1)

    # low/mid/high band groupings
    abs_low = abs_vals[:, 0:2].mean(axis=1)
    abs_mid = abs_vals[:, 2:5].mean(axis=1)
    abs_high = abs_vals[:, 5:7].mean(axis=1)

    # Concatenate all features into one matrix
    features = np.concatenate(
        [
            dims,           # 3
            src,            # 3
            rec,            # 3
            abs_vals,       # 7
            volume[:, None],
            surface_area[:, None],
            ratio_L_W[:, None],
            ratio_L_H[:, None],
            ratio_W_H[:, None],
            src_rec_dist[:, None],
            abs_mean[:, None],
            abs_min[:, None],
            abs_max[:, None],
            abs_low[:, None],
            abs_mid[:, None],
            abs_high[:, None],
        ],
        axis=1,
    )

    return features.astype(np.float32)


def load_edcs() -> np.ndarray:
    """Load all EDC .npy files and stack along axis 0."""
    edc_files = sorted(glob.glob(os.path.join(EDC_DIR, "*.npy")))
    if not edc_files:
        raise FileNotFoundError(f"No .npy files found in {EDC_DIR}")

    edcs = [np.load(f) for f in edc_files]
    Y = np.stack(edcs, axis=0).astype(np.float32)
    print(f"Loaded EDCs: {Y.shape[0]} rooms, {Y.shape[1]} time samples each.")
    return Y


def main():
    # 1) Load CSV
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"CSV not found at: {FEATURES_CSV}")

    df_raw = pd.read_csv(FEATURES_CSV)
    print(f"Loaded room features CSV with shape: {df_raw.shape}")

    # 2) Build feature matrix from numeric columns
    X = build_feature_matrix(df_raw)

    # 3) Load EDC targets
    Y = load_edcs()

    # 4) Check counts
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Number of feature rows ({X.shape[0]}) != number of EDC files ({Y.shape[0]}).\n"
            "Make sure you have the full EDC dataset matching the CSV."
        )

    # 5) Train/val/test split: 70/15/15
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.30, random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.50, random_state=42
    )

    print("Split sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

    # 6) Normalize X, keep Y raw
    scaler_X = StandardScaler()
    X_train_n = scaler_X.fit_transform(X_train)
    X_val_n = scaler_X.transform(X_val)
    X_test_n = scaler_X.transform(X_test)

    # 7) Save
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train_n)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val_n)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test_n)

    np.save(os.path.join(PROCESSED_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(PROCESSED_DIR, "Y_val.npy"), Y_val)
    np.save(os.path.join(PROCESSED_DIR, "Y_test.npy"), Y_test)

    joblib.dump(scaler_X, os.path.join(PROCESSED_DIR, "scaler_X_adsp.save"))

    print(f"Processed data saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
