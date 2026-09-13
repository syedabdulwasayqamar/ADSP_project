# utils/data_utils.py
#
# Complete, drop-in replacement with EDC + RIR target support.
# - load_dataset() accepts edc_folder, rir_folder, target_type
# - RIR: loads waveform from rir_folder/<ID>.npy, pad/trim to rir_length,
#        optional "log" representation (signed log1p magnitude)
# - EDC: loads from edc_folder/<ID>.npy, pad/trim to edc_length,
#        optional log1p + downsample
# - load_features_and_targets() handles separate y scalers per target:
#       scaler_y_edc.save / scaler_y_rir.save
#
# Notes:
# - Assumes CSV has column "ID" that maps to "<ID>.npy"
# - Assumes features columns match those used in baseline repo

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------
def _pad_or_trim_1d(arr: np.ndarray, target_len: int, dtype=np.float32) -> np.ndarray:
    """Pad with zeros or trim a 1D array to exactly target_len."""
    arr = np.asarray(arr).reshape(-1).astype(dtype, copy=False)

    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]

    out = np.zeros(target_len, dtype=dtype)
    out[: arr.shape[0]] = arr
    return out


def _safe_downsample_fixed_len(arr: np.ndarray, factor: int, out_len: int) -> np.ndarray:
    """
    Downsample by slicing arr[::factor] then enforce exact out_len by pad/trim.
    Avoids np.resize (which can repeat data).
    """
    down = arr[::factor]
    return _pad_or_trim_1d(down, out_len, dtype=down.dtype)


# ---------------------------------------------------------
# CORE FUNCTION: loads raw dataset (X raw, y raw)
# ---------------------------------------------------------
def load_dataset(
    csv_path: str,
    edc_folder: str,
    rir_folder: str | None = None,
    target_type: str = "edc",            # "edc" or "rir"
    # EDC options
    edc_length: int = 96000,
    edc_downsample: int = 32,
    edc_apply_log1p: bool = True,
    # RIR options
    rir_length: int = 65536,
    rir_representation: str = "wave",    # "wave" or "log"
):
    """
    Loads input features X and targets y based on target_type.
    """

    csv_path = Path(csv_path)
    edc_folder = Path(edc_folder)
    rir_folder_path = Path(rir_folder) if rir_folder is not None else None

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not edc_folder.exists():
        raise FileNotFoundError(f"EDC folder not found: {edc_folder}")
    if target_type == "rir" and (rir_folder_path is None or not rir_folder_path.exists()):
        raise FileNotFoundError(
            f"rir_folder must be provided and exist when target_type='rir'. Got: {rir_folder}"
        )

    df = pd.read_csv(csv_path)

    if "ID" not in df.columns:
        raise KeyError("CSV must contain an 'ID' column.")

    # -------------------------------------------------
    # Build target filenames (FIX IS HERE)
    # -------------------------------------------------
    if target_type == "edc":
        # EDC filenames match ID exactly
        df["target_file"] = df["ID"].astype(str).apply(lambda s: f"{s}.npy")
    elif target_type == "rir":
        # RIR filenames do NOT contain '_edc'
        df["target_file"] = df["ID"].astype(str).apply(
            lambda s: f"{s.replace('_edc', '')}.npy"
        )
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # --------------------
    # Input features (X)
    # --------------------
    feature_cols = [
        "Length", "Width", "Height",
        "Source_X", "Source_Y", "Source_Z",
        "Receiver_X", "Receiver_Y", "Receiver_Z",
        "f1", "f2", "f3", "f4", "f5", "f6", "f7",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV is missing required feature columns: {missing}")

    X = df[feature_cols].values.astype(np.float32)

    # --------------------
    # Target loading (y)
    # --------------------
    y_list: list[np.ndarray] = []

    if target_type == "edc":
        out_len = edc_length // edc_downsample
        for fname in df["target_file"]:
            raw = np.load(edc_folder / fname)
            raw = _pad_or_trim_1d(raw, edc_length, dtype=np.float32)

            if edc_apply_log1p:
                raw = np.log1p(raw)

            down = _safe_downsample_fixed_len(raw, edc_downsample, out_len)
            y_list.append(down.astype(np.float32, copy=False))

        y = np.stack(y_list, axis=0)

    elif target_type == "rir":
        for fname in df["target_file"]:
            raw = np.load(rir_folder_path / fname)
            raw = _pad_or_trim_1d(raw, rir_length, dtype=np.float32)

            if rir_representation == "log":
                raw = np.sign(raw) * np.log1p(np.abs(raw))
            elif rir_representation != "wave":
                raise ValueError("rir_representation must be 'wave' or 'log'")

            y_list.append(raw.astype(np.float32, copy=False))

        y = np.stack(y_list, axis=0)

    print(f"[load_dataset] target_type = {target_type}")
    print(f"[load_dataset] X shape = {X.shape}")
    print(f"[load_dataset] y shape = {y.shape}")

    return X, y



# ---------------------------------------------------------
# WRAPPER FUNCTION: prepares data for training / inference
# ---------------------------------------------------------
def load_features_and_targets(
    csv_path: str,
    edc_folder: str,
    scaler_folder: str,
    *,
    rir_folder: str | None = None,
    target_type: str = "edc",
    # EDC options
    edc_length: int = 96000,
    edc_downsample: int = 32,
    edc_apply_log1p: bool = True,
    # RIR options
    rir_length: int = 65536,
    rir_representation: str = "wave",
    # Split
    test_size: float = 0.2,
    random_state: int = 42,
    # Scaler behavior
    fit_if_missing: bool = False,
):
    """
    Loads X, y using load_dataset(). Scales using saved scalers.

    Expected scaler files in scaler_folder:
      - scaler_X.save
      - scaler_y_edc.save   (when target_type='edc')
      - scaler_y_rir.save   (when target_type='rir')

    If fit_if_missing=True:
      - fits missing scalers on the loaded dataset and saves them.
    """

    scaler_folder_path = Path(scaler_folder)
    scaler_folder_path.mkdir(parents=True, exist_ok=True)

    # 1) Load raw
    X, y = load_dataset(
        csv_path=csv_path,
        edc_folder=edc_folder,
        rir_folder=rir_folder,
        target_type=target_type,
        edc_length=edc_length,
        edc_downsample=edc_downsample,
        edc_apply_log1p=edc_apply_log1p,
        rir_length=rir_length,
        rir_representation=rir_representation,
    )

    # 2) Load or fit scalers
    scaler_x_path = scaler_folder_path / "scaler_X.save"
    scaler_y_path = scaler_folder_path / (
        "scaler_y_edc.save" if target_type == "edc" else "scaler_y_rir.save"
    )

    scaler_X = None
    scaler_y = None

    if scaler_x_path.exists():
        scaler_X = joblib.load(scaler_x_path)
    elif fit_if_missing:
        scaler_X = StandardScaler().fit(X)
        joblib.dump(scaler_X, scaler_x_path)
    else:
        raise FileNotFoundError(
            f"Missing scaler_X.save at {scaler_x_path}. "
            f"Create it during training or set fit_if_missing=True."
        )

    if scaler_y_path.exists():
        scaler_y = joblib.load(scaler_y_path)
    elif fit_if_missing:
        scaler_y = StandardScaler().fit(y)
        joblib.dump(scaler_y, scaler_y_path)
    else:
        raise FileNotFoundError(
            f"Missing target scaler at {scaler_y_path}. "
            f"Create it during training or set fit_if_missing=True."
        )

    # 3) Scale
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    # 4) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val, scaler_X, scaler_y
