# utils/preprocessing.py

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler 
from typing import Literal

def scale_and_save(
    X: np.ndarray,
    y: np.ndarray,
    scaler_dir: str,
    target_type: Literal["edc", "rir"] = "edc",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize X and y, and save scalers to scaler_dir.
    ...
    """
    scaler_dir_path = Path(scaler_dir)
    scaler_dir_path.mkdir(parents=True, exist_ok=True)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    joblib.dump(scaler_X, scaler_dir_path / "scaler_X.save")

    if target_type == "edc":
        joblib.dump(scaler_y, scaler_dir_path / "scaler_y_edc.save")
    elif target_type == "rir":
        joblib.dump(scaler_y, scaler_dir_path / "scaler_y_rir.save")
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return X_scaled, y_scaled
