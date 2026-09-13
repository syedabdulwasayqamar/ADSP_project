import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Paths (edit if needed)
# -----------------------------
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "room_acoustic_largedataset")
EDC_DIR = os.path.join(DATASET_DIR, "EDC")
FEATURE_CSV = os.path.join(DATASET_DIR, "roomFeaturesDataset.csv")

PREDICTIONS_FILE = "sample_predictions.npy"      # TCN predictions
LSTM_RESULTS_FILE = "lstm_sample_predictions.npy" # LSTM predictions if available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load TCN predictions
# -----------------------------
y_pred = np.load(PREDICTIONS_FILE)
print(f"TCN Predictions shape: {y_pred.shape}")

# Load feature CSV to get IDs
df = pd.read_csv(FEATURE_CSV)
if len(df) > y_pred.shape[0]:
    df = df.sample(n=y_pred.shape[0], random_state=42).reset_index(drop=True)
ids = df["ID"].tolist()

# -----------------------------
# Metric functions
# -----------------------------
def edt(edc):
    """Early Decay Time (EDT) in seconds"""
    edc_db = 10 * np.log10(edc / np.max(edc) + 1e-12)
    idx = np.where(edc_db <= -10)[0]
    if len(idx) == 0 or idx[0] == 0:
        return 0.0
    slope = (edc_db[idx[0]] - edc_db[0]) / idx[0]
    return -60 / slope if slope != 0 else 0.0

def t20(edc):
    """T20 in seconds"""
    edc_db = 10 * np.log10(edc / np.max(edc) + 1e-12)
    i1 = np.where(edc_db <= -5)[0]
    i2 = np.where(edc_db <= -25)[0]
    if len(i1) == 0 or len(i2) == 0 or i2[0] == i1[0]:
        return 0.0
    slope = (edc_db[i2[0]] - edc_db[i1[0]]) / (i2[0] - i1[0])
    return -60 / slope if slope != 0 else 0.0

def c50(edc):
    """Clarity C50 in dB"""
    split = int(0.05 * len(edc))
    early = np.sum(edc[:split] ** 2)
    late = np.sum(edc[split:] ** 2)
    if late <= 0:
        return 0.0
    return 10 * np.log10(early / late)

# -----------------------------
# Compute metrics for TCN
# -----------------------------
edt_pred, t20_pred, c50_pred = [], [], []
edt_true, t20_true, c50_true = [], [], []

for i, id_ in enumerate(ids):
    y_t = np.load(os.path.join(EDC_DIR, f"{id_}.npy"))
    y_p = np.clip(y_pred[i], 1e-12, None)

    edt_true.append(edt(y_t))
    t20_true.append(t20(y_t))
    c50_true.append(c50(y_t))

    edt_pred.append(edt(y_p))
    t20_pred.append(t20(y_p))
    c50_pred.append(c50(y_p))

def compute_metrics(y_true_list, y_pred_list):
    mae = mean_absolute_error(y_true_list, y_pred_list)
    rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
    r2 = r2_score(y_true_list, y_pred_list)
    return mae, rmse, r2

metrics_tcn = {
    'EDT (s)': compute_metrics(edt_true, edt_pred),
    'T20 (s)': compute_metrics(t20_true, t20_pred),
    'C50 (dB)': compute_metrics(c50_true, c50_pred)
}

# -----------------------------
# Compute metrics for LSTM (if available)
# -----------------------------
if os.path.exists(LSTM_RESULTS_FILE):
    y_lstm = np.load(LSTM_RESULTS_FILE)
    edt_lstm, t20_lstm, c50_lstm = [], [], []
    for i, id_ in enumerate(ids):
        y_t = np.load(os.path.join(EDC_DIR, f"{id_}.npy"))
        y_p = np.clip(y_lstm[i], 1e-12, None)
        edt_lstm.append(edt(y_p))
        t20_lstm.append(t20(y_p))
        c50_lstm.append(c50(y_p))
    metrics_lstm = {
        'EDT (s)': compute_metrics(edt_true, edt_lstm),
        'T20 (s)': compute_metrics(t20_true, t20_lstm),
        'C50 (dB)': compute_metrics(c50_true, c50_lstm)
    }
else:
    metrics_lstm = None

# -----------------------------
# Display results in table
# -----------------------------
df_results = pd.DataFrame({
    "Metric": metrics_tcn.keys(),
    "MAE (TCN)": [metrics_tcn[k][0] for k in metrics_tcn],
    "RMSE (TCN)": [metrics_tcn[k][1] for k in metrics_tcn],
    "R² (TCN)": [metrics_tcn[k][2] for k in metrics_tcn],
})

if metrics_lstm is not None:
    df_results["MAE (LSTM)"] = [metrics_lstm[k][0] for k in metrics_lstm]
    df_results["RMSE (LSTM)"] = [metrics_lstm[k][1] for k in metrics_lstm]
    df_results["R² (LSTM)"] = [metrics_lstm[k][2] for k in metrics_lstm]

print("\nModel Performance Comparison:")
print(df_results.to_string(index=False))
