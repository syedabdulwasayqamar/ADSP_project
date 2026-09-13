# plot_model_comparison.py
# Plot TCN vs LSTM vs Transformer performance for EDC metrics

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Dummy example data (replace with actual predictions if available)
# ---------------------------------------------------------
# Ground truth metrics (from dataset)
EDT_true = np.random.uniform(0.5, 1.5, 100)
T20_true = np.random.uniform(0.6, 1.6, 100)
C50_true = np.random.uniform(0, 10, 100)

# TCN predictions (from your original run)
EDT_tcn = EDT_true + np.random.normal(0, 15000, 100)  # huge error
T20_tcn = T20_true + np.random.normal(0, 22000, 100)
C50_tcn = C50_true + np.random.normal(0, 70, 100)

# LSTM predictions (professor reference)
EDT_lstm = EDT_true + np.random.normal(0, 0.02, 100)
T20_lstm = T20_true + np.random.normal(0, 0.03, 100)
C50_lstm = C50_true + np.random.normal(0, 2, 100)

# Transformer predictions (our model)
EDT_trans = EDT_true + np.random.normal(0, 0.028, 100)
T20_trans = T20_true + np.random.normal(0, 0.092, 100)
C50_trans = C50_true + np.random.normal(0, 1.79, 100)

# ---------------------------------------------------------
# Plotting function
# ---------------------------------------------------------
def plot_metric_comparison(metric_name, y_true, y_tcn, y_lstm, y_trans, unit=""):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, 'k-', label="Ground truth")
    plt.plot(y_tcn, 'r--', label="TCN")
    plt.plot(y_lstm, 'b--', label="LSTM")
    plt.plot(y_trans, 'g--', label="Transformer")
    plt.xlabel("Sample Index")
    plt.ylabel(f"{metric_name} {unit}")
    plt.title(f"{metric_name} Prediction Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Generate plots
# ---------------------------------------------------------
plot_metric_comparison("EDT", EDT_true, EDT_tcn, EDT_lstm, EDT_trans, "s")
plot_metric_comparison("T20", T20_true, T20_tcn, T20_lstm, T20_trans, "s")
plot_metric_comparison("C50", C50_true, C50_tcn, C50_lstm, C50_trans, "dB")
