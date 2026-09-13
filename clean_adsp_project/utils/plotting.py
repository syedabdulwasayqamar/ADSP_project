import matplotlib.pyplot as plt
import numpy as np

def plot_edc(true_edc, pred_edc, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(true_edc, label="True EDC")
    plt.plot(pred_edc, label="Predicted EDC")
    plt.xlabel("Time steps")
    plt.ylabel("Energy Decay (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_rir(rir, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(rir)
    plt.title("Reconstructed RIR")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
