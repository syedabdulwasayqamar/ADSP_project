import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from adsp_project.models.transformer_edc import TransformerEDCModel
from adsp_project.utils.data_utils import load_test_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = ROOT / "Results/2026-01-03/18-33-44/Trained_Models/transformer.ckpt"
OUTPUT_PATH = ROOT / "generated_outputs/transformer_predictions.npy"

def main():
    model = TransformerEDCModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location=DEVICE
    ).to(DEVICE)
    model.eval()

    test_loader = load_test_data()

    preds = []
    gts = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y_hat = model(x)
            preds.append(y_hat.cpu().numpy())
            gts.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    np.save(OUTPUT_PATH, {
        "predictions": preds,
        "ground_truth": gts
    })

    print(f"Saved transformer predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
