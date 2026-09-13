import os
import numpy as np
import soundfile as sf
from pathlib import Path

WAV_DIR = Path("data/RIRs")
OUT_DIR = Path("data/RIRs_npy")

OUT_DIR.mkdir(exist_ok=True)

for wav_file in WAV_DIR.glob("*.wav"):
    rir, sr = sf.read(wav_file)

    # If stereo → convert to mono
    if rir.ndim > 1:
        rir = rir.mean(axis=1)

    rir = rir.astype(np.float32)

    out_file = OUT_DIR / (wav_file.stem + ".npy")
    np.save(out_file, rir)

    print(f"Saved {out_file} | shape={rir.shape} | sr={sr}")
