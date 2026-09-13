import numpy as np
import os

folder = "data/EDC"

for f in sorted(os.listdir(folder)):
    if f.endswith(".npy"):
        arr = np.load(os.path.join(folder, f))
        print(f"{f} shape = {arr.shape}")
