import numpy as np
import os

folder = "data/EDC"

for f in os.listdir(folder):
    if f.endswith(".npy"):
        arr = np.load(os.path.join(folder, f))
        print(f, arr.shape)
        break
