import numpy as np
from scipy.signal import lfilter

def reconstruct_rir(edc, decay_threshold=0.999, sign_sticky_prob=0.85):
    """
    Random sign-sticky RIR reconstruction method from EDC.
    """

    # Normalize EDC
    edc = edc / edc.max()

    # Convert EDC to magnitude envelope
    rir_mag = np.sqrt(edc[::-1])

    # Random phase generation (sign-sticky)
    rng = np.random.default_rng()
    signs = [1]
    for _ in range(len(rir_mag) - 1):
        if rng.random() < sign_sticky_prob:
            signs.append(signs[-1])
        else:
            signs.append(-signs[-1])

    signs = np.array(signs)

    rir = rir_mag * signs

    # Apply small smoothing filter
    rir = lfilter([1, -0.95], [1], rir)

    # Normalize
    rir /= np.max(np.abs(rir))

    return rir
