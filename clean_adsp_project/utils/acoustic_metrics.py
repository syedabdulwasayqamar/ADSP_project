import numpy as np

def compute_decay_curve(rir):
    """Returns Schroeder integration curve."""
    energy = rir**2
    edc = np.flip(np.cumsum(np.flip(energy)))
    edc = edc / np.max(edc)
    return edc

def compute_EDT(edc):
    """EDT = early decay time (0 to -10 dB extrapolated to -60 dB)."""
    edc_db = 10 * np.log10(edc + 1e-9)
    idx_0 = np.where(edc_db <= 0)[0][0]
    idx_m10 = np.where(edc_db <= -10)[0][0]

    # linear regression
    y = edc_db[idx_0:idx_m10]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)

    EDT = -60 / slope
    return EDT / 48000  # in seconds

def compute_T20(edc):
    """T20 = decay time from -5 to -25 dB * 3."""
    edc_db = 10 * np.log10(edc + 1e-9)

    idx_m5 = np.where(edc_db <= -5)[0][0]
    idx_m25 = np.where(edc_db <= -25)[0][0]

    y = edc_db[idx_m5:idx_m25]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)

    T20 = -60 / slope
    return T20 / 48000

def compute_C50(rir):
    """Clarity index C50 = 10 log10(EarlyEnergy / LateEnergy)."""
    early = np.sum(rir[:2400] ** 2)   # first 50 ms
    late  = np.sum(rir[2400:] ** 2)

    return 10 * np.log10((early + 1e-9) / (late + 1e-9))

def edc_to_db(edc):
    """
    Convert EDC (energy domain) to dB safely.
    """
    edc = np.maximum(edc, 1e-12)   # avoid negatives / zeros
    edc_db = 10 * np.log10(edc)
    return edc_db

def compute_C50_from_edc(edc_lin, fs=48000):
    """
    Compute C50 from linear EDC.
    """
    idx_50ms = int(0.05 * fs)

    idx_50ms = min(idx_50ms, len(edc_lin) - 1)

    E_early = edc_lin[0] - edc_lin[idx_50ms]
    E_late = edc_lin[idx_50ms] - edc_lin[-1]

    if E_late <= 0 or E_early <= 0:
        return np.nan

    return 10 * np.log10(E_early / E_late)
