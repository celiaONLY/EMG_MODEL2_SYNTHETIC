from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend, resample_poly
from fractions import Fraction

from .config import (
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ, NOTCH_Q, USE_NOTCH,
    RECTIFY, TARGET_FS
)


def infer_fs(t: np.ndarray, default_fs: int = TARGET_FS) -> float:
    """
    Infer sampling rate from Time column.
    Handles:
      - time already in seconds
      - time in milliseconds
      - time is sample index (0,1,2,...) => uses default_fs
    """
    t = np.asarray(t, dtype=float)

    # Remove NaNs
    t = t[np.isfinite(t)]
    if len(t) < 3:
        return float(default_fs)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return float(default_fs)

    med = float(np.median(dt))

    # If time is basically sample index (integers, step ~1)
    # Use default_fs
    if np.allclose(t, np.round(t)) and 0.9 <= med <= 1.1 and np.max(t) > 50:
        return float(default_fs)

    # If looks like milliseconds: dt ~ 1 (ms) or 0.2 (for 5kHz in ms)
    # Convert to seconds: fs = 1000/dt
    if np.max(t) > 20 and med >= 0.05:
        return float(1000.0 / med)

    # Otherwise assume seconds
    return float(1.0 / med)


def resample_to_target(x: np.ndarray, fs_in: float, fs_target: int = TARGET_FS) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if fs_in <= 0:
        return x
    if abs(fs_in - fs_target) < 1e-6:
        return x

    frac = Fraction(fs_target, int(round(fs_in))).limit_denominator(200)
    up, down = frac.numerator, frac.denominator
    return resample_poly(x, up, down)


def design_filters(fs: float, bp_low: float = BANDPASS_LOW, bp_high: float = BANDPASS_HIGH):
    nyq = fs / 2.0
    hi = min(bp_high, nyq - 1.0)  # clip
    lo = min(bp_low, hi - 1.0)
    b_bp, a_bp = butter(4, [lo, hi], btype="bandpass", fs=fs)

    if USE_NOTCH and NOTCH_FREQ < nyq:
        b_notch, a_notch = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=fs)
    else:
        b_notch, a_notch = None, None

    return (b_bp, a_bp), (b_notch, a_notch)


def apply_filters(sig: np.ndarray, fs: float) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    sig = detrend(sig, type="constant")

    (b_bp, a_bp), (b_notch, a_notch) = design_filters(fs)
    sig = filtfilt(b_bp, a_bp, sig)

    if USE_NOTCH and b_notch is not None:
        sig = filtfilt(b_notch, a_notch, sig)

    if RECTIFY:
        sig = np.abs(sig)

    return sig