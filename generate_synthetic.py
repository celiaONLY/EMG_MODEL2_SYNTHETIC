# Synthetic EMG Data Generation Script (v2: distinct classes + better balance)
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
fs = 1000
duration = 1.0
n_samples = int(fs * duration)

low_cut = 20.0
high_cut = 450.0
notch_freq = 50.0
quality_factor = 30.0
apply_notch = True

# Base filters (default)
def make_filters(fs, low=low_cut, high=high_cut):
    b_band, a_band = signal.butter(4, [low, high], btype="band", fs=fs)
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs=fs)
    return (b_band, a_band), (b_notch, a_notch)

(b_band, a_band), (b_notch, a_notch) = make_filters(fs, low_cut, high_cut)

# =========================
# Step 2.1: Class Spec (fix duplicates)
# Each class has:
#   w = per-channel weights [LT, LM, RT, RM]
#   shape = envelope type
#   freq_scale = tiny frequency signature (optional)
# =========================
CLASS_SPEC = {
    "Normal_bite":        {"w": [1.0, 1.0, 1.0, 1.0], "shape": "burst",   "freq_scale": 1.00},
    # Anterior: make masseters stronger than temporalis (distinct from Normal)
    "Anterior_bite":      {"w": [0.7, 1.3, 0.7, 1.3], "shape": "burst",   "freq_scale": 1.05},

    # High point = sharper spike + strong unilateral
    "Left_high_point":    {"w": [1.7, 1.7, 0.6, 0.6], "shape": "spike",   "freq_scale": 1.00},
    "Right_high_point":   {"w": [0.6, 0.6, 1.7, 1.7], "shape": "spike",   "freq_scale": 1.00},

    # Lateral bite = longer plateau + unilateral (distinct from high_point)
    "Left_lateral_bite":  {"w": [1.5, 1.5, 0.8, 0.8], "shape": "plateau", "freq_scale": 0.95},
    "Right_lateral_bite": {"w": [0.8, 0.8, 1.5, 1.5], "shape": "plateau", "freq_scale": 0.95},

    # Chewing = multi-burst; unilateral vs bilateral
    "Left_chewing":       {"w": [1.5, 1.5, 0.7, 0.7], "shape": "chew",    "freq_scale": 1.00},
    "Right_chewing":      {"w": [0.7, 0.7, 1.5, 1.5], "shape": "chew",    "freq_scale": 1.00},
    "Bilateral_chewing":  {"w": [1.2, 1.2, 1.2, 1.2], "shape": "chew",    "freq_scale": 1.00},
}

classes = list(CLASS_SPEC.keys())

# =========================
# Step 2.2: Balance control (fix binary imbalance)
# Make Normal have more files than each abnormal class
# =========================
N_PER_CLASS = {c: 5 for c in classes}
N_PER_CLASS["Normal_bite"] = 12  # <- boost normal (tune 10~20 if you want)

# =========================
# Paths
# =========================
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
data_dir = project_dir / "data"

# Create class folders
data_dir.mkdir(parents=True, exist_ok=True)
for cls in classes:
    (data_dir / cls).mkdir(parents=True, exist_ok=True)

# OPTIONAL cleanup (recommended): remove old xlsx so you don't mix old/new samples
def cleanup_class_folders():
    for cls in classes:
        folder = data_dir / cls
        for fp in folder.glob("*.xlsx"):
            fp.unlink()

# Uncomment if you want a clean regen each time:
cleanup_class_folders()

# =========================
# Envelope builders
# =========================
def envelope_burst(t, center=0.5, dur=0.40, base=0.10, peak_add=0.90):
    env = np.full_like(t, base, dtype=float)
    center_idx = int(center * fs)
    burst_len = int(dur * fs)
    burst_len = max(burst_len, 5)
    if burst_len % 2 == 0:
        burst_len += 1
    half = burst_len // 2
    s = max(0, center_idx - half)
    e = min(len(t), center_idx + half + 1)
    w = np.hanning(e - s)
    w = w / (w.max() + 1e-8)
    env[s:e] += peak_add * w
    return np.clip(env, 0, 1.0)

def envelope_spike(t):
    # sharper + slightly higher peak
    center = 0.5 + np.random.uniform(-0.03, 0.03)
    dur = 0.12 + np.random.uniform(-0.02, 0.02)
    return envelope_burst(t, center=center, dur=dur, base=0.08, peak_add=0.92)

def envelope_plateau(t):
    # plateau with smooth ramps
    env = np.full_like(t, 0.10, dtype=float)
    start = 0.30 + np.random.uniform(-0.03, 0.03)
    end   = 0.70 + np.random.uniform(-0.03, 0.03)
    start = max(0.05, start)
    end = min(0.95, end)
    s = int(start * fs)
    e = int(end * fs)
    s = max(0, min(s, len(t)-1))
    e = max(s+1, min(e, len(t)))

    env[s:e] = 1.0

    # smooth ramps (20 ms)
    ramp = int(0.02 * fs)
    if ramp > 3:
        # left ramp
        ls = max(0, s - ramp)
        le = s
        w = np.hanning((le - ls) * 2)[: (le - ls)]
        env[ls:le] = 0.10 + (1.0 - 0.10) * (w / (w.max() + 1e-8))
        # right ramp
        rs = e
        re = min(len(t), e + ramp)
        w = np.hanning((re - rs) * 2)[(re - rs):]
        env[rs:re] = 0.10 + (1.0 - 0.10) * (w / (w.max() + 1e-8))
    return np.clip(env, 0, 1.0)

def envelope_chew(t):
    env = np.full_like(t, 0.08, dtype=float)
    # 3–5 bursts with jitter
    burst_count = np.random.randint(3, 6)
    centers = np.linspace(0.2, 0.8, burst_count) + np.random.uniform(-0.03, 0.03, size=burst_count)
    for c in centers:
        dur = 0.12 + np.random.uniform(-0.02, 0.02)
        env = np.maximum(env, envelope_burst(t, center=float(c), dur=float(dur), base=0.08, peak_add=0.92))
    return np.clip(env, 0, 1.0)

def build_envelope(t, shape):
    if shape == "burst":
        center = 0.5 + np.random.uniform(-0.03, 0.03)
        dur = 0.40 + np.random.uniform(-0.05, 0.05)
        return envelope_burst(t, center=center, dur=dur)
    if shape == "spike":
        return envelope_spike(t)
    if shape == "plateau":
        return envelope_plateau(t)
    if shape == "chew":
        return envelope_chew(t)
    return envelope_burst(t)

# =========================
# Generator
# =========================
def generate_emg_record(label):
    t = np.linspace(0, duration, n_samples, endpoint=False)

    spec = CLASS_SPEC[label]
    w = np.array(spec["w"], dtype=float)   # [LT, LM, RT, RM]
    shape = spec["shape"]
    freq_scale = float(spec.get("freq_scale", 1.0))

    env = build_envelope(t, shape)

    # Optional tiny frequency signature: tweak bandpass slightly per class
    # (keeps it simple; enough to separate Anterior vs Normal a bit)
    if abs(freq_scale - 1.0) > 1e-6:
        low = max(10.0, low_cut * freq_scale)
        high = min((fs/2) - 5.0, high_cut * freq_scale)
        (bB, aB), (bN, aN) = make_filters(fs, low, high)
    else:
        bB, aB, bN, aN = b_band, a_band, b_notch, a_notch

    # Noise -> bandpass -> (optional notch)
    noises = [np.random.randn(n_samples) for _ in range(4)]
    sigs = [signal.filtfilt(bB, aB, n) for n in noises]
    if apply_notch:
        sigs = [signal.filtfilt(bN, aN, s) for s in sigs]

    # Apply envelope + per-channel weights
    # Add small per-record amplitude variation to avoid “too clean” signals
    amp_jitter = 1.0 + np.random.uniform(-0.10, 0.10)
    LT = sigs[0] * env * w[0] * amp_jitter
    LM = sigs[1] * env * w[1] * amp_jitter
    RT = sigs[2] * env * w[2] * amp_jitter
    RM = sigs[3] * env * w[3] * amp_jitter

    return t, LT, LM, RT, RM

# =========================
# Generate files + meta
# =========================
meta_records = []
example_signals = {}

print("Generating synthetic EMG data...")
for label in classes:
    n_files = int(N_PER_CLASS[label])
    for i in range(1, n_files + 1):
        t, LT, LM, RT, RM = generate_emg_record(label)

        file_name = f"{label}_{i}.xlsx"
        file_path = data_dir / label / file_name

        df = pd.DataFrame({"Time": t, "LT": LT, "LM": LM, "RT": RT, "RM": RM})
        df.to_excel(file_path, index=False)

        meta_records.append({"label": label, "filepath": str(file_path)})

        if label in ["Normal_bite", "Left_high_point"] and i == 1:
            example_signals[label] = (t, LT, LM, RT, RM)

    print(f"  - Created {n_files} samples for class '{label}'")

meta_df = pd.DataFrame(meta_records)
meta_csv_path = data_dir / "meta.csv"
meta_df.to_csv(meta_csv_path, index=False)
print(f"Saved metadata to {meta_csv_path}")

# =========================
# Plot examples
# =========================
if "Normal_bite" in example_signals and "Left_high_point" in example_signals:
    t_nb, LT_nb, LM_nb, RT_nb, RM_nb = example_signals["Normal_bite"]
    t_lhp, LT_lhp, LM_lhp, RT_lhp, RM_lhp = example_signals["Left_high_point"]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(t_nb, LT_nb, label="LT")
    axs[0].plot(t_nb, LM_nb, label="LM")
    axs[0].plot(t_nb, RT_nb, label="RT")
    axs[0].plot(t_nb, RM_nb, label="RM")
    axs[0].set_title("Normal_bite Example (4 channels)")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude [a.u.]")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t_lhp, LT_lhp, label="LT")
    axs[1].plot(t_lhp, LM_lhp, label="LM")
    axs[1].plot(t_lhp, RT_lhp, label="RT")
    axs[1].plot(t_lhp, RM_lhp, label="RM")
    axs[1].set_title("Left_high_point Example (4 channels)")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Amplitude [a.u.]")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Example signals for plotting not available.")