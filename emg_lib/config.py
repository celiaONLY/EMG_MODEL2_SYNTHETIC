from pathlib import Path

# Project root folder (EMG_PROJECT)
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Canonical channel names used everywhere
CHANNELS = ["LT", "LM", "RT", "RM"]

# Target sampling rate for the whole pipeline
# (Even if real data is 5000 Hz, we can resample to this)
TARGET_FS = 1000

# Filters (will be auto-clipped if needed)
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0

# Mains noise (set to 60 if needed for some labs/countries)
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
USE_NOTCH = True

# Windowing (in seconds)
WINDOW_SEC = 0.25
STEP_SEC = 0.125

# Feature options
RECTIFY = False
USE_MDF = False

EPS = 1e-8

# Optional: aliases mapping (lowercase keys)
DEFAULT_COLUMN_ALIASES = {
    "time": "Time",
    "t": "Time",
    "timestamp": "Time",
    "lt": "LT",
    "lm": "LM",
    "rt": "RT",
    "rm": "RM",
}