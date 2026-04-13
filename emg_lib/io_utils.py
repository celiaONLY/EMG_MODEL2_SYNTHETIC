from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .config import PROJECT_DIR, CHANNELS, DEFAULT_COLUMN_ALIASES, TARGET_FS


def resolve_filepath(fp: str | Path, project_dir: Path = PROJECT_DIR) -> Path:
    """
    Handles absolute Windows paths stored in meta.csv and converts them into usable paths
    on any machine by falling back to the trailing ".../data/..." part.
    """
    p = Path(str(fp))

    # If it already exists, use it
    if p.exists():
        return p

    s = str(fp).replace("\\", "/")

    # Try to locate "/data/" tail and rebuild under current project_dir
    idx = s.lower().find("/data/")
    if idx != -1:
        tail = s[idx + 1 :]  # remove leading slash
        candidate = project_dir / tail
        if candidate.exists():
            return candidate

    # Otherwise try as relative path directly
    candidate = project_dir / s
    if candidate.exists():
        return candidate

    # Last fallback: return original path (will fail later with a clear error)
    return p


def _normalize_columns(cols):
    return [str(c).strip() for c in cols]


def load_emg_file(
    fp: str | Path,
    column_map: dict[str, str] | None = None,
    channels: list[str] = CHANNELS,
    project_dir: Path = PROJECT_DIR,
    fs_hint: int = TARGET_FS,
) -> tuple[np.ndarray, dict[str, np.ndarray], Path]:
    """
    Loads .xlsx or .csv and returns:
      t (1D array),
      sigs dict: {LT, LM, RT, RM -> 1D arrays},
      resolved_path
    If Time is missing, it creates Time from sample index using fs_hint.
    """
    path = resolve_filepath(fp, project_dir=project_dir)

    if not path.exists():
        raise FileNotFoundError(f"File not found after resolve: {fp} -> {path}")

    # Load
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df.columns = _normalize_columns(df.columns)

    # Build a rename map: lowercase column name -> canonical
    rename = {}
    aliases = dict(DEFAULT_COLUMN_ALIASES)
    if column_map:
        # user-specified mapping overrides defaults
        for k, v in column_map.items():
            aliases[str(k).strip().lower()] = v

    for c in df.columns:
        key = str(c).strip().lower()
        if key in aliases:
            rename[c] = aliases[key]

    if rename:
        df = df.rename(columns=rename)

    # If Time missing, create from index
    if "Time" not in df.columns:
        n = len(df)
        df["Time"] = np.arange(n) / float(fs_hint)

    # Check channels
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required channels {missing} in file {path.name}. Columns={list(df.columns)}")

    t = df["Time"].to_numpy(dtype=float)
    sigs = {c: df[c].to_numpy(dtype=float) for c in channels}
    return t, sigs, path