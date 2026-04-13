from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"D:\EMG_Project3")
DATA_DIR = PROJECT_DIR / "data"

records = []

# Scan each class folder inside data/
for class_dir in DATA_DIR.iterdir():
    if not class_dir.is_dir():
        continue
    if class_dir.name == "processed":
        continue

    label = class_dir.name

    # collect xlsx/csv EMG files
    for fp in sorted(class_dir.glob("*.xlsx")):
        records.append({
            "filepath": str(fp),
            "label": label,
        })

    for fp in sorted(class_dir.glob("*.csv")):
        records.append({
            "filepath": str(fp),
            "label": label,
        })

meta = pd.DataFrame(records)

out_path = DATA_DIR / "meta.csv"
meta.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"Saved: {out_path}")
print(meta.head())
print(f"Total files: {len(meta)}")
print("\nCounts by label:")
print(meta["label"].value_counts())