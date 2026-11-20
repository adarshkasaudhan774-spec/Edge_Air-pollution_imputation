from pathlib import Path
import pandas as pd
import numpy as np

# ============================================
# STEP 8.5 — INSPECT MASK ALIGNMENT
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "Data" / "raw data"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

AIR_FILE = RAW_DIR / "air_quality_2019_2021_hourly.csv"
MET_FILE = RAW_DIR / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = PROCESSED_DIR / "fixed_artificial_mask_new.csv"

FEATURES = [
    "pm25", "pm10", "no2", "o3", "so2", "nox", "co",
    "temperature", "relative_humidity", "wind_speed",
    "wind_direction", "pressure", "rainfall"
]

print("=" * 60)
print("STEP 8.5 — INSPECT MASK ALIGNMENT")
print("=" * 60)

# Load files without modifying them
air = pd.read_csv(AIR_FILE)
weather = pd.read_csv(MET_FILE)
mask = pd.read_csv(MASK_FILE)

air["timestamp"] = pd.to_datetime(air["timestamp"])
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

# Merge using the same keys as the existing verification script
merged = pd.merge(
    air,
    weather,
    on=["station_id", "station_name", "timestamp"],
    how="inner",
    validate="one_to_one"
)

# Inspect original merge order
print("\nOriginal merged shape:", merged.shape)
print("Mask shape:", mask.shape)

print("\nFirst 10 rows in original merged data:")
print(
    merged[["timestamp", "station_id"]]
    .head(10)
    .to_string(index=False)
)

print("\nFirst 10 mask rows:")
print(mask[FEATURES].head(10).to_string(index=False))

# Inspect whether the original merge is chronologically sorted
original_order_sorted = merged[
    ["timestamp", "station_id"]
].equals(
    merged.sort_values(
        ["timestamp", "station_id"],
        kind="mergesort"
    ).reset_index(drop=True)[["timestamp", "station_id"]]
)

print("\nOriginal merge already chronologically sorted:", original_order_sorted)

# Check mask shape and values
print("\nMask feature columns match:", list(mask.columns) == FEATURES)
print("Mask rows match merged rows:", len(mask) == len(merged))
print("Mask values limited to 0 and 1:",
      set(np.unique(mask.to_numpy())).issubset({0, 1}))

print("\nIMPORTANT:")
print("These checks cannot prove exact mask alignment.")
print("The mask has no station_id or timestamp columns.")
print("No files have been modified.")

print("\nSTEP 8.5 INSPECTION COMPLETED")