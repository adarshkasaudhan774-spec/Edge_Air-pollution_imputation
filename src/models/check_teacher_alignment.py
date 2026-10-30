# ============================================================
# STEP 1— Teacher Transformer Data/Mask Alignment Check
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "Data" / "raw data"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

AIR_FILE = RAW_DIR / "air_quality_2019_2021_hourly.csv"
MET_FILE = RAW_DIR / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = PROCESSED_DIR / "fixed_artificial_mask_new.csv"

FEATURES = [
    "pm25",
    "pm10",
    "no2",
    "o3",
    "so2",
    "nox",
    "co",
    "temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "rainfall",
]

NUM_STATIONS = 16
NUM_FEATURES = 13


# ============================================================
# 2. Load Data
# ============================================================

print("=" * 70)
print("STEP 8.2 — DATA/MASK ALIGNMENT CHECK")
print("=" * 70)

air = pd.read_csv(AIR_FILE)
met = pd.read_csv(MET_FILE)
mask = pd.read_csv(MASK_FILE)

air["timestamp"] = pd.to_datetime(
    air["timestamp"],
    errors="raise"
)

met["timestamp"] = pd.to_datetime(
    met["timestamp"],
    errors="raise"
)


# ============================================================
# 3. Merge and Sort Data
# ============================================================

merged = pd.merge(
    air,
    met,
    on=[
        "station_id",
        "station_name",
        "timestamp"
    ],
    how="inner",
    validate="one_to_one"
)

merged = merged.sort_values(
    ["timestamp", "station_id"],
    kind="mergesort"
).reset_index(drop=True)

print("\nMerged data shape:", merged.shape)
print("Mask shape:", mask.shape)


# ============================================================
# 4. Verify Dimensions
# ============================================================

assert len(merged) == len(mask), (
    f"Row mismatch: data={len(merged)}, mask={len(mask)}"
)

assert list(mask.columns) == FEATURES, (
    "Mask feature order does not match expected feature order."
)

assert merged["station_id"].nunique() == NUM_STATIONS, (
    "Expected 16 stations."
)

assert merged["timestamp"].nunique() == 26304, (
    "Expected 26,304 timestamps."
)

assert len(merged) == NUM_STATIONS * 26304, (
    "Unexpected number of rows."
)

print("\nDimension checks: PASSED")


# ============================================================
# 5. Inspect Row Ordering
# ============================================================

print("\nFirst 20 rows of sorted data:")

print(
    merged[
        ["timestamp", "station_id", "station_name"]
    ].head(20).to_string(index=False)
)

print("\nFirst 20 mask rows:")

print(mask.head(20).to_string(index=False))


# ============================================================
# 6. Verify Station Order Within Each Timestamp
# ============================================================

station_order = (
    merged.groupby("timestamp", sort=True)["station_id"]
    .apply(list)
)

expected_station_order = station_order.iloc[0]

consistent_station_order = all(
    station_ids == expected_station_order
    for station_ids in station_order
)

print("\nStation order:", expected_station_order)
print("Consistent station order:", consistent_station_order)

assert consistent_station_order, (
    "Station order differs between timestamps."
)


# ============================================================
# 7. Verify Mask Values
# ============================================================

mask_values = set(
    np.unique(mask.to_numpy())
)

print("\nUnique mask values:", mask_values)

assert mask_values.issubset({0, 1}), (
    "Mask must contain only 0 and 1."
)

print("Binary mask check: PASSED")


# ============================================================
# 8. Final Summary
# ============================================================

print("\n" + "=" * 70)
print("STEP 8.2 — ALIGNMENT CHECK COMPLETED")
print("=" * 70)

print("Merged rows:", len(merged))
print("Mask rows:", len(mask))
print("Stations:", merged["station_id"].nunique())
print("Timestamps:", merged["timestamp"].nunique())
print("Features:", len(FEATURES))

print(
    "\nImportant: This script verifies dimensions and ordering "
    "structure. It cannot prove that the mask was generated from "
    "this exact row ordering because the mask has no timestamp "
    "or station identifiers."
)