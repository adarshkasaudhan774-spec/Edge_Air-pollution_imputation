from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

AIR_FILE = ROOT / "Data" / "raw data" / "air_quality_2019_2021_hourly.csv"
MET_FILE = ROOT / "Data" / "raw data" / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = ROOT / "Data" / "processed" / "fixed_artificial_mask_new.csv"

FEATURES = [
    "pm25", "pm10", "no2", "o3", "so2", "nox", "co",
    "temperature", "relative_humidity", "wind_speed",
    "wind_direction", "pressure", "rainfall"
]

EXPECTED_ROWS = 16 * 26304

print("=" * 65)
print("STEP 8.8 — DATA/MASK ALIGNMENT CHECK")
print("=" * 65)

for path in [AIR_FILE, MET_FILE, MASK_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

air = pd.read_csv(AIR_FILE)
met = pd.read_csv(MET_FILE)
mask = pd.read_csv(MASK_FILE)

air["timestamp"] = pd.to_datetime(air["timestamp"], errors="raise")
met["timestamp"] = pd.to_datetime(met["timestamp"], errors="raise")

print("\nFiles loaded successfully.")
print("Air shape:", air.shape)
print("Weather shape:", met.shape)
print("Mask shape:", mask.shape)

join_keys = ["station_id", "station_name", "timestamp"]

air_order_matches = (
    air[join_keys].reset_index(drop=True).equals(
        met[join_keys].reset_index(drop=True)
    )
)

print("\nCHECK 1 — Air/weather row order matches:", air_order_matches)

merged = pd.merge(
    air,
    met,
    on=join_keys,
    how="inner",
    validate="one_to_one",
    sort=False
).reset_index(drop=True)

print("\nCHECK 2 — Merge")
print("Merged shape:", merged.shape)
print("Expected rows:", EXPECTED_ROWS)
print("Row count correct:", len(merged) == EXPECTED_ROWS)

if len(merged) != EXPECTED_ROWS:
    raise ValueError("Unexpected merged row count.")

original_keys = merged[
    ["station_id", "timestamp"]
].reset_index(drop=True)

sorted_data = merged.sort_values(
    ["timestamp", "station_id"],
    kind="mergesort"
).reset_index(drop=True)

sorted_keys = sorted_data[
    ["station_id", "timestamp"]
].reset_index(drop=True)

same_order = original_keys.equals(sorted_keys)
different_rows = int(
    (
        original_keys["station_id"].ne(sorted_keys["station_id"])
        | original_keys["timestamp"].ne(sorted_keys["timestamp"])
    ).sum()
)

print("\nCHECK 3 — Sorting comparison")
print("Original order equals sorted order:", same_order)
print("Rows in different positions:", different_rows)

print("\nCHECK 4 — Mask structure")
print("Mask columns match:", list(mask.columns) == FEATURES)
print("Mask row count matches:", len(mask) == len(merged))

if list(mask.columns) != FEATURES:
    raise ValueError("Mask feature columns or order are incorrect.")

if len(mask) != len(merged):
    raise ValueError("Mask and merged data row counts differ.")

mask_numeric = mask.apply(pd.to_numeric, errors="raise")

if mask_numeric.isna().any().any():
    raise ValueError("Mask contains missing values.")

values = set(np.unique(mask_numeric.to_numpy()))
print("Mask unique values:", values)

if not values.issubset({0, 1}):
    raise ValueError("Mask must contain only 0 and 1.")

print("Binary mask check: PASSED")

print("\nCHECK 5 — Dataset structure")
print("Unique stations:", merged["station_id"].nunique())
print("Unique timestamps:", merged["timestamp"].nunique())
print("Feature count:", len(FEATURES))

assert merged["station_id"].nunique() == 16
assert merged["timestamp"].nunique() == 26304

print("\nIMPORTANT:")
print("The mask contains no station_id or timestamp identifiers.")
print("Therefore, exact row-by-row mask alignment cannot be proven")
print("from dimensions and binary values alone.")
print("No input files or model checkpoints were modified.")

print("\nSTEP 8.8 COMPLETED")
