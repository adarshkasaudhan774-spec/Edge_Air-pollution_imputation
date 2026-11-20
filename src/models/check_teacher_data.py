from pathlib import Path
import pandas as pd
import numpy as np

# ============================================
# STEP 8.3 — VERIFY TEACHER TRANSFORMER DATA
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AIR_FILE = PROJECT_ROOT / "Data" / "raw data" / "air_quality_2019_2021_hourly.csv"
WEATHER_FILE = PROJECT_ROOT / "Data" / "raw data" / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = PROJECT_ROOT / "Data" / "processed" / "fixed_artificial_mask_new.csv"

print("=" * 60)
print("STEP 8.3 — TEACHER DATA VERIFICATION")
print("=" * 60)

# 1. Check that files exist
for file_path in [AIR_FILE, WEATHER_FILE, MASK_FILE]:
    print(f"{file_path.name}: {'FOUND' if file_path.exists() else 'NOT FOUND'}")

if not all(path.exists() for path in [AIR_FILE, WEATHER_FILE, MASK_FILE]):
    raise FileNotFoundError("One or more required files are missing.")

# 2. Load existing files without modifying them
air = pd.read_csv(AIR_FILE)
weather = pd.read_csv(WEATHER_FILE)
mask = pd.read_csv(MASK_FILE)

print("\n--- Dataset Shapes ---")
print("Air-quality:", air.shape)
print("Weather:", weather.shape)
print("Fixed mask:", mask.shape)

# 3. Display columns
print("\n--- Air-quality Columns ---")
print(air.columns.tolist())

print("\n--- Weather Columns ---")
print(weather.columns.tolist())

print("\n--- Mask Columns ---")
print(mask.columns.tolist())

# 4. Identify keys
required_keys = {"station_id", "timestamp"}

for name, frame in [("Air-quality", air), ("Weather", weather)]:
    missing_keys = required_keys - set(frame.columns)

    if missing_keys:
        raise ValueError(f"{name} dataset is missing keys: {missing_keys}")

# 5. Normalize timestamps for comparison
air["timestamp"] = pd.to_datetime(air["timestamp"], errors="coerce")
weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")

print("\n--- Timestamp Verification ---")
print("Invalid air timestamps:", air["timestamp"].isna().sum())
print("Invalid weather timestamps:", weather["timestamp"].isna().sum())

# 6. Check duplicate station-time rows
print("\n--- Duplicate Verification ---")
print(
    "Air duplicates:",
    air.duplicated(["station_id", "timestamp"]).sum()
)
print(
    "Weather duplicates:",
    weather.duplicated(["station_id", "timestamp"]).sum()
)

# 7. Compare row ordering
same_order = (
    len(air) == len(weather)
    and air[["station_id", "timestamp"]].reset_index(drop=True).equals(
        weather[["station_id", "timestamp"]].reset_index(drop=True)
    )
)

print("\nAir/weather keys in identical row order:", same_order)

# 8. Verify mask values
print("\n--- Fixed Mask Verification ---")
mask_values = set(pd.unique(mask.to_numpy().ravel()))
print("Unique mask values:", mask_values)
print("Mask contains only 0 and 1:", mask_values.issubset({0, 1}))
print("Mask rows match air rows:", len(mask) == len(air))

# 9. Final warning
print("\n--- Important Alignment Check ---")

if len(mask) == len(air):
    print(
        "Row counts match, but the mask has no station/timestamp keys "
        "to independently prove row-by-row alignment."
    )
else:
    print(
        "Mask and air-quality row counts differ. "
        "Do not proceed to training until this is resolved."
    )

print("\nSTEP 8.3 VERIFICATION FINISHED")