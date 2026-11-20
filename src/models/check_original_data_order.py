from pathlib import Path
import pandas as pd

# ============================================
# STEP 8.6 — CHECK ORIGINAL DATA ORDER
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "Data" / "raw data"

AIR_FILE = RAW_DIR / "air_quality_2019_2021_hourly.csv"
MET_FILE = RAW_DIR / "hourly_meteorology_16stations_2019_2021.csv"

air = pd.read_csv(AIR_FILE)
met = pd.read_csv(MET_FILE)

air["timestamp"] = pd.to_datetime(air["timestamp"])
met["timestamp"] = pd.to_datetime(met["timestamp"])

print("=" * 60)
print("STEP 8.6 — ORIGINAL DATA ORDER CHECK")
print("=" * 60)

# 1. Inspect original air-quality order
print("\nFirst 10 rows of original air-quality data:")
print(
    air[["timestamp", "station_id"]]
    .head(10)
    .to_string(index=False)
)

# 2. Inspect original weather order
print("\nFirst 10 rows of original weather data:")
print(
    met[["timestamp", "station_id"]]
    .head(10)
    .to_string(index=False)
)

# 3. Check whether both raw files have identical keys in row order
same_raw_order = (
    len(air) == len(met)
    and air[["station_id", "timestamp"]]
    .reset_index(drop=True)
    .equals(
        met[["station_id", "timestamp"]]
        .reset_index(drop=True)
    )
)

print("\nAir and weather original row order matches:", same_raw_order)

# 4. Inspect station IDs and timestamps at the beginning
print("\nFirst 20 station IDs in original air data:")
print(air["station_id"].head(20).tolist())

print("\nFirst 20 timestamps in original air data:")
print(air["timestamp"].head(20).tolist())

# 5. Check if the original air data is sorted by timestamp and station
air_sorted = air.sort_values(
    ["timestamp", "station_id"],
    kind="mergesort"
).reset_index(drop=True)

already_timestamp_sorted = (
    air[["timestamp", "station_id"]]
    .reset_index(drop=True)
    .equals(
        air_sorted[["timestamp", "station_id"]]
    )
)

print("\nAir data already timestamp/station sorted:", already_timestamp_sorted)

print("\nNo files modified.")
print("STEP 8.6 COMPLETED")