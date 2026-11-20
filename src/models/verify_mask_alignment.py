from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

AIR_FILE = ROOT / "Data" / "raw data" / "air_quality_2019_2021_hourly.csv"
WEATHER_FILE = ROOT / "Data" / "raw data" / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = ROOT / "Data" / "processed" / "fixed_artificial_mask_new.csv"

FEATURES = [
    "pm25", "pm10", "no2", "o3", "so2", "nox", "co",
    "temperature", "relative_humidity", "wind_speed",
    "wind_direction", "pressure", "rainfall"
]

print("=" * 65)
print("STEP 8.11 — VERIFY MASK ROW ORDER")
print("=" * 65)

air = pd.read_csv(AIR_FILE)
weather = pd.read_csv(WEATHER_FILE)
mask = pd.read_csv(MASK_FILE)

air["timestamp"] = pd.to_datetime(air["timestamp"])
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

keys = ["station_id", "station_name", "timestamp"]

if not air[keys].reset_index(drop=True).equals(
    weather[keys].reset_index(drop=True)
):
    raise ValueError("Air and weather row orders do not match.")

merged = pd.merge(
    air,
    weather,
    on=keys,
    how="inner",
    validate="one_to_one",
    sort=False
).reset_index(drop=True)

print("\nMerged shape:", merged.shape)
print("Mask shape:", mask.shape)

if len(merged) != len(mask):
    raise ValueError("Merged data and mask row counts differ.")

if list(mask.columns) != FEATURES:
    raise ValueError("Mask feature columns or order are incorrect.")

mask_values = mask.to_numpy()

if not np.isin(mask_values, [0, 1]).all():
    raise ValueError("Mask contains values other than 0 and 1.")

# Compare the current merge order with station/time sorting.
sorted_merged = merged.sort_values(
    ["station_id", "timestamp"],
    kind="mergesort"
).reset_index(drop=True)

original_keys = merged[keys].reset_index(drop=True)
sorted_keys = sorted_merged[keys].reset_index(drop=True)

same_order = original_keys.equals(sorted_keys)

print("\nOriginal merge order equals station/time sorted order:", same_order)

if same_order:
    print("The merge is already in the target sorting order.")
else:
    print("The merge order changes when station/time sorting is applied.")

# Compare masks by feature-wise values after sorting.
# This comparison tests whether sorting the mask rows changes their values.
sorted_index = merged.sort_values(
    ["station_id", "timestamp"],
    kind="mergesort"
).index

mask_sorted = mask.iloc[sorted_index].reset_index(drop=True)
mask_unsorted = mask.reset_index(drop=True)

same_mask_values_after_sort = mask_sorted.equals(mask_unsorted)

print(
    "Mask values unchanged by sorting:",
    same_mask_values_after_sort
)

print("\nImportant limitation:")
print(
    "The mask CSV contains no station/timestamp keys. "
    "This script can test ordering consistency, but cannot independently "
    "prove which station/timestamp each mask row originally belonged to."
)

print("\nNo source data, mask, or model checkpoint was modified.")
print("STEP 8.11 DIAGNOSTIC COMPLETED")