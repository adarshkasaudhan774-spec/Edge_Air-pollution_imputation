from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

AIR_FILE = ROOT / "Data" / "raw data" / "air_quality_2019_2021_hourly.csv"
WEATHER_FILE = ROOT / "Data" / "raw data" / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = ROOT / "Data" / "processed" / "fixed_artificial_mask_new.csv"

print("=" * 60)
print("STEP 8.7 — MASK ORDER CHECK")
print("=" * 60)

# Load files without modifying them
air = pd.read_csv(AIR_FILE)
weather = pd.read_csv(WEATHER_FILE)
mask = pd.read_csv(MASK_FILE)

# Parse timestamps
air["timestamp"] = pd.to_datetime(air["timestamp"])
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

# Check whether air and weather have identical row order
keys = ["station_id", "timestamp"]

air_keys = air[keys].reset_index(drop=True)
weather_keys = weather[keys].reset_index(drop=True)

print("\n1. Row counts")
print("Air:", len(air))
print("Weather:", len(weather))
print("Mask:", len(mask))

print("\n2. Air/weather original row order matches:",
      air_keys.equals(weather_keys))

print("\n3. Mask columns:")
print(mask.columns.tolist())

print("\n4. Mask shape:", mask.shape)
print("Mask contains only 0 and 1:",
      set(np.unique(mask.to_numpy())).issubset({0, 1}))

print("\n5. Original air data station transitions:")
station_ids = air["station_id"]
transition_positions = np.flatnonzero(
    station_ids.to_numpy()[1:] != station_ids.to_numpy()[:-1]
)

print("Number of station transitions:", len(transition_positions))
print("First transition positions:", transition_positions[:10].tolist())

print("\n6. IMPORTANT")
print("The mask has no station_id or timestamp columns.")
print("Therefore, its exact row-by-row alignment cannot be")
print("proven from the mask file alone.")

print("\nNo files were modified.")
print("STEP 8.7.1 COMPLETED")