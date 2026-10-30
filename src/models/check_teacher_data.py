print("=" * 60)
print("STEP 8.1 - TEACHER TRANSFORMER DATA VERIFICATION")
print("=" * 60)

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "Data" / "raw data"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

AIR_FILE = RAW_DIR / "air_quality_2019_2021_hourly.csv"
MET_FILE = RAW_DIR / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = PROCESSED_DIR / "fixed_artificial_mask_new.csv"

print("\nChecking required files...")

for file_path in [AIR_FILE, MET_FILE, MASK_FILE]:
    print(f"\nFile: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print("File exists: YES")

print("\nLoading datasets...")

air_df = pd.read_csv(AIR_FILE)
met_df = pd.read_csv(MET_FILE)
mask_df = pd.read_csv(MASK_FILE)

print("\nDataset shapes:")
print("Air quality:", air_df.shape)
print("Meteorology:", met_df.shape)
print("Artificial mask:", mask_df.shape)

print("\nAir quality columns:")
print(air_df.columns.tolist())

print("\nMeteorology columns:")
print(met_df.columns.tolist())

print("\nMask columns:")
print(mask_df.columns.tolist())

for df in [air_df, met_df]:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

print("\nTimestamp verification:")
print("Air timestamps:", air_df["timestamp"].nunique())
print("Meteorology timestamps:", met_df["timestamp"].nunique())

print("\nStation verification:")
print("Air stations:", air_df["station_id"].nunique())
print("Meteorology stations:", met_df["station_id"].nunique())

print("\nDuplicate rows:")
print(
    "Air duplicates:",
    air_df.duplicated(["station_id", "timestamp"]).sum()
)
print(
    "Meteorology duplicates:",
    met_df.duplicated(["station_id", "timestamp"]).sum()
)

print("\nMask preview:")
print(mask_df.head())

print("\nSTEP 8.1 VERIFICATION SCRIPT COMPLETED")
