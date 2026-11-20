import numpy as np
import os

# ============================================================
# TEACHER TRANSFORMER — TRAINING DATA SCALER
# ============================================================

path = "Data/processed/teacher_tensors/"

y = np.load(path + "ground_truth_tensor.npy")
mask = np.load(path + "observation_mask_tensor.npy")

train_end = 18412

y_train = y[:train_end]
mask_train = mask[:train_end]

feature_names = [
    "PM2.5", "PM10", "NO2", "O3", "SO2", "NOx", "CO",
    "Temperature", "Relative Humidity", "Wind Speed",
    "Wind Direction", "Pressure", "Rainfall"
]

# Valid observed training values only
valid_train = (
    (mask_train == 1) &
    np.isfinite(y_train)
)

num_features = y.shape[-1]

feature_mean = np.zeros(num_features, dtype=np.float32)
feature_std = np.zeros(num_features, dtype=np.float32)
feature_count = np.zeros(num_features, dtype=np.int64)

for i in range(num_features):

    values = y_train[:, :, i][valid_train[:, :, i]]

    feature_mean[i] = values.mean()
    feature_std[i] = values.std()
    feature_count[i] = len(values)

# Validate scaler statistics
assert np.isfinite(feature_mean).all()
assert np.isfinite(feature_std).all()
assert (feature_std > 0).all()

# Save scaler statistics
save_path = os.path.join(
    path,
    "teacher_scaler_stats.npz"
)

np.savez(
    save_path,
    mean=feature_mean,
    std=feature_std,
    count=feature_count,
    feature_names=np.array(feature_names),
    train_end=np.array(train_end)
)

print("TEACHER SCALER CREATED SUCCESSFULLY")
print("Saved at:", save_path)

print("\nFeature-wise scaler statistics:")

for i, name in enumerate(feature_names):
    print(
        f"{name:22s} "
        f"Mean={feature_mean[i]:10.4f} "
        f"Std={feature_std[i]:10.4f} "
        f"Count={feature_count[i]}"
    )