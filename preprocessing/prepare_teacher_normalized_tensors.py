import numpy as np
import os

# ============================================================
# TEACHER TRANSFORMER — NORMALIZED TENSOR PREPARATION
# Original tensors will NOT be modified.
# ============================================================

path = "Data/processed/teacher_tensors/"

# Load original tensors
x = np.load(path + "model_input_tensor.npy")
y = np.load(path + "ground_truth_tensor.npy")
mask = np.load(path + "observation_mask_tensor.npy")

# Load training-fitted scaler
scaler = np.load(path + "teacher_scaler_stats.npz")

mean = scaler["mean"]
std = scaler["std"]

# Verify dimensions
assert x.shape == y.shape == mask.shape
assert x.shape[-1] == len(mean) == len(std)

# ------------------------------------------------------------
# 1. Normalize observed model inputs
# ------------------------------------------------------------

normalized_x = np.zeros_like(x, dtype=np.float32)

observed_positions = (
    (mask == 1) &
    np.isfinite(x) &
    np.isfinite(y)
)

transformed_x = (x - mean) / std

normalized_x[observed_positions] = (
    transformed_x[observed_positions]
)

# Missing and artificially masked inputs remain zero.

# ------------------------------------------------------------
# 2. Normalize ground truth
# ------------------------------------------------------------

normalized_y = np.full_like(y, np.nan, dtype=np.float32)

valid_ground_truth = np.isfinite(y)

transformed_y = (y - mean) / std

normalized_y[valid_ground_truth] = (
    transformed_y[valid_ground_truth]
)

# ------------------------------------------------------------
# 3. Save separate normalized tensors
# ------------------------------------------------------------

input_save_path = path + "teacher_model_input_normalized.npy"
target_save_path = path + "teacher_ground_truth_normalized.npy"

np.save(input_save_path, normalized_x)
np.save(target_save_path, normalized_y)

# ------------------------------------------------------------
# 4. Verification
# ------------------------------------------------------------

print("NORMALIZED TENSORS CREATED SUCCESSFULLY")

print("\nInput shape:", normalized_x.shape)
print("Ground truth shape:", normalized_y.shape)

print(
    "Input NaN count:",
    np.isnan(normalized_x).sum()
)

print(
    "Ground truth NaN count:",
    np.isnan(normalized_y).sum()
)

print(
    "Masked input values remain zero:",
    np.all(normalized_x[mask == 0] == 0)
)

print(
    "Observation mask unchanged:",
    np.array_equal(mask, np.load(path + "observation_mask_tensor.npy"))
)

assert np.isfinite(normalized_x).all()
assert np.array_equal(
    np.isnan(normalized_y),
    np.isnan(y)
)
assert np.all(normalized_x[mask == 0] == 0)

print("\nNORMALIZATION VERIFICATION: PASSED")

print("\nSaved files:")
print(input_save_path)
print(target_save_path)