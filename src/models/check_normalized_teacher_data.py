import numpy as np

# ============================================================
# TEACHER TRANSFORMER — NORMALIZED DATA ALIGNMENT CHECK
# ============================================================

path = "Data/processed/teacher_tensors/"

x = np.load(path + "teacher_model_input_normalized.npy")
y = np.load(path + "teacher_ground_truth_normalized.npy")
mask = np.load(path + "observation_mask_tensor.npy")

# 1. Shape verification
assert x.shape == y.shape == mask.shape

# 2. Input finite values
assert np.isfinite(x).all()

# 3. Ground truth missing positions preserved
assert np.isnan(y).sum() == 284111

# 4. Observed input positions
observed = (
    (mask == 1) &
    np.isfinite(y)
)

# 5. Masked input positions must be zero
masked = (mask == 0)

assert np.all(x[masked] == 0)

# 6. Normalized input and target alignment
assert np.allclose(
    x[observed],
    y[observed],
    atol=1e-5
)

print("Input shape:", x.shape)
print("Valid observed values:", observed.sum())
print("Masked input positions:", masked.sum())

print("\nAll alignment checks passed.")
print("NORMALIZED TEACHER DATA ALIGNMENT: PASSED")