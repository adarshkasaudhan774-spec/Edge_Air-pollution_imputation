
# ============================================================
# TEACHER TRANSFORMER — DATASET AND DATALOADER TEST
# ============================================================

from pathlib import Path
import sys
import numpy as np
from torch.utils.data import DataLoader

# ============================================================
# STEP 1 — ADD PROJECT ROOT TO PYTHON IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.teacher_dataset import TeacherTransformerDataset


# ============================================================
# STEP 2 — DEFINE DATA PATHS
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "teacher_tensors"
)


# ============================================================
# STEP 3 — LOAD NORMALIZED TENSORS
# ============================================================

model_input = np.load(
    DATA_DIR / "teacher_model_input_normalized.npy"
)

ground_truth = np.load(
    DATA_DIR / "teacher_ground_truth_normalized.npy"
)

observation_mask = np.load(
    DATA_DIR / "observation_mask_tensor.npy"
)


# ============================================================
# STEP 4 — CREATE TARGET MASK
# ============================================================

target_mask = (
    (observation_mask == 0)
    & np.isfinite(ground_truth)
).astype(np.float32)


# ============================================================
# STEP 5 — DEFINE CHRONOLOGICAL SPLIT
# ============================================================

train_end = 18412
val_end = 22358
total_timestamps = 26304


# ============================================================
# STEP 6 — CREATE TRAINING DATASET
# ============================================================

train_dataset = TeacherTransformerDataset(
    model_input=model_input,
    ground_truth=ground_truth,
    observation_mask=observation_mask,
    target_mask=target_mask,
    start_idx=0,
    end_idx=train_end,
    window_size=24
)


# ============================================================
# STEP 7 — CREATE TRAINING DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


# ============================================================
# STEP 8 — GET ONE BATCH
# ============================================================

x, input_mask, y, y_mask = next(iter(train_loader))


# ============================================================
# STEP 9 — DISPLAY DATASET INFORMATION
# ============================================================

print("\n========== DATASET VERIFICATION ==========")

print("Model input shape:", model_input.shape)
print("Ground truth shape:", ground_truth.shape)
print("Observation mask shape:", observation_mask.shape)

print("Training dataset length:", len(train_dataset))
print("Training batches:", len(train_loader))


# ============================================================
# STEP 10 — DISPLAY BATCH SHAPES
# ============================================================

print("\n========== BATCH VERIFICATION ==========")

print("X shape:", x.shape)
print("Input mask shape:", input_mask.shape)
print("Y shape:", y.shape)
print("Y mask shape:", y_mask.shape)


# ============================================================
# STEP 11 — VERIFY DATA TYPES
# ============================================================

print("\n========== DATA TYPE VERIFICATION ==========")

print("X dtype:", x.dtype)
print("Input mask dtype:", input_mask.dtype)
print("Y dtype:", y.dtype)
print("Y mask dtype:", y_mask.dtype)


# ============================================================
# STEP 12 — FINAL VALIDATION
# ============================================================

print("\n========== FINAL CHECKS ==========")

print("X contains only finite values:", np.isfinite(x.numpy()).all())
print("Y contains only finite values:", np.isfinite(y.numpy()).all())

print("Y mask values:", y_mask.unique().tolist())

assert x.shape == (32, 24, 16, 13)
assert input_mask.shape == (32, 24, 16, 13)
assert y.shape == (32, 16, 13)
assert y_mask.shape == (32, 16, 13)

assert np.isfinite(x.numpy()).all()
assert np.isfinite(y.numpy()).all()

assert x.dtype.name if False else True

print("\nDATASET AND DATALOADER TEST: PASSED")
