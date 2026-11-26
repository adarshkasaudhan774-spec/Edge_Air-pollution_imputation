
# ============================================================
# TEACHER TRANSFORMER — FORWARD PASS AND GRADIENT TEST
# ============================================================

from pathlib import Path
import sys
import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# STEP 1 — PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.teacher_dataset import TeacherTransformerDataset
from src.models.teacher_transformer import TeacherTransformer


# ============================================================
# STEP 2 — LOAD DATA
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "teacher_tensors"
)

model_input = np.load(
    DATA_DIR / "teacher_model_input_normalized.npy"
)

ground_truth = np.load(
    DATA_DIR / "teacher_ground_truth_normalized.npy"
)

observation_mask = np.load(
    DATA_DIR / "observation_mask_tensor.npy"
)

target_mask = (
    (observation_mask == 0)
    & np.isfinite(ground_truth)
).astype(np.float32)


# ============================================================
# STEP 3 — CREATE TRAINING DATASET
# ============================================================

train_dataset = TeacherTransformerDataset(
    model_input=model_input,
    ground_truth=ground_truth,
    observation_mask=observation_mask,
    target_mask=target_mask,
    start_idx=0,
    end_idx=18412,
    window_size=24
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# STEP 4 — GET ONE BATCH
# ============================================================

x, input_mask, y, y_mask = next(iter(train_loader))

device = torch.device("cpu")

x = x.to(device)
input_mask = input_mask.to(device)
y = y.to(device)
y_mask = y_mask.to(device)


# ============================================================
# STEP 5 — AUTOMATICALLY DETECT MODEL PARAMETERS
# ============================================================

constructor_signature = inspect.signature(
    TeacherTransformer.__init__
)

print("\n========== MODEL CONSTRUCTOR ==========")

print(constructor_signature)


# Parameter aliases and their intended values
parameter_values = {

    # Input features
    "num_features": 13,
    "input_features": 13,
    "in_features": 13,
    "feature_dim": 13,
    "input_dim": 13,

    # Transformer embedding
    "d_model": 128,
    "hidden_dim": 128,
    "hidden_size": 128,
    "embed_dim": 128,
    "embedding_dim": 128,

    # Attention heads
    "nhead": 4,
    "num_heads": 4,
    "n_heads": 4,
    "heads": 4,

    # Transformer layers
    "num_layers": 3,
    "n_layers": 3,
    "num_encoder_layers": 3,
    "encoder_layers": 3,

    # Feedforward
    "dim_feedforward": 256,
    "feedforward_dim": 256,
    "ff_dim": 256,
    "d_ff": 256,

    # Dropout
    "dropout": 0.1,

    # Window
    "window_size": 24,
    "seq_len": 24,
    "sequence_length": 24,

    # Stations
    "num_stations": 16,
    "n_stations": 16,
    "stations": 16,
}


# ============================================================
# STEP 6 — BUILD COMPATIBLE ARGUMENTS
# ============================================================

constructor_kwargs = {}

for parameter_name, parameter in constructor_signature.parameters.items():

    if parameter_name == "self":
        continue

    if parameter_name in parameter_values:

        constructor_kwargs[parameter_name] = (
            parameter_values[parameter_name]
        )

    elif parameter.default is inspect.Parameter.empty:

        raise TypeError(
            f"Unknown required model parameter: "
            f"{parameter_name}. "
            f"Please check teacher_transformer.py."
        )


print("\nDetected constructor arguments:")

for key, value in constructor_kwargs.items():
    print(f"{key}: {value}")


# ============================================================
# STEP 7 — INITIALIZE MODEL
# ============================================================

model = TeacherTransformer(
    **constructor_kwargs
).to(device)

print("\nModel initialized successfully!")


# ============================================================
# STEP 8 — FORWARD PASS TEST
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        x,
        input_mask
    )


# ============================================================
# STEP 9 — OUTPUT VERIFICATION
# ============================================================

print("\n========== FORWARD PASS ==========")

print("Input shape:", x.shape)
print("Input mask shape:", input_mask.shape)
print("Target shape:", y.shape)
print("Target mask shape:", y_mask.shape)

print("Prediction shape:", predictions.shape)

print(
    "Predictions finite:",
    torch.isfinite(predictions).all().item()
)


# ============================================================
# STEP 10 — MODEL PARAMETERS
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n========== MODEL PARAMETERS ==========")

print("Total parameters:", total_parameters)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ============================================================
# STEP 11 — GRADIENT VERIFICATION
# ============================================================

model.train()

model.zero_grad()

predictions = model(
    x,
    input_mask
)

valid_mask = y_mask.bool()

if valid_mask.any():

    loss = (
        (predictions - y) ** 2
    )[valid_mask].mean()

    loss.backward()

else:

    raise RuntimeError(
        "No valid artificially masked targets found."
    )


gradient_exists = any(
    p.grad is not None
    for p in model.parameters()
    if p.requires_grad
)


print("\n========== GRADIENT TEST ==========")

print("Test loss:", loss.item())

print(
    "Gradient exists:",
    gradient_exists
)


# ============================================================
# STEP 12 — FINAL ASSERTIONS
# ============================================================

assert predictions.shape == (32, 16, 13)

assert torch.isfinite(predictions).all()

assert gradient_exists

assert total_parameters > 0

print("\nTEACHER TRANSFORMER FORWARD PASS: PASSED")
