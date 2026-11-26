
# ============================================================
# EDGE-EFFICIENT AIR POLLUTION IMPUTATION
# TEACHER TRANSFORMER TRAINING SCRIPT
# ============================================================

from pathlib import Path
import sys
import inspect
import random
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
# STEP 2 — REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# STEP 3 — DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training device:", DEVICE)


# ============================================================
# STEP 4 — DATA PATHS
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "teacher_tensors"
)

MODEL_DIR = PROJECT_ROOT / "Models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    MODEL_DIR / "teacher_transformer_best.pth"
)


# ============================================================
# STEP 5 — LOAD NORMALIZED DATA
# ============================================================

print("\nLoading normalized tensors...")

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
# STEP 6 — CREATE TARGET MASK
# ============================================================

target_mask = (
    (observation_mask == 0)
    & np.isfinite(ground_truth)
).astype(np.float32)


# ============================================================
# STEP 7 — CHRONOLOGICAL SPLIT
# ============================================================

TRAIN_END = 18412
VAL_END = 22358
TOTAL_TIMESTAMPS = 26304

WINDOW_SIZE = 24

print("\nDataset boundaries:")
print("Train:", 0, "to", TRAIN_END - 1)
print("Validation:", TRAIN_END, "to", VAL_END - 1)
print("Test:", VAL_END, "to", TOTAL_TIMESTAMPS - 1)


# ============================================================
# STEP 8 — CREATE TRAIN DATASET
# ============================================================

train_dataset = TeacherTransformerDataset(
    model_input=model_input,
    ground_truth=ground_truth,
    observation_mask=observation_mask,
    target_mask=target_mask,
    start_idx=0,
    end_idx=TRAIN_END,
    window_size=WINDOW_SIZE
)


# ============================================================
# STEP 9 — CREATE VALIDATION DATASET
# ============================================================

val_dataset = TeacherTransformerDataset(
    model_input=model_input,
    ground_truth=ground_truth,
    observation_mask=observation_mask,
    target_mask=target_mask,
    start_idx=TRAIN_END,
    end_idx=VAL_END,
    window_size=WINDOW_SIZE
)


# ============================================================
# STEP 10 — CREATE DATALOADERS
# ============================================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# STEP 11 — DISPLAY DATASET INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# STEP 12 — MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "num_features": 13,

    "d_model": 128,

    "nhead": 4,

    "num_layers": 3,

    "dim_feedforward": 256,

    "dropout": 0.1,

    "window_size": 24,

    "num_stations": 16
}


# ============================================================
# STEP 13 — DETECT MODEL CONSTRUCTOR PARAMETERS
# ============================================================

constructor_signature = inspect.signature(
    TeacherTransformer.__init__
)

parameter_values = {

    "num_features": 13,
    "input_features": 13,
    "in_features": 13,
    "feature_dim": 13,
    "input_dim": 13,

    "d_model": 128,
    "hidden_dim": 128,
    "hidden_size": 128,
    "embed_dim": 128,
    "embedding_dim": 128,

    "nhead": 4,
    "num_heads": 4,
    "n_heads": 4,
    "heads": 4,

    "num_layers": 3,
    "n_layers": 3,
    "num_encoder_layers": 3,
    "encoder_layers": 3,

    "dim_feedforward": 256,
    "feedforward_dim": 256,
    "ff_dim": 256,
    "d_ff": 256,

    "dropout": 0.1,

    "window_size": 24,
    "seq_len": 24,
    "sequence_length": 24,

    "num_stations": 16,
    "n_stations": 16,
    "stations": 16
}


constructor_kwargs = {}

for parameter_name, parameter in (
    constructor_signature.parameters.items()
):

    if parameter_name == "self":
        continue

    if parameter_name in parameter_values:

        constructor_kwargs[parameter_name] = (
            parameter_values[parameter_name]
        )

    elif parameter.default == inspect.Parameter.empty:

        raise TypeError(
            f"Unknown required constructor parameter: "
            f"{parameter_name}"
        )


# ============================================================
# STEP 14 — INITIALIZE TEACHER MODEL
# ============================================================

model = TeacherTransformer(
    **constructor_kwargs
).to(DEVICE)


print("\n========== MODEL INFORMATION ==========")

print(model)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Total parameters:", total_parameters)


# ============================================================
# STEP 15 — OPTIMIZER
# ============================================================

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# STEP 16 — TRAINING CONFIGURATION
# ============================================================

MAX_EPOCHS = 50

PATIENCE = 7

GRADIENT_CLIP = 1.0


# ============================================================
# STEP 17 — MASKED MSE LOSS
# ============================================================

def masked_mse_loss(predictions, targets, masks):

    masks = masks.bool()

    if not masks.any():

        raise RuntimeError(
            "Batch contains no valid masked targets."
        )

    squared_errors = (
        (predictions - targets) ** 2
    )

    loss = squared_errors[masks].mean()

    return loss


# ============================================================
# STEP 18 — TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, optimizer):

    model.train()

    total_squared_error = 0.0

    total_valid_targets = 0

    for x, input_mask, y, y_mask in loader:

        x = x.to(DEVICE)

        input_mask = input_mask.to(DEVICE)

        y = y.to(DEVICE)

        y_mask = y_mask.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            x,
            input_mask
        )

        loss = masked_mse_loss(
            predictions,
            y,
            y_mask
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        valid_count = y_mask.bool().sum().item()

        total_squared_error += (
            loss.item() * valid_count
        )

        total_valid_targets += valid_count


    if total_valid_targets == 0:

        raise RuntimeError(
            "No valid training targets found."
        )


    epoch_loss = (
        total_squared_error
        / total_valid_targets
    )

    return epoch_loss


# ============================================================
# STEP 19 — VALIDATE ONE EPOCH
# ============================================================

def validate_one_epoch(model, loader):

    model.eval()

    total_squared_error = 0.0

    total_valid_targets = 0

    with torch.no_grad():

        for x, input_mask, y, y_mask in loader:

            x = x.to(DEVICE)

            input_mask = input_mask.to(DEVICE)

            y = y.to(DEVICE)

            y_mask = y_mask.to(DEVICE)

            predictions = model(
                x,
                input_mask
            )

            loss = masked_mse_loss(
                predictions,
                y,
                y_mask
            )

            valid_count = (
                y_mask.bool().sum().item()
            )

            total_squared_error += (
                loss.item() * valid_count
            )

            total_valid_targets += valid_count


    if total_valid_targets == 0:

        raise RuntimeError(
            "No valid validation targets found."
        )


    validation_loss = (
        total_squared_error
        / total_valid_targets
    )

    return validation_loss


# ============================================================
# STEP 20 — TRAINING LOOP WITH EARLY STOPPING
# ============================================================

best_val_loss = float("inf")

epochs_without_improvement = 0

history = []

print("\n========== TEACHER TRAINING ==========")

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss = validate_one_epoch(
        model,
        val_loader
    )


    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "val_loss": val_loss

    })


    print(

        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "

        f"Train Loss: {train_loss:.6f} | "

        f"Val Loss: {val_loss:.6f}"

    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(

            {

                "epoch": epoch,

                "model_state_dict": (
                    model.state_dict()
                ),

                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),

                "best_val_loss": (
                    best_val_loss
                ),

                "model_config": (
                    MODEL_CONFIG
                ),

                "seed": SEED

            },

            CHECKPOINT_PATH

        )


        print(
            "Best model checkpoint saved."
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# STEP 21 — TRAINING SUMMARY
# ============================================================

print("\n========== TRAINING COMPLETED ==========")

print("Epochs completed:", len(history))

print("Best validation loss:", best_val_loss)

print("Checkpoint path:", CHECKPOINT_PATH)

print("\nTEACHER TRAINING SCRIPT COMPLETED")
