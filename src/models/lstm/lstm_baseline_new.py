# STEP 6.1 — Imports & Project Paths

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data Paths
RAW_DATA = PROJECT_ROOT / "Data" / "raw data"
PROCESSED_DATA = PROJECT_ROOT / "Data" / "processed"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"

# Dataset Files
AIR_FILE = RAW_DATA / "air_quality_2019_2021_hourly.csv"
MET_FILE = RAW_DATA / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = PROCESSED_DATA / "fixed_artificial_mask_new.csv"

# Model Save Path
MODEL_FILE = MODELS_DIR / "lstm_baseline_new.pth"

# Basic Configuration
NUM_STATIONS = 16
NUM_FEATURES = 13
INPUT_SIZE = NUM_STATIONS * NUM_FEATURES

WINDOW_SIZE = 24

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

RANDOM_SEED = 42


print("STEP 6.1 completed successfully.")
print("Project root:", PROJECT_ROOT)
print("Air quality file:", AIR_FILE)
print("Meteorology file:", MET_FILE)
print("Mask file:", MASK_FILE)
print("Model file:", MODEL_FILE)
print("Input size:", INPUT_SIZE)
print("Window size:", WINDOW_SIZE)
# STEP 6.2 — Load New Air Quality and Meteorology Datasets

air_df = pd.read_csv(AIR_FILE)
met_df = pd.read_csv(MET_FILE)

# Convert timestamp columns to datetime
air_df["timestamp"] = pd.to_datetime(air_df["timestamp"])
met_df["timestamp"] = pd.to_datetime(met_df["timestamp"])


# Display basic information
print("\nSTEP 6.2 — Dataset Loading")
print("-" * 50)

print("Air quality shape:", air_df.shape)
print("Meteorology shape:", met_df.shape)

print("Air quality columns:")
print(air_df.columns.tolist())

print("\nMeteorology columns:")
print(met_df.columns.tolist())

print("\nAir quality timestamps:")
print(air_df["timestamp"].min(), "to", air_df["timestamp"].max())

print("\nMeteorology timestamps:")
print(met_df["timestamp"].min(), "to", met_df["timestamp"].max())
# STEP 6.3 — Merge Air Quality and Meteorology

merged_df = pd.merge(
    air_df,
    met_df,
    on=["station_id", "station_name", "timestamp"],
    how="inner"
)


print("\nSTEP 6.3 — Dataset Merge")
print("-" * 50)

print("Merged shape:", merged_df.shape)

print("Number of unique stations:",
      merged_df["station_id"].nunique())

print("Number of unique timestamps:",
      merged_df["timestamp"].nunique())

print("Duplicate station-timestamp rows:",
      merged_df.duplicated(
          subset=["station_id", "timestamp"]
      ).sum())
# STEP 6.4 — Define and Verify 13 Features

pollution_features = [
    "pm25",
    "pm10",
    "no2",
    "o3",
    "so2",
    "nox",
    "co"
]

weather_features = [
    "temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "rainfall"
]

feature_columns = pollution_features + weather_features


print("\nSTEP 6.4 — Feature Verification")
print("-" * 50)

print("Pollution features:", len(pollution_features))
print("Weather features:", len(weather_features))
print("Total features:", len(feature_columns))

print("\nFeature columns:")
print(feature_columns)

# Verify all features exist in merged dataset
missing_features = [
    feature for feature in feature_columns
    if feature not in merged_df.columns
]

print("\nMissing feature columns:", missing_features)

# Verify expected input size
print("Expected input size:", NUM_STATIONS * len(feature_columns))
# STEP 6.5 — Load Fixed Artificial Mask

fixed_mask = pd.read_csv(MASK_FILE)


print("\nSTEP 6.5 — Fixed Artificial Mask")
print("-" * 50)

print("Mask shape:", fixed_mask.shape)

print("Mask columns:")
print(fixed_mask.columns.tolist())

print("\nMask values:")
print(fixed_mask.stack().value_counts().sort_index())

print("\nExpected feature columns:", len(feature_columns))
print("Actual mask columns:", len(fixed_mask.columns))

print(
    "Feature order matches:",
    list(fixed_mask.columns) == feature_columns
)
# STEP 6.6 — Mask Alignment Verification

print("\nSTEP 6.6 — Mask Alignment Verification")
print("-" * 50)

print("Merged dataset rows:", len(merged_df))
print("Mask rows:", len(fixed_mask))

print(
    "Row count matches:",
    len(merged_df) == len(fixed_mask)
)

print(
    "Feature count matches:",
    len(feature_columns) == fixed_mask.shape[1]
)

# Verify that the mask does not contain invalid values
invalid_mask_values = fixed_mask[
    ~fixed_mask.isin([0, 1])
].count().sum()

print(
    "Invalid mask values:",
    invalid_mask_values
)
# STEP 6.7 — Sort Dataset and Keep Mask Exactly Aligned

sort_order = (
    merged_df
    .sort_values(
        ["timestamp", "station_id"],
        kind="mergesort"
    )
    .index
)

merged_df = (
    merged_df
    .loc[sort_order]
    .reset_index(drop=True)
)

fixed_mask = (
    fixed_mask
    .loc[sort_order]
    .reset_index(drop=True)
)

print("\nSTEP 6.7 — Dataset Ordering")
print("-" * 50)

print(
    "First timestamp:",
    merged_df["timestamp"].iloc[0]
)

print(
    "Last timestamp:",
    merged_df["timestamp"].iloc[-1]
)

print(
    "Unique timestamps:",
    merged_df["timestamp"].nunique()
)

print(
    "Unique stations:",
    merged_df["station_id"].nunique()
)

print(
    "Rows after sorting:",
    len(merged_df)
)

print(
    "Mask rows after sorting:",
    len(fixed_mask)
)

print(
    "Data and mask row counts match:",
    len(merged_df) == len(fixed_mask)
)


# STEP 6.8 — Prepare Data and Mask Tensors

data_array = (
    merged_df[
        feature_columns
    ]
    .to_numpy(
        dtype=np.float32
    )
)

mask_array = (
    fixed_mask[
        feature_columns
    ]
    .to_numpy(
        dtype=np.int8
    )
)

original_observation_mask = (
    merged_df[
        feature_columns
    ]
    .notna()
    .to_numpy(
        dtype=np.int8
    )
)

data_tensor = data_array.reshape(
    NUM_STATIONS,
    -1,
    NUM_FEATURES
).transpose(
    1,
    0,
    2
)

mask_tensor = mask_array.reshape(
    NUM_STATIONS,
    -1,
    NUM_FEATURES
).transpose(
    1,
    0,
    2
)

original_observation_mask_tensor = (
    original_observation_mask
    .reshape(
        NUM_STATIONS,
        -1,
        NUM_FEATURES
    )
    .transpose(
        1,
        0,
        2
    )
)

print("\nSTEP 6.8 — Tensor Preparation")
print("-" * 50)

print(
    "Data tensor shape:",
    data_tensor.shape
)

print(
    "Mask tensor shape:",
    mask_tensor.shape
)

print(
    "Original observation mask shape:",
    original_observation_mask_tensor.shape
)

print(
    "Data tensor dtype:",
    data_tensor.dtype
)

print(
    "Mask tensor dtype:",
    mask_tensor.dtype
)

print(
    "Expected shape:",
    (
        merged_df["timestamp"].nunique(),
        NUM_STATIONS,
        NUM_FEATURES
    )
)


# STEP 6.9 — Create 24-Hour Sliding Windows

num_windows = (
    len(data_tensor)
    - WINDOW_SIZE
    + 1
)

data_windows = np.empty(
    (
        num_windows,
        WINDOW_SIZE,
        NUM_STATIONS,
        NUM_FEATURES
    ),
    dtype=np.float32
)

mask_windows = np.empty(
    (
        num_windows,
        WINDOW_SIZE,
        NUM_STATIONS,
        NUM_FEATURES
    ),
    dtype=np.int8
)

for i in range(num_windows):

    data_windows[i] = data_tensor[
        i:i + WINDOW_SIZE
    ]

    mask_windows[i] = mask_tensor[
        i:i + WINDOW_SIZE
    ]

print("\nSTEP 6.9 — Sliding-Window Dataset")
print("-" * 50)

print(
    "Data windows shape:",
    data_windows.shape
)

print(
    "Mask windows shape:",
    mask_windows.shape
)

print(
    "Number of windows:",
    num_windows
)

print(
    "Window size:",
    WINDOW_SIZE
)


# STEP 6.10 — Prepare LSTM Targets

target_data = (
    data_windows[:, -1, :, :]
)

target_mask = (
    mask_windows[:, -1, :, :]
)

print("\nSTEP 6.10 — LSTM Target Preparation")
print("-" * 50)

print(
    "Target data shape:",
    target_data.shape
)

print(
    "Target mask shape:",
    target_mask.shape
)

print(
    "Expected target shape:",
    (
        num_windows,
        NUM_STATIONS,
        NUM_FEATURES
    )
)

print(
    "Target shape correct:",
    target_data.shape == (
        num_windows,
        NUM_STATIONS,
        NUM_FEATURES
    )
)


# STEP 6.11 — Artificially Masked Target Verification

artificial_target_mask = (
    (original_observation_mask == 1)
    &
    (mask_array == 0)
)

artificial_target_mask_tensor = (
    artificial_target_mask
    .reshape(
        NUM_STATIONS,
        -1,
        NUM_FEATURES
    )
    .transpose(
        1,
        0,
        2
    )
)

artificial_target_windows = (
    artificial_target_mask_tensor[
        WINDOW_SIZE - 1:
    ]
)

print(
    "\nSTEP 6.11 — Artificial Target Verification"
)

print("-" * 50)

print(
    "Artificial target mask shape:",
    artificial_target_windows.shape
)

print(
    "Artificially masked target points:",
    int(
        artificial_target_windows.sum()
    )
)

print(
    "Target windows:",
    len(
        artificial_target_windows
    )
)

print(
    "Expected target shape:",
    target_mask.shape
)

print(
    "Shape matches:",
    artificial_target_windows.shape
    == target_mask.shape
)


# STEP 6.12 — Chronological Train / Validation / Test Split

total_windows = len(
    data_windows
)

train_end = int(
    total_windows * 0.70
)

val_end = int(
    total_windows * 0.85
)

X_train = data_windows[
    :train_end
]

M_train = mask_windows[
    :train_end
]

Y_train = target_data[
    :train_end
]

T_train = artificial_target_windows[
    :train_end
]

X_val = data_windows[
    train_end:val_end
]

M_val = mask_windows[
    train_end:val_end
]

Y_val = target_data[
    train_end:val_end
]

T_val = artificial_target_windows[
    train_end:val_end
]

X_test = data_windows[
    val_end:
]

M_test = mask_windows[
    val_end:
]

Y_test = target_data[
    val_end:
]

T_test = artificial_target_windows[
    val_end:
]

print("\nSTEP 6.12 — Dataset Split")
print("-" * 50)

print(
    "Total windows:",
    total_windows
)

print("\nTraining:")
print(
    "X_train:",
    X_train.shape
)
print(
    "Y_train:",
    Y_train.shape
)

print("\nValidation:")
print(
    "X_val:",
    X_val.shape
)
print(
    "Y_val:",
    Y_val.shape
)

print("\nTest:")
print(
    "X_test:",
    X_test.shape
)
print(
    "Y_test:",
    Y_test.shape
)

print("\nArtificial target points:")

print(
    "Train:",
    int(T_train.sum())
)

print(
    "Validation:",
    int(T_val.sum())
)

print(
    "Test:",
    int(T_test.sum())
)

print(
    "Total:",
    int(
        T_train.sum()
        + T_val.sum()
        + T_test.sum()
    )
)
# STEP 6.13 — Calculate Training-Only Feature Statistics

train_values_flat = (
    X_train
    .reshape(
        -1,
        NUM_FEATURES
    )
)

feature_means = np.nanmean(
    train_values_flat,
    axis=0
)

feature_stds = np.nanstd(
    train_values_flat,
    axis=0
)

feature_stds[
    feature_stds == 0
] = 1.0

print(
    "\nSTEP 6.13 — Training Statistics"
)

print("-" * 50)

print("Feature means:")

for feature, mean in zip(
    feature_columns,
    feature_means
):

    print(
        f"{feature:20s}: {mean:.6f}"
    )

print(
    "\nFeature standard deviations:"
)

for feature, std in zip(
    feature_columns,
    feature_stds
):

    print(
        f"{feature:20s}: {std:.6f}"
    )

print(
    "\nNaN in feature means:",
    np.isnan(feature_means).sum()
)

print(
    "NaN in feature stds:",
    np.isnan(feature_stds).sum()
)


# STEP 6.14 — Fill Missing Input Values Using Training Means

def fill_missing_with_training_mean(
    data,
    means
):

    data = data.copy()

    for feature_idx in range(
        NUM_FEATURES
    ):

        missing_positions = np.isnan(
            data[
                :,
                :,
                :,
                feature_idx
            ]
        )

        data[
            :,
            :,
            :,
            feature_idx
        ][
            missing_positions
        ] = means[
            feature_idx
        ]

    return data


X_train_filled = (
    fill_missing_with_training_mean(
        X_train,
        feature_means
    )
)

X_val_filled = (
    fill_missing_with_training_mean(
        X_val,
        feature_means
    )
)

X_test_filled = (
    fill_missing_with_training_mean(
        X_test,
        feature_means
    )
)

print(
    "\nSTEP 6.14 — Missing Value Handling"
)

print("-" * 50)

print(
    "NaN in X_train:",
    np.isnan(X_train_filled).sum()
)

print(
    "NaN in X_val:",
    np.isnan(X_val_filled).sum()
)

print(
    "NaN in X_test:",
    np.isnan(X_test_filled).sum()
)


# STEP 6.15 — Feature-wise Normalization

def normalize_data(
    data,
    means,
    stds
):

    data = data.copy()

    for feature_idx in range(
        NUM_FEATURES
    ):

        data[
            :,
            :,
            :,
            feature_idx
        ] = (
            data[
                :,
                :,
                :,
                feature_idx
            ]
            - means[
                feature_idx
            ]
        ) / stds[
            feature_idx
        ]

    return data


X_train_normalized = (
    normalize_data(
        X_train_filled,
        feature_means,
        feature_stds
    )
)

X_val_normalized = (
    normalize_data(
        X_val_filled,
        feature_means,
        feature_stds
    )
)

X_test_normalized = (
    normalize_data(
        X_test_filled,
        feature_means,
        feature_stds
    )
)

print(
    "\nSTEP 6.15 — Feature Normalization"
)

print("-" * 50)

print(
    "NaN in normalized train:",
    np.isnan(
        X_train_normalized
    ).sum()
)

print(
    "NaN in normalized validation:",
    np.isnan(
        X_val_normalized
    ).sum()
)

print(
    "NaN in normalized test:",
    np.isnan(
        X_test_normalized
    ).sum()
)


# STEP 6.16 — Target Normalization with Safe NaN Handling

def normalize_target(
    target,
    means,
    stds
):

    target = target.copy()

    for feature_idx in range(
        NUM_FEATURES
    ):

        target[
            :,
            :,
            feature_idx
        ] = (
            target[
                :,
                :,
                feature_idx
            ]
            - means[
                feature_idx
            ]
        ) / stds[
            feature_idx
        ]

    target = np.nan_to_num(
        target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return target


Y_train_normalized = (
    normalize_target(
        Y_train,
        feature_means,
        feature_stds
    )
)

Y_val_normalized = (
    normalize_target(
        Y_val,
        feature_means,
        feature_stds
    )
)

Y_test_normalized = (
    normalize_target(
        Y_test,
        feature_means,
        feature_stds
    )
)

print(
    "\nSTEP 6.16 — Target Normalization"
)

print("-" * 50)

print(
    "Y_train shape:",
    Y_train_normalized.shape
)

print(
    "Y_val shape:",
    Y_val_normalized.shape
)

print(
    "Y_test shape:",
    Y_test_normalized.shape
)

print(
    "\nNaN in Y_train:",
    np.isnan(
        Y_train_normalized
    ).sum()
)

print(
    "NaN in Y_val:",
    np.isnan(
        Y_val_normalized
    ).sum()
)

print(
    "NaN in Y_test:",
    np.isnan(
        Y_test_normalized
    ).sum()
)
# STEP 6.17 — Convert Data to LSTM Input Format

X_train_lstm = X_train_normalized.reshape(
    len(X_train_normalized),
    WINDOW_SIZE,
    INPUT_SIZE
)

X_val_lstm = X_val_normalized.reshape(
    len(X_val_normalized),
    WINDOW_SIZE,
    INPUT_SIZE
)

X_test_lstm = X_test_normalized.reshape(
    len(X_test_normalized),
    WINDOW_SIZE,
    INPUT_SIZE
)

Y_train_lstm = Y_train_normalized.reshape(
    len(Y_train_normalized),
    INPUT_SIZE
)

Y_val_lstm = Y_val_normalized.reshape(
    len(Y_val_normalized),
    INPUT_SIZE
)

Y_test_lstm = Y_test_normalized.reshape(
    len(Y_test_normalized),
    INPUT_SIZE
)

T_train_lstm = T_train.reshape(
    len(T_train),
    INPUT_SIZE
)

T_val_lstm = T_val.reshape(
    len(T_val),
    INPUT_SIZE
)

T_test_lstm = T_test.reshape(
    len(T_test),
    INPUT_SIZE
)

print("\nSTEP 6.17 — LSTM Data Format")
print("-" * 50)

print("X_train:", X_train_lstm.shape)
print("Y_train:", Y_train_lstm.shape)
print("T_train:", T_train_lstm.shape)

print("\nX_val:", X_val_lstm.shape)
print("Y_val:", Y_val_lstm.shape)
print("T_val:", T_val_lstm.shape)

print("\nX_test:", X_test_lstm.shape)
print("Y_test:", Y_test_lstm.shape)
print("T_test:", T_test_lstm.shape)
# STEP 6.18 — Create PyTorch Dataset

class AirPollutionLSTMDataset(Dataset):

    def __init__(self, X, Y, target_mask):
        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.Y = torch.tensor(
            Y,
            dtype=torch.float32
        )

        self.target_mask = torch.tensor(
            target_mask,
            dtype=torch.bool
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.Y[index],
            self.target_mask[index]
        )


print("\nSTEP 6.18 — PyTorch Dataset")
print("-" * 50)

print("Dataset class created successfully.")
# STEP 6.19 — Create Dataset Objects

train_dataset = AirPollutionLSTMDataset(
    X_train_lstm,
    Y_train_lstm,
    T_train_lstm
)

val_dataset = AirPollutionLSTMDataset(
    X_val_lstm,
    Y_val_lstm,
    T_val_lstm
)

test_dataset = AirPollutionLSTMDataset(
    X_test_lstm,
    Y_test_lstm,
    T_test_lstm
)

print("\nSTEP 6.19 — Dataset Objects")
print("-" * 50)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

sample_X, sample_Y, sample_mask = train_dataset[0]

print("\nFirst training sample:")
print("X shape:", sample_X.shape)
print("Y shape:", sample_Y.shape)
print("Mask shape:", sample_mask.shape)

print("\nData types:")
print("X:", sample_X.dtype)
print("Y:", sample_Y.dtype)
print("Mask:", sample_mask.dtype)

# STEP 6.20 — Create PyTorch DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nSTEP 6.20 — DataLoaders")
print("-" * 50)

print("Batch size:", BATCH_SIZE)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

# STEP 6.21 — Define LSTM Model

class LSTMBaseline(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        output_size=INPUT_SIZE,
        dropout=0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_output = lstm_output[:, -1, :]

        output = self.fc(last_output)

        return output


print("\nSTEP 6.21 — LSTM Model")
print("-" * 50)

model = LSTMBaseline()

print(model)

# STEP 6.22 — Verify LSTM Forward Pass

sample_batch_X, sample_batch_Y, sample_batch_mask = next(
    iter(train_loader)
)

with torch.no_grad():
    sample_output = model(sample_batch_X)

print("\nSTEP 6.22 — Forward Pass Verification")
print("-" * 50)

print("Input batch shape:", sample_batch_X.shape)
print("Target batch shape:", sample_batch_Y.shape)
print("Mask batch shape:", sample_batch_mask.shape)
print("Model output shape:", sample_output.shape)

print(
    "\nOutput matches target:",
    sample_output.shape == sample_batch_Y.shape
)

# STEP 6.23 — Define Masked MSE Loss

def masked_mse_loss(prediction, target, target_mask):
    """
    Calculate MSE only on artificially masked target positions.

    target_mask:
        True  -> artificially masked, valid reconstruction target
        False -> not used for loss
    """

    mask = target_mask.float()

    squared_error = (prediction - target) ** 2

    masked_error = squared_error * mask

    valid_points = mask.sum()

    loss = masked_error.sum() / valid_points.clamp(min=1.0)

    return loss


print("\nSTEP 6.23 — Masked MSE Loss")
print("-" * 50)

print("Masked MSE loss function created successfully.")

# STEP 6.24 — Verify Masked MSE Loss

sample_batch_X, sample_batch_Y, sample_batch_mask = next(
    iter(train_loader)
)

with torch.no_grad():
    sample_prediction = model(sample_batch_X)

sample_loss = masked_mse_loss(
    sample_prediction,
    sample_batch_Y,
    sample_batch_mask
)

print("\nSTEP 6.24 — Masked Loss Verification")
print("-" * 50)

print("Prediction shape:", sample_prediction.shape)
print("Target shape:", sample_batch_Y.shape)
print("Mask shape:", sample_batch_mask.shape)

print(
    "Valid masked points in batch:",
    sample_batch_mask.sum().item()
)

print(
    "Masked MSE loss:",
    sample_loss.item()
)

print(
    "Loss is finite:",
    torch.isfinite(sample_loss).item()
)

# STEP 6.25 — Verify Masked Loss Calculation

mask_count = sample_batch_mask.sum().item()

total_elements = sample_batch_mask.numel()

masked_percentage = (
    mask_count / total_elements
) * 100

print("\nSTEP 6.25 — Mask Verification")
print("-" * 50)

print("Total target values in batch:", total_elements)
print("Artificially masked targets:", mask_count)
print(
    "Masked target percentage:",
    f"{masked_percentage:.2f}%"
)

print(
    "\nMask contains only True/False:",
    sample_batch_mask.dtype == torch.bool
)

print(
    "At least one valid target:",
    mask_count > 0
)

# STEP 6.26 — Define Optimizer

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nSTEP 6.26 — Optimizer")
print("-" * 50)

print("Optimizer: Adam")
print("Learning rate:", LEARNING_RATE)

print(
    "Trainable parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
)

# STEP 6.27 — Define One-Epoch Training Function

def train_one_epoch(model, loader, optimizer):
    """
    Train the model for one complete epoch.
    Loss is calculated only on artificially masked targets.
    """

    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_X, batch_Y, batch_mask in loader:

        optimizer.zero_grad()

        prediction = model(batch_X)

        loss = masked_mse_loss(
            prediction,
            batch_Y,
            batch_mask
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches

    return average_loss


print("\nSTEP 6.27 — Training Function")
print("-" * 50)

print("One-epoch training function created successfully.")

# STEP 6.28 — Define Validation Function

def validate_one_epoch(model, loader):
    """
    Evaluate the model on validation data.
    No gradients are calculated.
    Loss is calculated only on artificially masked targets.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for batch_X, batch_Y, batch_mask in loader:

            prediction = model(batch_X)

            loss = masked_mse_loss(
                prediction,
                batch_Y,
                batch_mask
            )

            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / total_batches

    return average_loss


print("\nSTEP 6.28 — Validation Function")
print("-" * 50)

print("Validation function created successfully.")

# STEP 6.29 — Define Training Loop

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    model_file
):
    """
    Train the LSTM model and save the best validation model.
    """

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_loss = validate_one_epoch(
            model,
            val_loader
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                model_file
            )

            print(
                f"  Best model saved → {model_file}"
            )

    return train_losses, val_losses


print("\nSTEP 6.29 — Training Loop")
print("-" * 50)

print("Training loop function created successfully.")
print("Configured epochs:", EPOCHS)
print("Model save path:", MODEL_FILE)
# STEP 6.30 — Final Training Safety Check

model_parameters_finite = all(
    torch.isfinite(parameter).all().item()
    for parameter in model.parameters()
)

optimizer_parameters = sum(
    len(group["params"])
    for group in optimizer.param_groups
)

train_loader_valid = len(train_loader) > 0
val_loader_valid = len(val_loader) > 0

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSTEP 6.30 — Final Training Safety Check")
print("-" * 50)

print(
    "Model parameters finite:",
    model_parameters_finite
)

print(
    "Optimizer parameter groups:",
    optimizer_parameters
)

print(
    "Training DataLoader valid:",
    train_loader_valid
)

print(
    "Validation DataLoader valid:",
    val_loader_valid
)

print(
    "Model directory exists:",
    MODEL_FILE.parent.exists()
)

print("\nReady for training:",
      model_parameters_finite
      and optimizer_parameters > 0
      and train_loader_valid
      and val_loader_valid)

# ============================================================
# STEP 6.31 — Start LSTM Training
# ============================================================

print("\nSTEP 6.31 — LSTM Training Started")
print("=" * 60)

train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    epochs=EPOCHS,
    model_file=MODEL_FILE
)

print("\n" + "=" * 60)
print("LSTM training completed.")
print("Best model saved at:")
print(MODEL_FILE)


# ============================================================
# STEP 6.32 — Load Best Model
# ============================================================

print("\nSTEP 6.32 — Loading Best Model")
print("-" * 60)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location="cpu"
    )
)

model.eval()

print("Best model loaded successfully.")


# ============================================================
# STEP 6.33 — Display Training Summary
# ============================================================

print("\nSTEP 6.33 — Training Summary")
print("-" * 60)

best_epoch = np.argmin(val_losses) + 1
best_val_loss = min(val_losses)
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]

print("Total epochs:", EPOCHS)
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Final training loss:", final_train_loss)
print("Final validation loss:", final_val_loss)


# ============================================================
# STEP 6.34 — Verify Saved Model
# ============================================================

print("\nSTEP 6.34 — Verify Saved Model")
print("-" * 60)

print("Model file exists:", MODEL_FILE.exists())

if MODEL_FILE.exists():
    print(
        "Model file size:",
        f"{MODEL_FILE.stat().st_size / (1024 * 1024):.2f} MB"
    )

    print("Saved model path:")
    print(MODEL_FILE)


# ============================================================
# STEP 6.35 — Final STEP 6 Completion Check
# ============================================================

print("\nSTEP 6.35 — LSTM Baseline Completed")
print("=" * 60)

print("Training completed successfully.")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Model saved:", MODEL_FILE.exists())

print("\nSTEP 6 completed successfully.")