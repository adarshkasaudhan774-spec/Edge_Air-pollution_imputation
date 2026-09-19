# STEP 6.1 — Import Required Libraries

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("Libraries imported successfully!")
print("PyTorch version:", torch.__version__)


# STEP 6.2 — Define Project Paths
from pathlib import Path

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data folders
DATA_RAW = PROJECT_ROOT / "Data" / "raw data"
DATA_PROCESSED = PROJECT_ROOT / "Data" / "processed"

# Model and result folders
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"

print("Project root:", PROJECT_ROOT)
print("Raw data folder:", DATA_RAW)
print("Processed data folder:", DATA_PROCESSED)
print("Models folder:", MODELS_DIR)
print("Results folder:", RESULTS_DIR)

# STEP 6.3 — Load Fixed Artificial Mask
MASK_FILE = DATA_PROCESSED / "fixed_artificial_mask.csv"

artificial_mask = pd.read_csv(MASK_FILE)

print("Fixed artificial mask loaded successfully!")
print("Mask shape:", artificial_mask.shape)
print("Number of stations:", artificial_mask.shape[1])
print("Mask values:")
print(artificial_mask.stack().value_counts())

# STEP 6.4 — Load Original Dataset

DATA_FILE = DATA_RAW / "air_weather_hourly_2019_2021 (2).csv"
df = pd.read_csv(DATA_FILE)

# Convert timestamp
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

# Sort by timestamp
df = df.sort_values("TIMESTAMP").reset_index(drop=True)

print("Original dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Unique timestamps:", df["TIMESTAMP"].nunique())
print("Columns:", len(df.columns))

# STEP 6.5 — Define Station and Pollutant Columns
station_columns = [
    "SHATIN",
    "TSUEN WAN",
    "CENTRAL",
    "EASTERN",
    "KWUN TONG",
    "TUEN MUN",
    "TUNG CHUNG",
    "SHAM SHUI PO",
    "SOUTHERN",
    "YUEN LONG",
    "CENTRAL/WESTERN",
    "KWAI CHUNG",
    "TSEUNG KWAN O",
    "TAI PO",
    "MONG KOK",
    "CAUSEWAY BAY"
]

pollutant_columns = [
    "Sulphur Dioxide",
    "Respirable Suspended Particulates",
    "Fine Suspended Particulates ",
    "Nitrogen Oxides",
    "Nitrogen Dioxide",
    "Carbon Monoxide",
    "Ozone"
]

print("Number of stations:", len(station_columns))
print("Number of pollutants:", len(pollutant_columns))

print("\nStations:")
print(station_columns)

print("\nPollutants:")
print(pollutant_columns)

# STEP 6.6 — Create Masked Dataset
df_masked = df.copy()

# Original observed positions
original_observed = df[station_columns].notna()

# Artificially masked positions
artificial_mask_positions = (
    (artificial_mask == 0) &
    (original_observed == True)
)

# Apply artificial missing values
for column in station_columns:
    df_masked.loc[
        artificial_mask_positions[column],
        column
    ] = np.nan

print("Masked dataset created successfully!")
print("Original dataset shape:", df.shape)
print("Masked dataset shape:", df_masked.shape)

print(
    "Artificially masked values:",
    artificial_mask_positions.sum().sum()
)

# STEP 6.7 — Define the 13 Features
weather_features = [
    "Temperature_mean",
    "Humidity_mean",
    "Sea_Level_Pressure_mean",
    "Wind_Speed_mean",
    "Available_Stations",
    "Wind_Direction_mean"
]

feature_columns = pollutant_columns + weather_features

print("Number of pollutant features:", len(pollutant_columns))
print("Number of weather features:", len(weather_features))
print("Total features:", len(feature_columns))

print("\nAll 13 features:")
for i, feature in enumerate(feature_columns, start=1):
    print(i, "→", feature)
    
# STEP 6.8 — Efficiently Create Data Tensor
# Sort dataset
df_masked = df_masked.sort_values(
    ["TIMESTAMP", "POLLUTANT"]
).reset_index(drop=True)

# Unique timestamps
timestamps = sorted(df_masked["TIMESTAMP"].unique())

num_timestamps = len(timestamps)
num_stations = len(station_columns)
num_features = len(feature_columns)

# Create empty tensor
data_tensor = np.full(
    (num_timestamps, num_stations, num_features),
    np.nan,
    dtype=np.float32
)

# Create timestamp index
timestamp_index = pd.Index(timestamps)

# Fill pollutant features
for feature_index, pollutant in enumerate(pollutant_columns):

    pollutant_data = df_masked[
        df_masked["POLLUTANT"] == pollutant
    ].copy()

    pollutant_data = pollutant_data.set_index("TIMESTAMP")

    pollutant_matrix = pollutant_data[
        station_columns
    ].reindex(timestamp_index)

    data_tensor[:, :, feature_index] = (
        pollutant_matrix.to_numpy(dtype=np.float32)
    )

# Fill weather features
for feature_index, weather in enumerate(weather_features):

    weather_data = (
        df_masked
        .drop_duplicates("TIMESTAMP")
        .set_index("TIMESTAMP")[weather]
        .reindex(timestamp_index)
    )

    weather_values = weather_data.to_numpy(
        dtype=np.float32
    )

    # Same weather value for all 16 stations
    data_tensor[
        :, :, feature_index + len(pollutant_columns)
    ] = weather_values[:, np.newaxis]

print("Data tensor created successfully!")
print("Tensor shape:", data_tensor.shape)
print("Tensor dtype:", data_tensor.dtype)

# STEP 6.9 — Verify Tensor Missing Values
total_values = data_tensor.size
missing_values = np.isnan(data_tensor).sum()
available_values = total_values - missing_values

print("Tensor verification")
print("-------------------")
print("Total tensor values:", total_values)
print("Available values:", available_values)
print("Missing values:", missing_values)
print(
    "Missing percentage:",
    round((missing_values / total_values) * 100, 2),
    "%"
)

# STEP 6.10 — Create 24-Hour Sliding Windows
WINDOW_SIZE = 24

num_windows = num_timestamps - WINDOW_SIZE + 1

sliding_windows = np.empty(
    (
        num_windows,
        WINDOW_SIZE,
        num_stations,
        num_features
    ),
    dtype=np.float32
)

for i in range(num_windows):
    sliding_windows[i] = data_tensor[
        i:i + WINDOW_SIZE
    ]

print("24-hour sliding windows created successfully!")
print("Window size:", WINDOW_SIZE)
print("Number of windows:", num_windows)
print("Sliding window shape:", sliding_windows.shape)

# STEP 6.11 — Create Artificial Mask Tensor
# Convert station mask from
# (184128, 16)
# to
# (26304, 16, 13)

mask_tensor = np.ones(
    (num_timestamps, num_stations, num_features),
    dtype=np.float32
)

# Original/artificial mask is available only
# for station values in the long-format dataset.
for feature_index, pollutant in enumerate(pollutant_columns):

    pollutant_indices = (
        df["POLLUTANT"] == pollutant
    )

    mask_values = artificial_mask.loc[
        pollutant_indices
    ].to_numpy(dtype=np.float32)

    mask_tensor[
        :,
        :,
        feature_index
    ] = mask_values

print("Artificial mask tensor created successfully!")
print("Mask tensor shape:", mask_tensor.shape)

# STEP 6.12 — Create Mask Sliding Windows
mask_sliding_windows = np.empty(
    (
        num_windows,
        WINDOW_SIZE,
        num_stations,
        num_features
    ),
    dtype=np.float32
)

for i in range(num_windows):
    mask_sliding_windows[i] = mask_tensor[
        i:i + WINDOW_SIZE
    ]

print("Mask sliding windows created successfully!")
print("Mask window shape:", mask_sliding_windows.shape)

# STEP 6.13 — Train / Validation / Test Split
total_windows = len(sliding_windows)

train_end = int(total_windows * 0.70)
val_end = int(total_windows * 0.85)

# Chronological split — NO random shuffle
X_train = sliding_windows[:train_end]
X_val = sliding_windows[train_end:val_end]
X_test = sliding_windows[val_end:]

M_train = mask_sliding_windows[:train_end]
M_val = mask_sliding_windows[train_end:val_end]
M_test = mask_sliding_windows[val_end:]

print("Chronological split completed!")
print("--------------------------------")

print("Training windows:", X_train.shape)
print("Validation windows:", X_val.shape)
print("Testing windows:", X_test.shape)

print("\nMask shapes:")
print("Training mask:", M_train.shape)
print("Validation mask:", M_val.shape)
print("Testing mask:", M_test.shape)

# STEP 6.14 — Prepare LSTM Input and Target
# Input: complete 24-hour sequence
X_train_lstm = X_train

X_val_lstm = X_val

X_test_lstm = X_test

# Target: 24th hour of each window
Y_train_lstm = X_train[:, -1, :, :]

Y_val_lstm = X_val[:, -1, :, :]

Y_test_lstm = X_test[:, -1, :, :]

# Target masks
M_train_target = M_train[:, -1, :, :]

M_val_target = M_val[:, -1, :, :]

M_test_target = M_test[:, -1, :, :]

print("LSTM input-target preparation completed!")
print("---------------------------------------")

print("X_train:", X_train_lstm.shape)
print("Y_train:", Y_train_lstm.shape)

print("X_val:", X_val_lstm.shape)
print("Y_val:", Y_val_lstm.shape)

print("X_test:", X_test_lstm.shape)
print("Y_test:", Y_test_lstm.shape)

print("\nTarget mask shapes:")
print("M_train:", M_train_target.shape)
print("M_val:", M_val_target.shape)
print("M_test:", M_test_target.shape)

# STEP 6.15 — Define LSTM Model Architecture
class LSTMBaseline(nn.Module):

    def __init__(
        self,
        input_size=208,
        hidden_size=128,
        num_layers=2,
        output_size=208,
        dropout=0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # x shape:
        # (batch, 24, 16, 13)

        batch_size = x.size(0)

        # Convert:
        # (batch, 24, 16, 13)
        # →
        # (batch, 24, 208)

        x = x.reshape(
            batch_size,
            x.size(1),
            16 * 13
        )

        # LSTM
        lstm_output, _ = self.lstm(x)

        # Take last time step
        last_output = lstm_output[:, -1, :]

        # Predict 208 values
        output = self.output_layer(last_output)

        # Convert:
        # (batch, 208)
        # →
        # (batch, 16, 13)

        output = output.reshape(
            batch_size,
            16,
            13
        )

        return output


# Create model
lstm_model = LSTMBaseline()

print("LSTM model created successfully!")
print(lstm_model)

# STEP 6.16 — Handle Missing Values in LSTM Input
# Calculate feature-wise mean using ONLY training data
# Shape before calculation:
# (samples, 24, 16, 13)

train_feature_means = np.nanmean(
    X_train_lstm,
    axis=(0, 1, 2)
)

print("Training feature means calculated!")
print("Shape:", train_feature_means.shape)

# Replace NaN values using training feature means
X_train_lstm_clean = np.where(
    np.isnan(X_train_lstm),
    train_feature_means.reshape(1, 1, 1, -1),
    X_train_lstm
)

X_val_lstm_clean = np.where(
    np.isnan(X_val_lstm),
    train_feature_means.reshape(1, 1, 1, -1),
    X_val_lstm
)

X_test_lstm_clean = np.where(
    np.isnan(X_test_lstm),
    train_feature_means.reshape(1, 1, 1, -1),
    X_test_lstm
)

print("\nMissing-value handling completed!")
print("--------------------------------")

print(
    "Remaining NaN in X_train:",
    np.isnan(X_train_lstm_clean).sum()
)

print(
    "Remaining NaN in X_val:",
    np.isnan(X_val_lstm_clean).sum()
)

print(
    "Remaining NaN in X_test:",
    np.isnan(X_test_lstm_clean).sum()
)
# STEP 6.17 — Create PyTorch Dataset
class LSTMDataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.Y = torch.tensor(
            Y,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# Create datasets
train_dataset = LSTMDataset(
    X_train_lstm_clean,
    Y_train_lstm
)

val_dataset = LSTMDataset(
    X_val_lstm_clean,
    Y_val_lstm
)

test_dataset = LSTMDataset(
    X_test_lstm_clean,
    Y_test_lstm
)


print("PyTorch datasets created successfully!")
print("--------------------------------------")

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))

# Check one sample
sample_X, sample_Y = train_dataset[0]

print("\nSingle sample shapes:")
print("Input:", sample_X.shape)
print("Target:", sample_Y.shape)

# STEP 6.18 — Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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


print("DataLoaders created successfully!")
print("---------------------------------")

print("Batch size:", BATCH_SIZE)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))


# Check one batch
sample_batch_X, sample_batch_Y = next(iter(train_loader))

print("\nFirst batch shapes:")
print("Input batch:", sample_batch_X.shape)
print("Target batch:", sample_batch_Y.shape)

# STEP 6.19 — Define Masked MSE Loss
def masked_mse_loss(prediction, target, mask):
    """
    Calculate MSE only on valid target positions.
    
    prediction : (batch, 16, 13)
    target     : (batch, 16, 13)
    mask       : (batch, 16, 13)
    """

    # Calculate squared error
    squared_error = (prediction - target) ** 2

    # Keep only valid positions
    masked_error = squared_error * mask

    # Avoid division by zero
    valid_count = mask.sum().clamp(min=1.0)

    # Mean squared error
    loss = masked_error.sum() / valid_count

    return loss


print("Masked MSE loss function created successfully!")

# STEP 6.20 — Configure Optimizer
LEARNING_RATE = 0.001

optimizer = torch.optim.Adam(
    lstm_model.parameters(),
    lr=LEARNING_RATE
)

print("Optimizer configured successfully!")
print("---------------------------------")
print("Optimizer: Adam")
print("Learning rate:", LEARNING_RATE)

# STEP 6.21 — Prepare True Ground-Truth Targets
# Create original data tensor from the ORIGINAL dataset
# (before artificial masking)

original_data_tensor = np.full(
    (num_timestamps, num_stations, num_features),
    np.nan,
    dtype=np.float32
)

# Timestamp index
timestamp_index = pd.Index(timestamps)

# Fill pollutant features from original dataset
for feature_index, pollutant in enumerate(pollutant_columns):

    pollutant_data = df[
        df["POLLUTANT"] == pollutant
    ].copy()

    pollutant_data = pollutant_data.set_index("TIMESTAMP")

    pollutant_matrix = pollutant_data[
        station_columns
    ].reindex(timestamp_index)

    original_data_tensor[
        :, :, feature_index
    ] = pollutant_matrix.to_numpy(
        dtype=np.float32
    )


# Fill weather features
for feature_index, weather in enumerate(weather_features):

    weather_data = (
        df
        .drop_duplicates("TIMESTAMP")
        .set_index("TIMESTAMP")[weather]
        .reindex(timestamp_index)
    )

    weather_values = weather_data.to_numpy(
        dtype=np.float32
    )

    original_data_tensor[
        :, :,
        feature_index + len(pollutant_columns)
    ] = weather_values[:, np.newaxis]


print("Original ground-truth tensor created!")
print("------------------------------------")
print("Shape:", original_data_tensor.shape)


# Target of each 24-hour window = last hour
target_start = WINDOW_SIZE - 1

Y_train_ground_truth = original_data_tensor[
    target_start:target_start + train_end
]

Y_val_ground_truth = original_data_tensor[
    target_start + train_end:
    target_start + val_end
]

Y_test_ground_truth = original_data_tensor[
    target_start + val_end:
    target_start + total_windows
]


print("\nGround-truth targets prepared!")
print("--------------------------------")

print("Y_train:", Y_train_ground_truth.shape)
print("Y_val:", Y_val_ground_truth.shape)
print("Y_test:", Y_test_ground_truth.shape)

# ============================================
# STEP 6.22 — Create Artificial-Only Target Masks
# ============================================

# Create mask containing ONLY artificially masked positions
# 1 = artificially masked position
# 0 = everything else

artificial_only_mask_tensor = np.zeros(
    (num_timestamps, num_stations, num_features),
    dtype=np.float32
)

# Fill only pollutant features
for feature_index, pollutant in enumerate(pollutant_columns):

    pollutant_indices = (
        df["POLLUTANT"] == pollutant
    )

    artificial_positions = artificial_mask_positions.loc[
        pollutant_indices
    ].to_numpy(dtype=np.float32)

    artificial_only_mask_tensor[
        :,
        :,
        feature_index
    ] = artificial_positions


# Last hour of every 24-hour window
target_artificial_mask = artificial_only_mask_tensor[
    WINDOW_SIZE - 1:
]


# Chronological split
M_train_target_artificial = target_artificial_mask[
    :train_end
]

M_val_target_artificial = target_artificial_mask[
    train_end:val_end
]

M_test_target_artificial = target_artificial_mask[
    val_end:total_windows
]


print("Artificial-only target masks prepared!")
print("---------------------------------------")

print(
    "M_train:",
    M_train_target_artificial.shape
)

print(
    "M_val:",
    M_val_target_artificial.shape
)

print(
    "M_test:",
    M_test_target_artificial.shape
)


print("\nArtificially masked target positions:")

print(
    "Train:",
    int(M_train_target_artificial.sum())
)

print(
    "Validation:",
    int(M_val_target_artificial.sum())
)

print(
    "Test:",
    int(M_test_target_artificial.sum())
)
# ============================================
# STEP 6.23 — Prepare PyTorch Targets and Masks
# ============================================

# Convert ground-truth targets to PyTorch tensors
Y_train_tensor = torch.tensor(
    Y_train_ground_truth,
    dtype=torch.float32
)

Y_val_tensor = torch.tensor(
    Y_val_ground_truth,
    dtype=torch.float32
)

Y_test_tensor = torch.tensor(
    Y_test_ground_truth,
    dtype=torch.float32
)


# Convert artificial masks to PyTorch tensors
M_train_tensor = torch.tensor(
    M_train_target_artificial,
    dtype=torch.float32
)

M_val_tensor = torch.tensor(
    M_val_target_artificial,
    dtype=torch.float32
)

M_test_tensor = torch.tensor(
    M_test_target_artificial,
    dtype=torch.float32
)


print("PyTorch targets and masks prepared!")
print("-----------------------------------")

print("Y_train:", Y_train_tensor.shape)
print("Y_val:", Y_val_tensor.shape)
print("Y_test:", Y_test_tensor.shape)

print("\nMasks:")
print("M_train:", M_train_tensor.shape)
print("M_val:", M_val_tensor.shape)
print("M_test:", M_test_tensor.shape)
# ============================================
# STEP 6.24 — Create Imputation Dataset
# ============================================

class LSTMImputationDataset(Dataset):

    def __init__(self, X, Y, mask):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.Y = Y

        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.Y[index],
            self.mask[index]
        )


# Create datasets
train_imputation_dataset = LSTMImputationDataset(
    X_train_lstm_clean,
    Y_train_tensor,
    M_train_tensor
)

val_imputation_dataset = LSTMImputationDataset(
    X_val_lstm_clean,
    Y_val_tensor,
    M_val_tensor
)

test_imputation_dataset = LSTMImputationDataset(
    X_test_lstm_clean,
    Y_test_tensor,
    M_test_tensor
)


print("LSTM imputation datasets created successfully!")
print("----------------------------------------------")

print(
    "Training samples:",
    len(train_imputation_dataset)
)

print(
    "Validation samples:",
    len(val_imputation_dataset)
)

print(
    "Testing samples:",
    len(test_imputation_dataset)
)


# Check one sample
sample_X, sample_Y, sample_M = (
    train_imputation_dataset[0]
)

print("\nSingle sample shapes:")
print("Input :", sample_X.shape)
print("Target:", sample_Y.shape)
print("Mask  :", sample_M.shape)
# ============================================
# STEP 6.25 — Create Imputation DataLoaders
# ============================================

BATCH_SIZE = 32

train_imputation_loader = DataLoader(
    train_imputation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_imputation_loader = DataLoader(
    val_imputation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_imputation_loader = DataLoader(
    test_imputation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Imputation DataLoaders created successfully!")
print("--------------------------------------------")

print(
    "Training batches:",
    len(train_imputation_loader)
)

print(
    "Validation batches:",
    len(val_imputation_loader)
)

print(
    "Testing batches:",
    len(test_imputation_loader)
)


# Check one batch
batch_X, batch_Y, batch_M = next(
    iter(train_imputation_loader)
)

print("\nFirst batch shapes:")
print("Input :", batch_X.shape)
print("Target:", batch_Y.shape)
print("Mask  :", batch_M.shape)
# ============================================
# STEP 6.26 — Setup Training Device
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move LSTM model to selected device
lstm_model = lstm_model.to(device)

print("Training device configured successfully!")
print("------------------------------------------")
print("Device:", device)
# ============================================
# STEP 6.27 — Set Training Parameters
# ============================================

NUM_EPOCHS = 20

BEST_VAL_LOSS = float("inf")

MODEL_SAVE_PATH = MODELS_DIR / "lstm_baseline.pth"

print("Training parameters configured!")
print("--------------------------------")
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Model save path:", MODEL_SAVE_PATH)
# ============================================
# STEP 6.28 — Test LSTM Forward Pass
# ============================================

# Get one batch
batch_X, batch_Y, batch_M = next(
    iter(train_imputation_loader)
)

# Move input to CPU/GPU device
batch_X = batch_X.to(device)

# Forward pass
with torch.no_grad():
    batch_prediction = lstm_model(batch_X)


print("LSTM forward pass successful!")
print("--------------------------------")

print("Input shape:")
print(batch_X.shape)

print("\nPrediction shape:")
print(batch_prediction.shape)

print("\nExpected prediction shape:")
print(batch_Y.shape)
# ============================================
# STEP 6.29 — Fix Masked MSE Loss
# ============================================

def masked_mse_loss(prediction, target, mask):
    """
    Calculate MSE only on artificially masked positions.

    prediction : (batch, 16, 13)
    target     : (batch, 16, 13)
    mask       : (batch, 16, 13)

    mask = 1 → artificially masked → calculate loss
    mask = 0 → ignore
    """

    # Select only artificially masked positions
    valid_positions = mask == 1

    # Extract prediction and target only at valid positions
    selected_prediction = prediction[valid_positions]
    selected_target = target[valid_positions]

    # Calculate MSE only on selected values
    loss = torch.mean(
        (selected_prediction - selected_target) ** 2
    )

    return loss


print("Masked MSE loss function fixed successfully!")
# ============================================
# STEP 6.29 — Test Fixed Masked Loss
# ============================================

batch_X, batch_Y, batch_M = next(
    iter(train_imputation_loader)
)

batch_X = batch_X.to(device)
batch_Y = batch_Y.to(device)
batch_M = batch_M.to(device)

prediction = lstm_model(batch_X)

loss = masked_mse_loss(
    prediction,
    batch_Y,
    batch_M
)

print("Masked loss test successful!")
print("-----------------------------")
print("Loss:", loss.item())
# ============================================
# STEP 6.30 — Train LSTM Model
# ============================================

print("Starting LSTM training...")
print("==========================")

training_losses = []

for epoch in range(NUM_EPOCHS):

    # Training mode
    lstm_model.train()

    epoch_loss = 0.0
    batch_count = 0

    for batch_X, batch_Y, batch_M in train_imputation_loader:

        # Move data to device
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        batch_M = batch_M.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        prediction = lstm_model(batch_X)

        # Calculate masked loss
        loss = masked_mse_loss(
            prediction,
            batch_Y,
            batch_M
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()
        batch_count += 1

    # Average training loss
    average_loss = epoch_loss / batch_count

    training_losses.append(average_loss)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"- Training Loss: {average_loss:.4f}"
    )

print("\nLSTM training completed successfully!")
# ============================================
# STEP 6.30 — Save Trained LSTM
# ============================================

torch.save(
    lstm_model.state_dict(),
    MODEL_SAVE_PATH
)

print("Trained LSTM model saved successfully!")
print("Saved at:", MODEL_SAVE_PATH)
# ============================================
# STEP 6.31 — Validate LSTM Model
# ============================================

lstm_model.eval()

validation_loss = 0.0
validation_batches = 0

with torch.no_grad():

    for batch_X, batch_Y, batch_M in val_imputation_loader:

        # Move data to device
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        batch_M = batch_M.to(device)

        # Prediction
        prediction = lstm_model(batch_X)

        # Calculate masked validation loss
        loss = masked_mse_loss(
            prediction,
            batch_Y,
            batch_M
        )

        validation_loss += loss.item()
        validation_batches += 1


average_validation_loss = (
    validation_loss / validation_batches
)


print("Validation completed successfully!")
print("----------------------------------")
print(
    "Validation Loss:",
    round(average_validation_loss, 4)
)
# ============================================
# STEP 6.32 — Save Trained LSTM Model
# ============================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    lstm_model.state_dict(),
    MODEL_SAVE_PATH
)

print("\nLSTM model saved successfully!")
print("-------------------------------")
print("Model path:")
print(MODEL_SAVE_PATH)
# ============================================
# STEP 6.33 — Generate Test Predictions
# ============================================

lstm_model.eval()

test_predictions = []

with torch.no_grad():

    for batch_X, batch_Y, batch_M in test_imputation_loader:

        batch_X = batch_X.to(device)

        prediction = lstm_model(batch_X)

        test_predictions.append(
            prediction.cpu().numpy()
        )


test_predictions = np.concatenate(
    test_predictions,
    axis=0
)

print("\nTest predictions generated successfully!")
print("------------------------------------------")
print("Prediction shape:", test_predictions.shape)
print("Expected shape:", Y_test_ground_truth.shape)
# ============================================
# STEP 6.34 — Calculate MAE and RMSE
# ============================================

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


evaluation_results = []

for feature_index, pollutant in enumerate(
    pollutant_columns
):

    # Ground truth for this pollutant
    true_values = Y_test_ground_truth[
        :,
        :,
        feature_index
    ]

    # Predictions
    predicted_values = test_predictions[
        :,
        :,
        feature_index
    ]

    # Artificially masked positions only
    evaluation_mask = M_test_target_artificial[
        :,
        :,
        feature_index
    ] == 1

    true_values = true_values[
        evaluation_mask
    ]

    predicted_values = predicted_values[
        evaluation_mask
    ]

    # Remove any unexpected NaN values
    valid = (
        np.isfinite(true_values) &
        np.isfinite(predicted_values)
    )

    true_values = true_values[valid]
    predicted_values = predicted_values[valid]

    if len(true_values) > 0:

        mae = mean_absolute_error(
            true_values,
            predicted_values
        )

        rmse = np.sqrt(
            mean_squared_error(
                true_values,
                predicted_values
            )
        )

        evaluation_results.append({
            "Pollutant": pollutant.strip(),
            "Evaluation_Points": len(true_values),
            "MAE": mae,
            "RMSE": rmse
        })


lstm_results_df = pd.DataFrame(
    evaluation_results
)

print("\nLSTM evaluation completed!")
print("--------------------------")
print(lstm_results_df)
# ============================================
# STEP 6.35 — Save LSTM Results
# ============================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LSTM_RESULTS_FILE = (
    RESULTS_DIR / "lstm_baseline_results.csv"
)

lstm_results_df.to_csv(
    LSTM_RESULTS_FILE,
    index=False
)

print("\nLSTM results saved successfully!")
print("---------------------------------")
print("Results file:")
print(LSTM_RESULTS_FILE)

print("\n================================")
print("STEP 6 — LSTM BASELINE COMPLETE")
print("================================")
