from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# STEP 7.1 — Imports & Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "Data" / "raw data"
PROCESSED_DATA = PROJECT_ROOT / "Data" / "processed"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"

AIR_FILE = RAW_DATA / "air_quality_2019_2021_hourly.csv"
MET_FILE = RAW_DATA / "hourly_meteorology_16stations_2019_2021.csv"
MASK_FILE = PROCESSED_DATA / "fixed_artificial_mask_new.csv"
MODEL_FILE = MODELS_DIR / "lstm_baseline_new.pth"

RESULT_FILE = RESULTS_DIR / "lstm_evaluation_results.csv"

NUM_STATIONS = 16
NUM_FEATURES = 13
INPUT_SIZE = NUM_STATIONS * NUM_FEATURES

WINDOW_SIZE = 24

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("STEP 7.1 — Project Configuration")
print("=" * 60)

print("Project root:", PROJECT_ROOT)
print("Air quality file:", AIR_FILE)
print("Meteorology file:", MET_FILE)
print("Mask file:", MASK_FILE)
print("Model file:", MODEL_FILE)
print("Results file:", RESULT_FILE)
print("Device:", DEVICE)


# ============================================================
# STEP 7.2 — Define Features
# ============================================================

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

feature_columns = (
    pollution_features
    + weather_features
)

print("\nSTEP 7.2 — Feature Configuration")
print("-" * 60)

print("Number of pollution features:", len(pollution_features))
print("Number of weather features:", len(weather_features))
print("Total features:", len(feature_columns))
print("Input size:", INPUT_SIZE)
print("Features:", feature_columns)


# ============================================================
# STEP 7.3 — Load Dataset
# ============================================================

air_df = pd.read_csv(AIR_FILE)
met_df = pd.read_csv(MET_FILE)
fixed_mask = pd.read_csv(MASK_FILE)

air_df["timestamp"] = pd.to_datetime(
    air_df["timestamp"]
)

met_df["timestamp"] = pd.to_datetime(
    met_df["timestamp"]
)

print("\nSTEP 7.3 — Dataset Loading")
print("-" * 60)

print("Air quality shape:", air_df.shape)
print("Meteorology shape:", met_df.shape)
print("Mask shape:", fixed_mask.shape)


# ============================================================
# STEP 7.4 — Merge Air Quality and Meteorology
# ============================================================

merged_df = pd.merge(
    air_df,
    met_df,
    on=[
        "station_id",
        "station_name",
        "timestamp"
    ],
    how="inner"
)

print("\nSTEP 7.4 — Dataset Merge")
print("-" * 60)

print("Merged shape:", merged_df.shape)

print(
    "Unique stations:",
    merged_df["station_id"].nunique()
)

print(
    "Unique timestamps:",
    merged_df["timestamp"].nunique()
)

print(
    "Duplicate station-timestamp rows:",
    merged_df.duplicated(
        subset=[
            "station_id",
            "timestamp"
        ]
    ).sum()
)


# ============================================================
# STEP 7.5 — Verify Dataset and Mask
# ============================================================

if len(merged_df) != len(fixed_mask):
    raise ValueError(
        "Merged dataset and fixed mask row counts do not match."
    )

if list(fixed_mask.columns) != feature_columns:
    raise ValueError(
        "Fixed mask feature order does not match feature_columns."
    )

invalid_mask_values = fixed_mask[
    ~fixed_mask.isin([0, 1])
].count().sum()

if invalid_mask_values != 0:
    raise ValueError(
        "Fixed mask contains values other than 0 and 1."
    )

print("\nSTEP 7.5 — Dataset and Mask Verification")
print("-" * 60)

print(
    "Dataset and mask row counts match:",
    len(merged_df) == len(fixed_mask)
)

print(
    "Feature order matches:",
    list(fixed_mask.columns) == feature_columns
)

print(
    "Invalid mask values:",
    invalid_mask_values
)


# ============================================================
# STEP 7.6 — Sort Dataset and Mask Exactly Like Step 6
# ============================================================

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

print("\nSTEP 7.6 — Dataset Ordering")
print("-" * 60)

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


# ============================================================
# STEP 7.7 — Prepare Data and Mask Arrays
# ============================================================

data_array = (
    merged_df[feature_columns]
    .to_numpy(dtype=np.float32)
)

mask_array = (
    fixed_mask[feature_columns]
    .to_numpy(dtype=np.int8)
)

original_observation_mask = (
    merged_df[feature_columns]
    .notna()
    .to_numpy(dtype=np.int8)
)

print("\nSTEP 7.7 — Array Preparation")
print("-" * 60)

print("Data array shape:", data_array.shape)
print("Mask array shape:", mask_array.shape)
print(
    "Original observation mask shape:",
    original_observation_mask.shape
)


# ============================================================
# STEP 7.8 — Create Data and Mask Tensors
# ============================================================

data_tensor = (
    data_array
    .reshape(
        NUM_STATIONS,
        -1,
        NUM_FEATURES
    )
    .transpose(1, 0, 2)
)

mask_tensor = (
    mask_array
    .reshape(
        NUM_STATIONS,
        -1,
        NUM_FEATURES
    )
    .transpose(1, 0, 2)
)

original_observation_mask_tensor = (
    original_observation_mask
    .reshape(
        NUM_STATIONS,
        -1,
        NUM_FEATURES
    )
    .transpose(1, 0, 2)
)

print("\nSTEP 7.8 — Tensor Preparation")
print("-" * 60)

print("Data tensor shape:", data_tensor.shape)
print("Mask tensor shape:", mask_tensor.shape)
print(
    "Original observation mask shape:",
    original_observation_mask_tensor.shape
)


# ============================================================
# STEP 7.9 — Create 24-Hour Sliding Windows
# ============================================================

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

print("\nSTEP 7.9 — Sliding Windows")
print("-" * 60)

print("Data windows shape:", data_windows.shape)
print("Mask windows shape:", mask_windows.shape)
print("Number of windows:", num_windows)


# ============================================================
# STEP 7.10 — Prepare Targets
# ============================================================

target_data = (
    data_windows[:, -1, :, :]
)

target_mask = (
    mask_windows[:, -1, :, :]
)

print("\nSTEP 7.10 — Target Preparation")
print("-" * 60)

print("Target data shape:", target_data.shape)
print("Target mask shape:", target_mask.shape)


# ============================================================
# STEP 7.11 — Create Artificial Target Mask
# ============================================================

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
    .transpose(1, 0, 2)
)

artificial_target_windows = (
    artificial_target_mask_tensor[
        WINDOW_SIZE - 1:
    ]
)

total_artificial_targets = int(
    artificial_target_windows.sum()
)

print("\nSTEP 7.11 — Artificial Target Verification")
print("-" * 60)

print(
    "Artificial target mask shape:",
    artificial_target_windows.shape
)

print(
    "Total artificial target points:",
    total_artificial_targets
)

if total_artificial_targets != 1036619:
    raise ValueError(
        "Artificial target count does not match Step 6. "
        f"Expected 1036619, got {total_artificial_targets}."
    )

print(
    "Artificial target count matches Step 6: True"
)


# ============================================================
# STEP 7.12 — Chronological Train / Validation / Test Split
# ============================================================

total_windows = len(data_windows)

train_end = int(
    total_windows * 0.70
)

val_end = int(
    total_windows * 0.85
)

X_train = data_windows[
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

Y_val = target_data[
    train_end:val_end
]

T_val = artificial_target_windows[
    train_end:val_end
]

X_test = data_windows[
    val_end:
]

Y_test = target_data[
    val_end:
]

T_test = artificial_target_windows[
    val_end:
]

print("\nSTEP 7.12 — Dataset Split")
print("-" * 60)

print("Total windows:", total_windows)

print("Training windows:", len(X_train))
print("Validation windows:", len(X_val))
print("Test windows:", len(X_test))

print(
    "Training artificial targets:",
    int(T_train.sum())
)

print(
    "Validation artificial targets:",
    int(T_val.sum())
)

print(
    "Test artificial targets:",
    int(T_test.sum())
)

print(
    "Total artificial targets:",
    int(
        T_train.sum()
        + T_val.sum()
        + T_test.sum()
    )
)

expected_train_targets = 725081
expected_val_targets = 155526
expected_test_targets = 156012

if int(T_train.sum()) != expected_train_targets:
    raise ValueError(
        "Training artificial target count does not match Step 6."
    )

if int(T_val.sum()) != expected_val_targets:
    raise ValueError(
        "Validation artificial target count does not match Step 6."
    )

if int(T_test.sum()) != expected_test_targets:
    raise ValueError(
        "Test artificial target count does not match Step 6."
    )

print("\nStep 6 split verification: PASSED")


# ============================================================
# STEP 7.13 — Calculate Training-Only Statistics
# ============================================================

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

print("\nSTEP 7.13 — Training-Only Statistics")
print("-" * 60)

for feature, mean, std in zip(
    feature_columns,
    feature_means,
    feature_stds
):

    print(
        f"{feature:20s} "
        f"mean={mean:.6f} "
        f"std={std:.6f}"
    )


# ============================================================
# STEP 7.14 — Fill Missing Input Values
# ============================================================

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

print("\nSTEP 7.14 — Missing Value Handling")
print("-" * 60)

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


# ============================================================
# STEP 7.15 — Normalize Inputs Using Training Statistics
# ============================================================

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


X_train_normalized = normalize_data(
    X_train_filled,
    feature_means,
    feature_stds
)

X_val_normalized = normalize_data(
    X_val_filled,
    feature_means,
    feature_stds
)

X_test_normalized = normalize_data(
    X_test_filled,
    feature_means,
    feature_stds
)

print("\nSTEP 7.15 — Input Normalization")
print("-" * 60)

print(
    "NaN in normalized train:",
    np.isnan(X_train_normalized).sum()
)

print(
    "NaN in normalized validation:",
    np.isnan(X_val_normalized).sum()
)

print(
    "NaN in normalized test:",
    np.isnan(X_test_normalized).sum()
)


# ============================================================
# STEP 7.16 — Normalize Targets
# ============================================================

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


Y_test_normalized = normalize_target(
    Y_test,
    feature_means,
    feature_stds
)

print("\nSTEP 7.16 — Target Normalization")
print("-" * 60)

print(
    "Y_test shape:",
    Y_test_normalized.shape
)

print(
    "NaN in Y_test:",
    np.isnan(Y_test_normalized).sum()
)


# ============================================================
# STEP 7.17 — Convert Test Data to LSTM Format
# ============================================================

X_test_lstm = (
    X_test_normalized
    .reshape(
        len(X_test_normalized),
        WINDOW_SIZE,
        INPUT_SIZE
    )
)

Y_test_lstm = (
    Y_test_normalized
    .reshape(
        len(Y_test_normalized),
        INPUT_SIZE
    )
)

T_test_lstm = (
    T_test
    .reshape(
        len(T_test),
        INPUT_SIZE
    )
)

print("\nSTEP 7.17 — Test LSTM Format")
print("-" * 60)

print("X_test:", X_test_lstm.shape)
print("Y_test:", Y_test_lstm.shape)
print("T_test:", T_test_lstm.shape)


# ============================================================
# STEP 7.18 — Define LSTM Model
# ============================================================

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
            dropout=dropout
            if num_layers > 1
            else 0.0
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_output = (
            lstm_output[:, -1, :]
        )

        output = self.fc(
            last_output
        )

        return output


# ============================================================
# STEP 7.19 — Load Trained Model
# ============================================================

print("\nSTEP 7.19 — Load Trained Model")
print("-" * 60)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}"
    )

model = LSTMBaseline()

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Model loaded successfully.")
print("Total parameters:", total_parameters)
print("Model path:", MODEL_FILE)


# ============================================================
# STEP 7.20 — Test Model Prediction
# ============================================================

print("\nSTEP 7.20 — Model Prediction")
print("-" * 60)

X_test_tensor = torch.tensor(
    X_test_lstm,
    dtype=torch.float32
)

predictions_normalized = []

with torch.no_grad():

    for start in range(
        0,
        len(X_test_tensor),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_test_tensor)
        )

        batch_X = (
            X_test_tensor[
                start:end
            ]
            .to(DEVICE)
        )

        batch_prediction = model(
            batch_X
        )

        predictions_normalized.append(
            batch_prediction
            .cpu()
            .numpy()
        )

predictions_normalized = np.concatenate(
    predictions_normalized,
    axis=0
)

print(
    "Prediction shape:",
    predictions_normalized.shape
)

print(
    "Target shape:",
    Y_test_lstm.shape
)


# ============================================================
# STEP 7.21 — Inverse Transform Predictions and Targets
# ============================================================

predictions_original = (
    predictions_normalized
    * feature_stds
) + feature_means

targets_original = (
    Y_test_lstm
    * feature_stds
) + feature_means

print("\nSTEP 7.21 — Inverse Normalization")
print("-" * 60)

print(
    "Prediction shape:",
    predictions_original.shape
)

print(
    "Target shape:",
    targets_original.shape
)


# ============================================================
# STEP 7.22 — Calculate MAE and RMSE
# ============================================================

results = []

for feature_idx, feature in enumerate(
    feature_columns
):

    feature_predictions = (
        predictions_original[
            :,
            feature_idx
            ::NUM_FEATURES
        ]
    )

    feature_targets = (
        targets_original[
            :,
            feature_idx
            ::NUM_FEATURES
        ]
    )

    feature_mask = (
        T_test_lstm[
            :,
            feature_idx
            ::NUM_FEATURES
        ]
        .astype(bool)
    )

    valid_predictions = (
        feature_predictions[
            feature_mask
        ]
    )

    valid_targets = (
        feature_targets[
            feature_mask
        ]
    )

    if len(valid_targets) == 0:

        mae = np.nan
        rmse = np.nan
        evaluation_points = 0

    else:

        errors = (
            valid_predictions
            - valid_targets
        )

        mae = np.mean(
            np.abs(errors)
        )

        rmse = np.sqrt(
            np.mean(
                errors ** 2
            )
        )

        evaluation_points = (
            len(valid_targets)
        )

    results.append(
        {
            "Feature": feature,
            "Evaluation_Points": evaluation_points,
            "MAE": mae,
            "RMSE": rmse
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# STEP 7.23 — Display Final Metrics
# ============================================================

print("\nSTEP 7.23 — Final LSTM Evaluation")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\nTotal evaluation points:")

print(
    results_df[
        "Evaluation_Points"
    ].sum()
)

print(
    "Expected total test artificial targets:",
    expected_test_targets
)

if (
    results_df[
        "Evaluation_Points"
    ].sum()
    != expected_test_targets
):

    raise ValueError(
        "Evaluation point count does not match "
        "the expected Step 6 test artificial target count."
    )

print(
    "\nEvaluation point verification: PASSED"
)


# ============================================================
# STEP 7.24 — Save Evaluation Results
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nSTEP 7.24 — Save Results")
print("-" * 60)

print(
    "Results saved:",
    RESULT_FILE
)

print(
    "Results file exists:",
    RESULT_FILE.exists()
)

if RESULT_FILE.exists():

    print(
        "Results file size:",
        RESULT_FILE.stat().st_size,
        "bytes"
    )


# ============================================================
# STEP 7.25 — Final Step 7 Completion Check
# ============================================================

print("\nSTEP 7.25 — LSTM Evaluation Completed")
print("=" * 60)

print("Model loaded:", MODEL_FILE.exists())
print(
    "Test windows:",
    len(X_test)
)
print(
    "Test artificial targets:",
    int(T_test.sum())
)
print(
    "Results saved:",
    RESULT_FILE.exists()
)

print("\nSTEP 7 completed successfully.")