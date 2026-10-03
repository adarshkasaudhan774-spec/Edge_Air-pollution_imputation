import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

AIR_QUALITY_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "raw data",
    "air_quality_2019_2021_hourly.csv"
)

METEOROLOGY_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "raw data",
    "hourly_meteorology_16stations_2019_2021.csv"
)

MASK_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "processed",
    "fixed_artificial_mask_new.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Models",
    "lstm_baseline_new.pth"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "Results"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "lstm_evaluation_results.csv"
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 208
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 208
DROPOUT = 0.1
WINDOW_SIZE = 24

FEATURE_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "o3",
    "so2",
    "nox",
    "co",
    "temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "rainfall"
]

NUM_STATIONS = 16
NUM_FEATURES = len(FEATURE_COLUMNS)


class LSTMBaseline(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


def load_model():

    print("=" * 60)
    print("STEP 7.1 — LSTM Model Loading")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Model path:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = LSTMBaseline(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model loaded successfully.")
    print("Total parameters:", total_parameters)

    if total_parameters != 331984:
        raise ValueError(
            f"Unexpected parameter count: {total_parameters}"
        )

    return model


def load_dataset():

    print()
    print("=" * 60)
    print("STEP 7.2 — Dataset Loading")
    print("=" * 60)

    air_quality = pd.read_csv(
        AIR_QUALITY_PATH
    )

    meteorology = pd.read_csv(
        METEOROLOGY_PATH
    )

    mask = pd.read_csv(
        MASK_PATH
    )

    print("Air quality shape:", air_quality.shape)
    print("Meteorology shape:", meteorology.shape)
    print("Mask shape:", mask.shape)

    air_quality["timestamp"] = pd.to_datetime(
        air_quality["timestamp"]
    )

    meteorology["timestamp"] = pd.to_datetime(
        meteorology["timestamp"]
    )

    merged = pd.merge(
        air_quality,
        meteorology,
        on=[
            "station_id",
            "station_name",
            "timestamp"
        ],
        how="inner"
    )

    print("Merged shape:", merged.shape)

    if len(merged) != len(mask):
        raise ValueError(
            "Merged dataset and mask row counts do not match."
        )

    if list(mask.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Mask columns do not match FEATURE_COLUMNS."
        )

    sort_order = (
        merged
        .sort_values(
            ["timestamp", "station_id"],
            kind="mergesort"
        )
        .index
    )

    merged = (
        merged
        .loc[sort_order]
        .reset_index(drop=True)
    )

    mask = (
        mask
        .loc[sort_order]
        .reset_index(drop=True)
    )

    if not merged[
        ["timestamp", "station_id"]
    ].reset_index(drop=True).equals(
        merged[
            ["timestamp", "station_id"]
        ].reset_index(drop=True)
    ):
        raise ValueError(
            "Dataset alignment verification failed."
        )

    print("Dataset and mask aligned successfully.")

    return merged, mask


def prepare_tensors(
    merged,
    mask
):

    print()
    print("=" * 60)
    print("STEP 7.3 — Tensor Preparation")
    print("=" * 60)

    num_stations = merged["station_id"].nunique()

    if num_stations != NUM_STATIONS:
        raise ValueError(
            f"Expected {NUM_STATIONS} stations, "
            f"found {num_stations}."
        )

    data_array = (
        merged[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    mask_array = (
        mask[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    original_observation_mask_array = (
        merged[
            FEATURE_COLUMNS
        ]
        .notna()
        .to_numpy(
            dtype=np.int8
        )
    )

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
        original_observation_mask_array
        .reshape(
            NUM_STATIONS,
            -1,
            NUM_FEATURES
        )
        .transpose(1, 0, 2)
    )

    print(
        "Stations:",
        num_stations
    )

    print(
        "Features:",
        NUM_FEATURES
    )

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

    return (
        data_tensor,
        mask_tensor,
        original_observation_mask_tensor
    )


def create_windows(
    data_tensor,
    mask_tensor,
    original_observation_mask
):

    print()
    print("=" * 60)
    print("STEP 7.4 — Sliding Window Preparation")
    print("=" * 60)

    data_windows = []

    mask_windows = []

    target_data = []

    artificial_target_mask = (
        (original_observation_mask == 1)
        &
        (mask_tensor == 0)
    )

    artificial_target_windows = (
        artificial_target_mask[
            WINDOW_SIZE - 1:
        ]
    )

    for i in range(
        len(data_tensor) - WINDOW_SIZE + 1
    ):

        data_windows.append(
            data_tensor[
                i:i + WINDOW_SIZE
            ]
        )

        mask_windows.append(
            mask_tensor[
                i:i + WINDOW_SIZE
            ]
        )

        target_data.append(
            data_tensor[
                i + WINDOW_SIZE - 1
            ]
        )

    data_windows = np.asarray(
        data_windows,
        dtype=np.float32
    )

    mask_windows = np.asarray(
        mask_windows,
        dtype=np.int8
    )

    target_data = np.asarray(
        target_data,
        dtype=np.float32
    )

    print(
        "X shape:",
        data_windows.shape
    )

    print(
        "Mask windows shape:",
        mask_windows.shape
    )

    print(
        "Y shape:",
        target_data.shape
    )

    print(
        "Artificial target mask shape:",
        artificial_target_windows.shape
    )

    total_artificial_targets = int(
        artificial_target_windows.sum()
    )

    print(
        "Total artificial target points:",
        total_artificial_targets
    )

    expected_total = 1036619

    if total_artificial_targets != expected_total:
        raise ValueError(
            "Artificial target count mismatch. "
            f"Expected {expected_total}, "
            f"got {total_artificial_targets}."
        )

    return (
        data_windows,
        mask_windows,
        target_data,
        artificial_target_windows
    )


def split_data(
    data_windows,
    mask_windows,
    target_data,
    artificial_target_windows
):

    print()
    print("=" * 60)
    print("STEP 7.5 — Chronological Train/Validation/Test Split")
    print("=" * 60)

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

    print(
        "Total windows:",
        total_windows
    )

    print(
        "Train windows:",
        len(X_train)
    )

    print(
        "Validation windows:",
        len(X_val)
    )

    print(
        "Test windows:",
        len(X_test)
    )

    train_targets = int(
        T_train.sum()
    )

    val_targets = int(
        T_val.sum()
    )

    test_targets = int(
        T_test.sum()
    )

    print(
        "Training artificial targets:",
        train_targets
    )

    print(
        "Validation artificial targets:",
        val_targets
    )

    print(
        "Test artificial targets:",
        test_targets
    )

    if train_targets != 725081:
        raise ValueError(
            f"Training artificial target mismatch: "
            f"{train_targets}"
        )

    if val_targets != 155526:
        raise ValueError(
            f"Validation artificial target mismatch: "
            f"{val_targets}"
        )

    if test_targets != 156012:
        raise ValueError(
            f"Test artificial target mismatch: "
            f"{test_targets}"
        )

    print(
        "Step 6 split verification: PASSED"
    )

    return (
        X_train,
        M_train,
        Y_train,
        T_train,
        X_val,
        M_val,
        Y_val,
        T_val,
        X_test,
        M_test,
        Y_test,
        T_test
    )


def normalize_using_training_stats(
    X_train,
    X_val,
    X_test,
    Y_train,
    Y_val,
    Y_test
):

    print()
    print("=" * 60)
    print("STEP 7.6 — Training-Only Normalization")
    print("=" * 60)

    train_values_flat = X_train.reshape(
        -1,
        NUM_FEATURES
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

    def fill_and_normalize(
        values
    ):

        values = values.copy()

        if values.ndim == 4:

            for feature_index in range(
                NUM_FEATURES
            ):

                missing = np.isnan(
                    values[:, :, :, feature_index]
                )

                values[
                    :, :, :, feature_index
                ][missing] = (
                    feature_means[
                        feature_index
                    ]
                )

        elif values.ndim == 3:

            for feature_index in range(
                NUM_FEATURES
            ):

                missing = np.isnan(
                    values[:, :, feature_index]
                )

                values[
                    :, :, feature_index
                ][missing] = (
                    feature_means[
                        feature_index
                    ]
                )

        values = (
            values - feature_means
        ) / feature_stds

        return values

    X_train = fill_and_normalize(
        X_train
    )

    X_val = fill_and_normalize(
        X_val
    )

    X_test = fill_and_normalize(
        X_test
    )

    Y_train = fill_and_normalize(
        Y_train
    )

    Y_val = fill_and_normalize(
        Y_val
    )

    Y_test = fill_and_normalize(
        Y_test
    )

    print(
        "NaN after normalization:",
        (
            np.isnan(X_train).sum()
            + np.isnan(X_val).sum()
            + np.isnan(X_test).sum()
            + np.isnan(Y_train).sum()
            + np.isnan(Y_val).sum()
            + np.isnan(Y_test).sum()
        )
    )

    return (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test,
        feature_means,
        feature_stds
    )


def evaluate_model(
    model,
    X_test,
    Y_test,
    target_mask,
    feature_means,
    feature_stds
):

    print()
    print("=" * 60)
    print("STEP 7.7 — LSTM Test Evaluation")
    print("=" * 60)

    X_test_flat = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        -1
    )

    Y_test_flat = Y_test.reshape(
        Y_test.shape[0],
        -1
    )

    predictions_list = []

    batch_size = 256

    for start in range(
        0,
        len(X_test_flat),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(X_test_flat)
        )

        X_batch = torch.tensor(
            X_test_flat[start:end],
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():

            predictions = model(
                X_batch
            )

        predictions_list.append(
            predictions
            .cpu()
            .numpy()
        )

    predictions = np.concatenate(
        predictions_list,
        axis=0
    )

    predictions = predictions.reshape(
        -1,
        NUM_FEATURES
    )

    targets = Y_test_flat.reshape(
        -1,
        NUM_FEATURES
    )

    masks = target_mask.reshape(
        -1,
        NUM_FEATURES
    )

    predictions_original = (
        predictions * feature_stds
    ) + feature_means

    targets_original = (
        targets * feature_stds
    ) + feature_means

    results = []

    total_evaluation_points = 0

    for feature_index, feature in enumerate(
        FEATURE_COLUMNS
    ):

        valid = masks[
            :,
            feature_index
        ]

        evaluation_points = int(
            valid.sum()
        )

        if evaluation_points == 0:
            continue

        prediction_values = (
            predictions_original[
                valid,
                feature_index
            ]
        )

        target_values = (
            targets_original[
                valid,
                feature_index
            ]
        )

        errors = (
            prediction_values
            - target_values
        )

        mae = np.mean(
            np.abs(errors)
        )

        rmse = np.sqrt(
            np.mean(
                errors ** 2
            )
        )

        total_evaluation_points += (
            evaluation_points
        )

        results.append(
            {
                "Feature": feature,
                "Evaluation_Points": evaluation_points,
                "MAE": float(mae),
                "RMSE": float(rmse)
            }
        )

    results_df = pd.DataFrame(
        results
    )

    expected_evaluation_points = 156012

    print()
    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Total evaluation points:",
        total_evaluation_points
    )

    if total_evaluation_points != expected_evaluation_points:
        raise ValueError(
            "Evaluation point mismatch. "
            f"Expected {expected_evaluation_points}, "
            f"got {total_evaluation_points}."
        )

    print(
        "Evaluation point verification: PASSED"
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print()
    print("Results saved to:")
    print(RESULTS_PATH)

    return results_df


def main():

    model = load_model()

    merged, mask = load_dataset()

    (
        data_tensor,
        mask_tensor,
        original_observation_mask
    ) = prepare_tensors(
        merged,
        mask
    )

    (
        data_windows,
        mask_windows,
        target_data,
        artificial_target_windows
    ) = create_windows(
        data_tensor,
        mask_tensor,
        original_observation_mask
    )

    (
        X_train,
        M_train,
        Y_train,
        T_train,
        X_val,
        M_val,
        Y_val,
        T_val,
        X_test,
        M_test,
        Y_test,
        T_test
    ) = split_data(
        data_windows,
        mask_windows,
        target_data,
        artificial_target_windows
    )

    (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test,
        feature_means,
        feature_stds
    ) = normalize_using_training_stats(
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test
    )

    evaluate_model(
        model,
        X_test,
        Y_test,
        T_test,
        feature_means,
        feature_stds
    )

    print()
    print("=" * 60)
    print("STEP 7 completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()