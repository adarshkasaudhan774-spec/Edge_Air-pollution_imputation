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

    merged = merged.sort_values(
        ["timestamp", "station_id"]
    ).reset_index(drop=True)

    if len(merged) != len(mask):
        raise ValueError(
            "Merged dataset and mask row counts do not match."
        )

    if list(mask.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Mask columns do not match FEATURE_COLUMNS."
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

    timestamps = sorted(
        merged["timestamp"].unique()
    )

    stations = sorted(
        merged["station_id"].unique()
    )

    num_timestamps = len(timestamps)
    num_stations = len(stations)
    num_features = len(FEATURE_COLUMNS)

    print("Timestamps:", num_timestamps)
    print("Stations:", num_stations)
    print("Features:", num_features)

    timestamp_to_index = {
        timestamp: index
        for index, timestamp in enumerate(timestamps)
    }

    station_to_index = {
        station: index
        for index, station in enumerate(stations)
    }

    data_tensor = np.full(
        (
            num_timestamps,
            num_stations,
            num_features
        ),
        np.nan,
        dtype=np.float32
    )

    for row in merged.itertuples():

        t = timestamp_to_index[row.timestamp]
        s = station_to_index[row.station_id]

        for f, feature in enumerate(FEATURE_COLUMNS):

            data_tensor[t, s, f] = getattr(
                row,
                feature
            )

    mask_values = mask[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.int8
    )

    mask_tensor = mask_values.reshape(
        num_timestamps,
        num_stations,
        num_features
    )

    print("Data tensor shape:", data_tensor.shape)
    print("Mask tensor shape:", mask_tensor.shape)

    return data_tensor, mask_tensor


def handle_missing_values(
    data_tensor
):

    print()
    print("=" * 60)
    print("STEP 7.4 — Missing Value Handling")
    print("=" * 60)

    feature_means = np.nanmean(
        data_tensor,
        axis=(0, 1)
    )

    feature_stds = np.nanstd(
        data_tensor,
        axis=(0, 1)
    )

    feature_stds[
        feature_stds == 0
    ] = 1.0

    filled_data = data_tensor.copy()

    for feature_index in range(
        len(FEATURE_COLUMNS)
    ):

        missing = np.isnan(
            filled_data[:, :, feature_index]
        )

        filled_data[
            missing,
            feature_index
        ] = feature_means[feature_index]

    normalized_data = (
        filled_data - feature_means
    ) / feature_stds

    print(
        "NaN after filling:",
        np.isnan(filled_data).sum()
    )

    print(
        "NaN after normalization:",
        np.isnan(normalized_data).sum()
    )

    return (
        normalized_data,
        feature_means,
        feature_stds
    )


def create_windows(
    data,
    mask
):

    print()
    print("=" * 60)
    print("STEP 7.5 — Sliding Window Preparation")
    print("=" * 60)

    num_windows = (
        len(data) - WINDOW_SIZE
    )

    X = np.zeros(
        (
            num_windows,
            WINDOW_SIZE,
            data.shape[1],
            data.shape[2]
        ),
        dtype=np.float32
    )

    Y = np.zeros(
        (
            num_windows,
            data.shape[1],
            data.shape[2]
        ),
        dtype=np.float32
    )

    target_mask = np.zeros(
        (
            num_windows,
            data.shape[1],
            data.shape[2]
        ),
        dtype=bool
    )

    for i in range(num_windows):

        X[i] = data[
            i:i + WINDOW_SIZE
        ]

        Y[i] = data[
            i + WINDOW_SIZE
        ]

        target_mask[i] = (
            mask[
                i + WINDOW_SIZE
            ] == 1
        )

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Target mask shape:", target_mask.shape)

    return X, Y, target_mask


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
    print("STEP 7.6 — LSTM Test Evaluation")
    print("=" * 60)

    X_test_flat = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        -1
    )

    X_tensor = torch.tensor(
        X_test_flat,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        predictions = model(
            X_tensor
        )

    predictions = predictions.cpu().numpy()

    predictions = predictions.reshape(
        -1,
        len(FEATURE_COLUMNS)
    )

    targets = Y_test.reshape(
        -1,
        len(FEATURE_COLUMNS)
    )

    masks = target_mask.reshape(
        -1,
        len(FEATURE_COLUMNS)
    )

    predictions_original = (
        predictions * feature_stds
    ) + feature_means

    targets_original = (
        targets * feature_stds
    ) + feature_means

    results = []

    for feature_index, feature in enumerate(
        FEATURE_COLUMNS
    ):

        valid = masks[
            :,
            feature_index
        ]

        if valid.sum() == 0:
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
            np.mean(errors ** 2)
        )

        results.append(
            {
                "Feature": feature,
                "Evaluation_Points": int(
                    valid.sum()
                ),
                "MAE": float(mae),
                "RMSE": float(rmse)
            }
        )

    results_df = pd.DataFrame(
        results
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
    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("Results saved to:")
    print(RESULTS_PATH)

    return results_df


def main():

    model = load_model()

    merged, mask = load_dataset()

    data_tensor, mask_tensor = prepare_tensors(
        merged,
        mask
    )

    (
        normalized_data,
        feature_means,
        feature_stds
    ) = handle_missing_values(
        data_tensor
    )

    X, Y, target_mask = create_windows(
        normalized_data,
        mask_tensor
    )

    total_windows = len(X)

    train_end = int(
        total_windows * 0.70
    )

    val_end = int(
        total_windows * 0.85
    )

    X_test = X[
        val_end:
    ]

    Y_test = Y[
        val_end:
    ]

    target_mask_test = target_mask[
        val_end:
    ]

    print()
    print("Test windows:", X_test.shape[0])
    print(
        "Artificial test targets:",
        int(target_mask_test.sum())
    )

    evaluate_model(
        model,
        X_test,
        Y_test,
        target_mask_test,
        feature_means,
        feature_stds
    )


if __name__ == "__main__":
    main()