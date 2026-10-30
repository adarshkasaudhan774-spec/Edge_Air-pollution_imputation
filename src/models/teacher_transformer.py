
# ============================================================
# STEP 8 — TEACHER TRANSFORMER MODEL
# Project: Edge-Efficient Air Pollution Imputation
# Purpose: Predict air quality and meteorological features
# Input:  (batch, 24, 16, 13)
# Output: (batch, 16, 13)
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# 1. Model Configuration
# ============================================================

NUM_STATIONS = 16
NUM_FEATURES = 13
WINDOW_SIZE = 24

D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1


# ============================================================
# 2. Teacher Transformer
# ============================================================

class TeacherTransformer(nn.Module):

    def __init__(
        self,
        num_features=NUM_FEATURES,
        num_stations=NUM_STATIONS,
        window_size=WINDOW_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ):

        super().__init__()

        self.num_features = num_features
        self.num_stations = num_stations
        self.window_size = window_size
        self.d_model = d_model

        # Input consists of:
        # 13 feature values + 13 observation-mask values
        self.input_projection = nn.Linear(
            num_features * 2,
            d_model
        )

        # Learnable station embeddings
        self.station_embedding = nn.Parameter(
            torch.randn(1, 1, num_stations, d_model) * 0.02
        )

        # Learnable temporal position embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(1, window_size, 1, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.output_norm = nn.LayerNorm(d_model)

        # Predict 13 features for each station
        self.output_projection = nn.Linear(
            d_model,
            num_features
        )

    # ========================================================
    # 3. Forward Pass
    # ========================================================

    def forward(self, x, mask):

        # Expected shape: (batch, time, stations, features)

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape "
                "(batch, time, stations, features)."
            )

        if mask.shape != x.shape:
            raise ValueError(
                "Input and mask shapes must match."
            )

        expected_shape = (
            self.window_size,
            self.num_stations,
            self.num_features
        )

        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected input shape (B, {expected_shape}), "
                f"but received {tuple(x.shape)}."
            )

        # Combine feature values and observation mask
        combined_input = torch.cat(
            [x, mask],
            dim=-1
        )

        # Project input to the model's hidden dimension
        hidden = self.input_projection(combined_input)

        # Add station and time information
        hidden = (
            hidden
            + self.station_embedding
            + self.position_embedding
        )

        batch_size = hidden.shape[0]

        # Process each station's 24-hour sequence.
        # Shape: (batch * stations, time, d_model)
        hidden = hidden.permute(
            0, 2, 1, 3
        ).reshape(
            batch_size * self.num_stations,
            self.window_size,
            self.d_model
        )

        # Temporal Transformer encoding
        hidden = self.transformer_encoder(hidden)

        # Use the final timestep representation
        final_hidden = hidden[:, -1, :]

        final_hidden = self.output_norm(final_hidden)

        # Predict all 13 features
        predictions = self.output_projection(final_hidden)

        # Restore station dimension
        predictions = predictions.reshape(
            batch_size,
            self.num_stations,
            self.num_features
        )

        return predictions


# ============================================================
# 4. Model Builder
# ============================================================

def build_teacher_model(device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = TeacherTransformer().to(device)

    return model


# ============================================================
# 5. Architecture Verification Test
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("STEP 8 — TEACHER TRANSFORMER ARCHITECTURE TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = build_teacher_model(device)

    # Small dummy batch for testing only
    dummy_input = torch.zeros(
        2,
        WINDOW_SIZE,
        NUM_STATIONS,
        NUM_FEATURES,
        dtype=torch.float32,
        device=device
    )

    dummy_mask = torch.ones_like(dummy_input)

    model.eval()

    with torch.no_grad():

        predictions = model(
            dummy_input,
            dummy_mask
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nInput shape:", tuple(dummy_input.shape))
    print("Mask shape:", tuple(dummy_mask.shape))
    print("Output shape:", tuple(predictions.shape))

    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    finite_output = bool(
        torch.isfinite(predictions).all()
    )

    print(
        "Output contains only finite values:",
        finite_output
    )

    assert predictions.shape == (
        2,
        NUM_STATIONS,
        NUM_FEATURES
    ), "Incorrect output shape."

    assert finite_output, "Model produced NaN or Inf values."

    print("\nTEACHER TRANSFORMER ARCHITECTURE TEST: PASSED")