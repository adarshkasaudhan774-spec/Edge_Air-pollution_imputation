import os
import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Models",
    "lstm_baseline_new.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 208
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 208
DROPOUT = 0.1


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


def main():

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

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model loaded successfully.")
    print("Total parameters:", total_parameters)
    print("Model:", model)


if __name__ == "__main__":
    main()