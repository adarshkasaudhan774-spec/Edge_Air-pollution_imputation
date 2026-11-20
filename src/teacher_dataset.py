# ============================================================
# TEACHER TRANSFORMER — NORMALIZED DATASET
# Uses existing 24-hour window logic from STEP 5.24
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherTransformerDataset(Dataset):

    def __init__(
        self,
        model_input,
        ground_truth,
        observation_mask,
        target_mask,
        start_idx,
        end_idx,
        window_size=24
    ):

        self.model_input = model_input
        self.ground_truth = ground_truth
        self.observation_mask = observation_mask
        self.target_mask = target_mask

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.window_size = window_size

    def __len__(self):

        return (
            self.end_idx
            - self.start_idx
            - self.window_size
            + 1
        )

    def __getitem__(self, idx):

        window_start = self.start_idx + idx
        target_idx = window_start + self.window_size - 1
        window_end = target_idx + 1

        # Normalized 24-hour input
        x = self.model_input[window_start:window_end]

        # Original observation mask
        input_mask = self.observation_mask[
            window_start:window_end
        ]

        # Normalized target
        y = np.nan_to_num(
            self.ground_truth[target_idx],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Artificially hidden valid target positions
        y_mask = self.target_mask[target_idx]

        return (
            torch.from_numpy(x),
            torch.from_numpy(input_mask),
            torch.from_numpy(y),
            torch.from_numpy(y_mask)
        )