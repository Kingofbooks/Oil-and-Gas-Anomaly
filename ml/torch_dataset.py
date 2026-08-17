import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):

    def __init__(self, windows: np.ndarray):
        if windows.ndim != 3:
            raise ValueError(
                "Expected windows with shape "
                "(n_windows, window_size, n_features)"
            )

        self.windows = torch.tensor(
            windows,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]

        return window