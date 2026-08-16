import numpy as np
import pandas as pd


class TimeSeriesWindowGenerator:

    def __init__(self,window_size: int = 120,stride: int = 60):
        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than 0"
            )

        if stride <= 0:
            raise ValueError(
                "stride must be greater than 0"
            )

        self.window_size = window_size
        self.stride = stride

    def create_windows(self,data: pd.DataFrame) -> np.ndarray:
        
        values = data.to_numpy(dtype=np.float32)

        n_rows = len(values)

        if n_rows < self.window_size:
            return np.empty(
                (0, self.window_size, values.shape[1]),
                dtype=np.float32,
            )

        windows = []

        for start in range(0,n_rows - self.window_size + 1,self.stride):
            end = start + self.window_size
            windows.append(values[start:end])

        return np.stack(windows)