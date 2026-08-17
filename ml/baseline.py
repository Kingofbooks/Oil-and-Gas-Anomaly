import numpy as np

from ml.model import AnomalyModel


class MeanBaseline(AnomalyModel):

    def __init__(self):
        self.mean_profile: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> None:

        if windows.ndim != 3:
            raise ValueError(
                "Expected shape (n_windows, window_size, n_features)"
            )

        self.mean_profile = windows.mean(axis=0)

    def score(self, windows: np.ndarray) -> np.ndarray:

        if self.mean_profile is None:
            raise RuntimeError(
                "Model must be fitted before scoring."
            )

        errors = (
            windows - self.mean_profile
        ) ** 2

        return errors.mean(axis=(1, 2))