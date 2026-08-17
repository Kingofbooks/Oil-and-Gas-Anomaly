from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing import Preprocessor
from ml.windowing import TimeSeriesWindowGenerator


class TrainingDataBuilder:

    def __init__(
        self,
        preprocessor: Preprocessor,
        window_generator: TimeSeriesWindowGenerator,
    ):
        self.preprocessor = preprocessor
        self.window_generator = window_generator

    def fit_preprocessor(
        self,
        metadata: pd.DataFrame,
        max_rows_per_instance: int = 2000,
    ) -> None:

        training_frames = []

        for index, row in enumerate(metadata.itertuples(index=False),start=1):
            path = row.path
            print(
                f"Fitting preprocessor "
                f"{index}/{len(metadata)}: "
                f"{path.name}"
            )

            df = pd.read_parquet(path)

            # Only normal observations are used
            # to learn preprocessing parameters.
            normal = df[
                df["class"] == 0
            ].copy()

            if normal.empty:
                continue

            # Limit how much data from each file
            # contributes to fitting.
            if len(normal) > max_rows_per_instance:
                normal = normal.iloc[
                    :max_rows_per_instance
                ]

            training_frames.append(normal)

        if not training_frames:
            raise RuntimeError(
                "No normal training data found."
            )

        combined = pd.concat(
            training_frames,
            axis=0,
        )

        print(
            "\nFitting preprocessor on:"
        )
        print(
            f"{len(combined)} normal rows"
        )

        self.preprocessor.fit(
            combined
        )

    def process_instance(
        self,
        path: Path,
    ) -> np.ndarray:

        df = pd.read_parquet(path)

        # Keep only normal observations.
        normal = df[
            df["class"] == 0
        ].copy()

        if len(normal) < (
            self.window_generator.window_size
        ):
            return np.empty(
                (
                    0,
                    self.window_generator.window_size,
                    len(
                        self.preprocessor.feature_columns
                    ),
                ),
                dtype=np.float32,
            )

        processed = (
            self.preprocessor.transform(
                normal
            )
        )

        windows = (
            self.window_generator.create_windows(
                processed
            )
        )

        return windows

    def build(
        self,
        metadata: pd.DataFrame,
    ) -> np.ndarray:

        all_windows = []

        for index, row in enumerate(metadata.itertuples(index=False),start=1):

            path = row.path

            print(
                f"Fitting preprocessor "
                f"{index}/{len(metadata)}: "
                f"{path.name}"
            )

            windows = self.process_instance(
                path
            )

            if len(windows) > 0:
                all_windows.append(
                    windows
                )

        if not all_windows:
            raise RuntimeError(
                "No training windows were created."
            )

        return np.concatenate(
            all_windows,
            axis=0,
        )