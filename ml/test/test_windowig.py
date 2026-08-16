from pathlib import Path

import pandas as pd

from ml.preprocessing import Preprocessor
from ml.windowing import TimeSeriesWindowGenerator


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

file = (
    DATASET_ROOT
    / "0"
    / "WELL-00001_20170201010207.parquet"
)


df = pd.read_parquet(file)

print("Original shape:")
print(df.shape)


preprocessor = Preprocessor()

preprocessor.fit(df)
processed = preprocessor.transform(df)

print("\nProcessed shape:")
print(processed.shape)


generator = TimeSeriesWindowGenerator(
    window_size=120,
    stride=60,
)

windows = generator.create_windows(processed)

print("\nWindow shape:")
print(windows.shape)

print("\nWindow dtype:")
print(windows.dtype)

print("\nFirst window:")
print(windows[0])

print("\nFirst window shape:")
print(windows[0].shape)