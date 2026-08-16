from pathlib import Path

import pandas as pd

from ml.preprocessing import Preprocessor


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

file_path = (
    DATASET_ROOT
    / "0"
    / "WELL-00001_20170201010207.parquet"
)


df = pd.read_parquet(file_path)

print("Original shape:")
print(df.shape)


preprocessor = Preprocessor()

preprocessor.fit(df)

processed = preprocessor.transform(df)


print("\nSelected features:")
print(preprocessor.feature_columns)

print("\nProcessed shape:")
print(processed.shape)

print("\nMissing values:")
print(processed.isna().sum().sum())

print("\nFirst rows:")
print(processed.head())