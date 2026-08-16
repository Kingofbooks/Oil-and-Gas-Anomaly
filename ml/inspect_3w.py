from pathlib import Path
import pandas as pd


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

file_path = DATASET_ROOT / "0" / "WELL-00001_20170201010207.parquet"

print("Reading:")
print(file_path)

df = pd.read_parquet(file_path)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nIndex information:")
print("Index name:", df.index.name)
print("Index type:", type(df.index))
print("Index dtype:", df.index.dtype)

print("\nFirst timestamp:", df.index[0])
print("Last timestamp:", df.index[-1])

print("\nTime difference between first two rows:")
print(df.index[1] - df.index[0])

print("\nClass distribution:")
print(df["class"].value_counts(dropna=False))

print("\nState distribution:")
print(df["state"].value_counts(dropna=False))

print("\nMissing percentage per column:")
print((df.isna().mean() * 100).sort_values(ascending=False))