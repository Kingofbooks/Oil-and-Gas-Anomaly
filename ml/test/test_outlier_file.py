from pathlib import Path

import pandas as pd


FILE = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset\5\WELL-00015_20170620122925.parquet"
)


df = pd.read_parquet(FILE)


print("=" * 70)
print("FILE")
print("=" * 70)

print(FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    df["class"].value_counts(
        dropna=False
    )
)


print("\n" + "=" * 70)
print("STATE DISTRIBUTION")
print("=" * 70)

print(
    df["state"].value_counts(
        dropna=False
    )
)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    df.isna()
    .mean()
    .sort_values(
        ascending=False
    )
    .head(30)
)


print("\n" + "=" * 70)
print("NUMERIC SUMMARY")
print("=" * 70)

print(
    df.describe()
    .T[
        [
            "min",
            "max",
            "mean",
            "std",
        ]
    ]
    .sort_values(
        "max",
        ascending=False,
    )
)


print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(
    df.head()
)


print("\n" + "=" * 70)
print("LAST 5 ROWS")
print("=" * 70)

print(
    df.tail()
)