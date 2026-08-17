from pathlib import Path

import pandas as pd


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

FEATURES = [
    "P-PDG",
    "T-PDG",
]


print("=" * 80)
print("MOST COMMON EXTREME VALUES")
print("=" * 80)


for feature in FEATURES:

    print("\n" + "-" * 80)
    print(feature)
    print("-" * 80)

    all_values = []

    for folder in DATASET_ROOT.iterdir():

        if not folder.is_dir():
            continue

        for file in folder.glob("*.parquet"):

            df = pd.read_parquet(file)

            values = df[feature].dropna()

            extreme = values[
                values.abs() > 1e10
            ]

            if not extreme.empty:
                all_values.append(extreme)


    if not all_values:
        print("No extreme values found.")
        continue


    values = pd.concat(
        all_values,
        ignore_index=True,
    )


    print(
        "Total extreme observations:",
        len(values),
    )


    print(
        "\nMost common extreme values:"
    )


    print(
        values.value_counts()
        .head(10)
    )