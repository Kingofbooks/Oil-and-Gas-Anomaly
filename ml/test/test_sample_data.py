from pathlib import Path

import pandas as pd


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)


SELECTED_WELLS = [
    "WELL-00011",
    "WELL-00014",
    "WELL-00016",
    "WELL-00038",
]


print("=" * 80)
print("SMALL TRAINING SAMPLE")
print("=" * 80)


total_rows = 0
total_normal_rows = 0


for folder in DATASET_ROOT.iterdir():

    if not folder.is_dir():
        continue

    for file in folder.glob("*.parquet"):

        if not file.name.startswith(
            tuple(SELECTED_WELLS)
        ):
            continue

        df = pd.read_parquet(
            file,
            columns=["class"],
        )

        rows = len(df)

        normal_rows = (
            df["class"]
            .fillna(0)
            .eq(0)
            .sum()
        )

        total_rows += rows
        total_normal_rows += normal_rows

        print(
            f"{file.name:<55}"
            f"rows={rows:<10,}"
            f"normal={normal_rows:,}"
        )


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Total rows:",
    f"{total_rows:,}"
)

print(
    "Normal rows:",
    f"{total_normal_rows:,}"
)

print(
    "Selected wells:",
    SELECTED_WELLS,
)

print("=" * 80)