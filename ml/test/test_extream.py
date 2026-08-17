from pathlib import Path

import pandas as pd


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)


FEATURES = [
    "P-PDG",
    "T-PDG",
    "P-TPT",
]


# ------------------------------------------------------------
# Find extreme values
# ------------------------------------------------------------

print("=" * 80)
print("SEARCHING FOR EXTREME TRAINING VALUES")
print("=" * 80)


results = []


for folder in DATASET_ROOT.iterdir():

    if not folder.is_dir():
        continue

    for file in folder.glob("*.parquet"):

        df = pd.read_parquet(file)


        # Only normal observations
        normal = df[
            df["class"] == 0
        ].copy()


        if normal.empty:
            continue


        for feature in FEATURES:

            if feature not in normal.columns:
                continue


            values = normal[feature].dropna()


            if values.empty:
                continue


            # Find largest absolute value
            idx = values.abs().idxmax()

            value = values.loc[idx]


            results.append(
                {
                    "file": file.name,
                    "folder": folder.name,
                    "feature": feature,
                    "value": value,
                    "timestamp": idx,
                }
            )


# ------------------------------------------------------------
# Show largest absolute values
# ------------------------------------------------------------

result_df = pd.DataFrame(results)


result_df["abs_value"] = (
    result_df["value"].abs()
)


result_df = result_df.sort_values(
    "abs_value",
    ascending=False,
)


print("\nTop 30 extreme values:\n")


print(
    result_df[
        [
            "feature",
            "value",
            "file",
            "folder",
            "timestamp",
        ]
    ].head(30).to_string(
        index=False
    )
)

for feature in FEATURES:

    print(f"\n{feature}")

    total = 0

    for folder in DATASET_ROOT.iterdir():

        if not folder.is_dir():
            continue

        for file in folder.glob("*.parquet"):

            df = pd.read_parquet(file)

            values = df[feature].dropna()

            count = (
                values.abs() > 1e10
            ).sum()

            total += count

    print(
        f"|value| > 1e10: {total}"
    )


print("\n" + "=" * 80)
print("EXTREME VALUE COUNTS")
print("=" * 80)