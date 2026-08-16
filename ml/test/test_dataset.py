from pathlib import Path

import pandas as pd

from ml.dataset import ThreeWDataset


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

dataset = ThreeWDataset(DATASET_ROOT)

instances = dataset.list_instances()

print("Inspecting first 20 instances...\n")

for path in instances[:20]:

    df = pd.read_parquet(path)

    class_values = (
        df["class"]
        .dropna()
        .unique()
        .tolist()
    )

    state_values = (
        df["state"]
        .dropna()
        .unique()
        .tolist()
    )

    print("=" * 80)

    print("File:", path.name)
    print("Folder:", path.parent.name)

    if path.name.startswith("SIMULATED_"):
        print("Source: simulated")
    else:
        print("Source: real")

    print("Rows:", len(df))

    print("Class values:", class_values)
    print("State values:", state_values)