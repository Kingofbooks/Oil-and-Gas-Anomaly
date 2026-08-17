from pathlib import Path

import pandas as pd

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)


# --------------------------------------------------
# 1. Discover dataset
# --------------------------------------------------

dataset = ThreeWDataset(DATASET_ROOT)

metadata = dataset.build_index()

records = [
    {
        "path": item.path,
        "well_id": item.well_id,
        "event_class": item.folder_type,
        "source": item.source,
        "num_rows": item.num_rows,
        "start_time": item.start_time,
        "end_time": item.end_time,
    }
    for item in metadata
]

metadata_df = pd.DataFrame(records)


# --------------------------------------------------
# 2. Split
# --------------------------------------------------

train, validation, test = split_real_instances(
    metadata_df
)


# --------------------------------------------------
# 3. Basic split information
# --------------------------------------------------

print("=" * 70)
print("TRAIN")
print("=" * 70)

print("Instances:", len(train))
print("Wells:", train["well_id"].nunique())


print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print("Instances:", len(validation))
print("Wells:", validation["well_id"].nunique())


print("\n" + "=" * 70)
print("TEST")
print("=" * 70)

print("Instances:", len(test))
print("Wells:", test["well_id"].nunique())


# --------------------------------------------------
# 4. Event-class distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("EVENT CLASS DISTRIBUTION")
print("=" * 70)


print("\nTRAIN:")
print(
    train["event_class"]
    .value_counts()
    .sort_index()
)


print("\nVALIDATION:")
print(
    validation["event_class"]
    .value_counts()
    .sort_index()
)


print("\nTEST:")
print(
    test["event_class"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 5. Check for well leakage
# --------------------------------------------------

train_wells = set(train["well_id"])
validation_wells = set(validation["well_id"])
test_wells = set(test["well_id"])


print("\n" + "=" * 70)
print("WELL LEAKAGE CHECK")
print("=" * 70)

print(
    "Train ∩ Validation:",
    train_wells & validation_wells
)

print(
    "Train ∩ Test:",
    train_wells & test_wells
)

print(
    "Validation ∩ Test:",
    validation_wells & test_wells
)


# --------------------------------------------------
# 6. Check totals
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOTALS")
print("=" * 70)

print(
    "Total split instances:",
    len(train) + len(validation) + len(test)
)

print(
    "Original real instances:",
    len(
        metadata_df[
            metadata_df["source"] == "real"
        ]
    )
)