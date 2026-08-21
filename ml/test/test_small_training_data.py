from pathlib import Path

import pandas as pd

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)


print("=" * 80)
print("SMALL TRAINING DATA INSPECTION")
print("=" * 80)


# ------------------------------------------------------------
# 1. Build dataset index
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. Same split used by training
# ------------------------------------------------------------

train_metadata, validation_metadata, test_metadata = (
    split_real_instances(metadata_df)
)


print("\nDataset split:")
print("Train:", len(train_metadata))
print("Validation:", len(validation_metadata))
print("Test:", len(test_metadata))


# ------------------------------------------------------------
# 3. Training wells
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING WELLS")
print("=" * 80)


for well_id, group in train_metadata.groupby("well_id"):

    print(
        f"{well_id:<15}"
        f" files={len(group):<5}"
        f" rows={group['num_rows'].sum():,}"
    )


# ------------------------------------------------------------
# 4. Training event/source distribution
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING SOURCES")
print("=" * 80)

print(
    train_metadata["source"]
    .value_counts()
)


print("\n" + "=" * 80)
print("TRAINING EVENT TYPES")
print("=" * 80)

print(
    train_metadata["event_class"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)