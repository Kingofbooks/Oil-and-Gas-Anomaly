from pathlib import Path

import pandas as pd

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)


dataset = ThreeWDataset(DATASET_ROOT)

metadata = dataset.build_index()

records = [
    {
        "path": item.path,
        "well_id": item.well_id,
        "folder_type": item.folder_type,
        "source": item.source,
        "num_rows": item.num_rows,
        "start_time": item.start_time,
        "end_time": item.end_time,
    }
    for item in metadata
]

df = pd.DataFrame(records)

train, validation, test = split_real_instances(df)


print("TRAIN")
print("Instances:", len(train))
print("Wells:", train["well_id"].nunique())

print("\nVALIDATION")
print("Instances:", len(validation))
print("Wells:", validation["well_id"].nunique())

print("\nTEST")
print("Instances:", len(test))
print("Wells:", test["well_id"].nunique())

print("\nTotal real instances:")
print(len(train) + len(validation) + len(test))

print("\nTotal real wells:")
print(
    train["well_id"].nunique()
    + validation["well_id"].nunique()
    + test["well_id"].nunique()
)

print("\nTrain wells:")
print(sorted(train["well_id"].unique()))

print("\nValidation wells:")
print(sorted(validation["well_id"].unique()))

print("\nTest wells:")
print(sorted(test["well_id"].unique()))