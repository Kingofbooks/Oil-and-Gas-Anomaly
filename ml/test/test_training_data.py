from pathlib import Path

import pandas as pd

from ml.dataset import ThreeWDataset
from ml.preprocessing import Preprocessor
from ml.windowing import TimeSeriesWindowGenerator
from ml.training_data import TrainingDataBuilder
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
        "folder_type": item.folder_type,
        "source": item.source,
        "num_rows": item.num_rows,
        "start_time": item.start_time,
        "end_time": item.end_time,
    }
    for item in metadata
]

metadata_df = pd.DataFrame(records)


# --------------------------------------------------
# 2. Split real wells
# --------------------------------------------------

train, validation, test = (
    split_real_instances(
        metadata_df
    )
)


print("Training instances:")
print(len(train))


print("\nTraining wells:")
print(train["well_id"].nunique())


# --------------------------------------------------
# 3. Create preprocessing + windowing
# --------------------------------------------------

preprocessor = Preprocessor()

window_generator = (
    TimeSeriesWindowGenerator(
        window_size=120,
        stride=60,
    )
)


builder = TrainingDataBuilder(
    preprocessor=preprocessor,
    window_generator=window_generator,
)


# --------------------------------------------------
# 4. Fit preprocessing ONLY on training data
# --------------------------------------------------

builder.fit_preprocessor(
    train,
    max_rows_per_instance=2000,
)


print("\nFeature columns:")
print(
    preprocessor.feature_columns
)


# --------------------------------------------------
# 5. Build training windows
# --------------------------------------------------

windows = builder.build(
    train
)


print("\nFinal training windows:")
print(windows.shape)


print("\nDtype:")
print(windows.dtype)