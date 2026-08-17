from pathlib import Path

import pandas as pd

from ml.dataset import ThreeWDataset
from ml.preprocessing import Preprocessor
from ml.split import split_real_instances


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

OUTLIER_FILE = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset\5\WELL-00015_20170620122925.parquet"
)


# ============================================================
# 1. Build metadata
# ============================================================

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


train_metadata, _, _ = split_real_instances(
    metadata_df
)


# ============================================================
# 2. Build training NORMAL data
# ============================================================

training_frames = []

for row in train_metadata.itertuples(
    index=False
):

    df = pd.read_parquet(row.path)

    normal = df[
        df["class"] == 0
    ].copy()

    if normal.empty:
        continue

    normal = normal.iloc[:2000]

    training_frames.append(
        normal
    )


training_data = pd.concat(
    training_frames,
    axis=0,
)


# ============================================================
# 3. Fit preprocessor
# ============================================================

preprocessor = Preprocessor()

preprocessor.fit(
    training_data
)


# ============================================================
# 4. Load outlier file
# ============================================================

outlier = pd.read_parquet(
    OUTLIER_FILE
)


outlier_normal = outlier[
    outlier["class"] == 0
].copy()


# ============================================================
# 5. Compare raw distributions
# ============================================================

features = (
    preprocessor.feature_columns
)


print("=" * 80)
print("TRAINING VS OUTLIER EVENT")
print("=" * 80)


for feature in features:

    train_min = (
        training_data[feature]
        .min()
    )

    train_max = (
        training_data[feature]
        .max()
    )

    train_mean = (
        training_data[feature]
        .mean()
    )

    event_min = (
        outlier_normal[feature]
        .min()
    )

    event_max = (
        outlier_normal[feature]
        .max()
    )

    event_mean = (
        outlier_normal[feature]
        .mean()
    )


    print(
        f"\n{feature}"
    )

    print(
        f"  TRAIN  min={train_min:.6g} "
        f"max={train_max:.6g} "
        f"mean={train_mean:.6g}"
    )

    print(
        f"  EVENT  min={event_min:.6g} "
        f"max={event_max:.6g} "
        f"mean={event_mean:.6g}"
    )


# ============================================================
# 6. Standardized event values
# ============================================================

processed = preprocessor.transform(
    outlier_normal
)


print("\n" + "=" * 80)
print("STANDARDIZED EVENT EXTREMES")
print("=" * 80)


for feature in features:

    values = processed[
        feature
    ]

    print(
        f"{feature:20s} "
        f"min={values.min():9.3f} "
        f"max={values.max():9.3f} "
        f"mean={values.mean():9.3f}"
    )