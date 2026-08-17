from pathlib import Path

import pandas as pd

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances
from ml.preprocessing import Preprocessor
from ml.tranad_detector import TranadAnomalyDetector


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

MODEL_PATH = Path(
    "artifacts/tranad_v1.pt"
)


# ============================================================
# 1. BUILD DATASET INDEX
# ============================================================

print("=" * 70)
print("TRANAD DETECTOR TEST")
print("=" * 70)

print("\nBuilding dataset index...")

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


# ============================================================
# 2. CREATE SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_metadata, validation_metadata, test_metadata = (
    split_real_instances(metadata_df)
)

print("\nDataset split:")

print("Train:", len(train_metadata))
print("Validation:", len(validation_metadata))
print("Test:", len(test_metadata))


# ============================================================
# 3. FIT PREPROCESSOR USING TRAINING DATA ONLY
# ============================================================

print("\nFitting preprocessor...")

training_frames = []

for row in train_metadata.itertuples(index=False):

    df = pd.read_parquet(row.path)

    # Only normal data is used to fit the scaler.
    normal = df[
        df["class"] == 0
    ].copy()

    if normal.empty:
        continue

    training_frames.append(normal)


training_data = pd.concat(
    training_frames,
    axis=0,
)


preprocessor = Preprocessor()

preprocessor.fit(
    training_data
)

print(
    "Features:",
    len(preprocessor.feature_columns)
)

print(
    preprocessor.feature_columns
)


# ============================================================
# 4. CREATE TRANAD DETECTOR
# ============================================================

print("\nLoading TranAD detector...")

detector = TranadAnomalyDetector(
    model_path=MODEL_PATH,
    preprocessor=preprocessor,
    threshold=0.03,
    window_size=120,
    stride=60,
)


print("Detector loaded successfully.")


# ============================================================
# 5. TEST ON ONE NORMAL FILE
# ============================================================

normal_row = train_metadata[
    train_metadata["event_class"] == 0
].iloc[0]

print("\n" + "=" * 70)
print("NORMAL FILE TEST")
print("=" * 70)

print(
    "File:",
    Path(normal_row["path"]).name
)


normal_df = pd.read_parquet(
    normal_row["path"]
)


normal_result = detector.detect(
    normal_df
)


print("\nResult:")
print(
    "Model:",
    normal_result.model_name
)

print(
    "Version:",
    normal_result.model_version
)

print(
    "Anomaly score:",
    normal_result.anomaly_score
)

print(
    "Is anomaly:",
    normal_result.is_anomaly
)


# ============================================================
# 6. TEST ON KNOWN ANOMALY FILE
# ============================================================

anomaly_rows = validation_metadata[
    validation_metadata["event_class"] != 0
]


if len(anomaly_rows) > 0:

    anomaly_row = anomaly_rows.iloc[0]

    print("\n" + "=" * 70)
    print("ANOMALY FILE TEST")
    print("=" * 70)

    print(
        "File:",
        Path(anomaly_row["path"]).name
    )

    print(
        "Event class:",
        anomaly_row["event_class"]
    )

    anomaly_df = pd.read_parquet(
        anomaly_row["path"]
    )

    anomaly_result = detector.detect(
        anomaly_df
    )

    print("\nResult:")

    print(
        "Model:",
        anomaly_result.model_name
    )

    print(
        "Version:",
        anomaly_result.model_version
    )

    print(
        "Anomaly score:",
        anomaly_result.anomaly_score
    )

    print(
        "Is anomaly:",
        anomaly_result.is_anomaly
    )


# ============================================================
# 7. VERIFY RESULT CONTRACT
# ============================================================

print("\n" + "=" * 70)
print("CONTRACT CHECK")
print("=" * 70)

assert normal_result.model_name == "TranAD"

assert normal_result.model_version == "v1"

assert isinstance(
    normal_result.anomaly_score,
    float,
)

assert isinstance(
    normal_result.is_anomaly,
    bool,
)


print("DetectionResult contract: PASSED")


# ============================================================
# 8. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRANAD DETECTOR TEST COMPLETE")
print("=" * 70)