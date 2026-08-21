from pathlib import Path

import pandas as pd
import torch

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances
from ml.preprocessing import Preprocessor
from ml.tranad_detector import TranADDetector


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

MODEL_PATH = Path(
    "artifacts/tranad_v1.pt"
)

THRESHOLD = 0.029617823120545266

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TARGET_TIME = pd.Timestamp(
    "2018-09-06 01:04:00"
)

WINDOW_SIZE = 120


# ============================================================
# PREPROCESSOR
# ============================================================

print("=" * 70)
print("EXACT LIVE WINDOW REPRODUCTION TEST")
print("=" * 70)

print("Device:", DEVICE)

dataset = ThreeWDataset(
    DATASET_ROOT
)

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

training_frames = []

for row in train_metadata.itertuples(index=False):

    df = pd.read_parquet(row.path)

    normal = df[
        df["class"] == 0
    ].copy()

    if not normal.empty:
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


# ============================================================
# LOAD MODEL
# ============================================================

detector = TranADDetector(
    model_path=MODEL_PATH,
    preprocessor=preprocessor,
    threshold=THRESHOLD,
    device=DEVICE,
)


# ============================================================
# LOAD EXACT FILE
# ============================================================

path = (
    DATASET_ROOT
    / "1"
    / "DRAWN_00001.parquet"
)

print("\n" + "=" * 70)
print("TARGET FILE")
print("=" * 70)

print(path)

df = pd.read_parquet(path)

print(
    "Rows:",
    len(df)
)

print(
    "Timeline:",
    df.index.min(),
    "→",
    df.index.max()
)


# ============================================================
# FIND TARGET TIMESTAMP
# ============================================================

target_position = df.index.get_indexer(
    [TARGET_TIME],
    method="nearest"
)[0]

actual_target_time = df.index[target_position]

print("\nTarget time:")
print(TARGET_TIME)

print("\nNearest dataset timestamp:")
print(actual_target_time)

print("\nRow index:")
print(target_position)


# ============================================================
# BUILD EXACT 120-SECOND WINDOW
# ============================================================

start = target_position - WINDOW_SIZE + 1
end = target_position + 1

window = df.iloc[
    start:end
].copy()

print("\n" + "=" * 70)
print("EXACT WINDOW")
print("=" * 70)

print(
    "Rows:",
    start,
    "→",
    end - 1
)

print(
    "Window size:",
    len(window)
)

print(
    "Timestamp:",
    window.index[0],
    "→",
    window.index[-1]
)


# ============================================================
# GROUND TRUTH
# ============================================================

classes = (
    window["class"]
    .value_counts(dropna=False)
    .to_dict()
)

print("\nClass distribution:")

print(classes)

print(
    "\nContains anomaly:",
    bool(
        (
            window["class"]
            .fillna(0)
            != 0
        ).any()
    )
)


# ============================================================
# SCORE
# ============================================================

features = (
    window[
        preprocessor.feature_columns
    ]
)

result = detector.detect(
    features
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("EXACT OFFLINE RESULT")
print("=" * 70)

print(
    f"Score:     {result.anomaly_score:.6f}"
)

print(
    f"Threshold: {THRESHOLD:.6f}"
)

print(
    "Prediction:",
    "ANOMALY"
    if result.is_anomaly
    else "NORMAL"
)

print(
    "Ground truth:",
    "ANOMALY"
    if (
        window["class"]
        .fillna(0)
        != 0
    ).any()
    else "NORMAL"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("LIVE vs OFFLINE")
print("=" * 70)

print(
    "Live score:    0.020082"
)

print(
    f"Offline score: {result.anomaly_score:.6f}"
)

print(
    f"Difference:    "
    f"{abs(result.anomaly_score - 0.020082):.6f}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)