from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances
from ml.preprocessing import Preprocessor
from ml.windowing import TimeSeriesWindowGenerator
from ml.model import TranADNetwork
from ml.scoring import AnomalyScorer


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

MODEL_PATH = Path(
    "artifacts/tranad_v1.pt"
)

WINDOW_SIZE = 120
STRIDE = 60


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("ANOMALY THRESHOLD CALIBRATION")
print("=" * 70)

print("Device:", DEVICE)

print("\nBuilding dataset index...")


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


metadata_df = pd.DataFrame(
    records
)


# ============================================================
# 2. SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_metadata, validation_metadata, _ = (
    split_real_instances(
        metadata_df
    )
)


print(
    "\nValidation instances:",
    len(validation_metadata)
)

print(
    "Validation wells:",
    validation_metadata["well_id"].nunique()
)


# ============================================================
# 3. FIT PREPROCESSOR USING TRAINING DATA ONLY
# ============================================================

print(
    "\nPreparing training data for preprocessor..."
)


training_frames = []


for row in train_metadata.itertuples(
    index=False
):

    df = pd.read_parquet(
        row.path
    )


    # Model learns NORMAL behaviour
    normal = df[
        df["class"] == 0
    ].copy()


    if normal.empty:
        continue


    training_frames.append(
        normal
    )


if not training_frames:

    raise RuntimeError(
        "No normal training data found."
    )


training_data = pd.concat(
    training_frames,
    axis=0
)


preprocessor = Preprocessor()

preprocessor.fit(
    training_data
)


print(
    "Features:",
    len(
        preprocessor.feature_columns
    )
)

print(
    preprocessor.feature_columns
)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\nLoading model...")


model = TranADNetwork(
    input_size=len(
        preprocessor.feature_columns
    ),
    hidden_size=64,
    num_heads=4,
)


state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)


model.load_state_dict(
    state_dict
)

model.to(DEVICE)
model.eval()


print(
    "Loaded:",
    MODEL_PATH
)


# ============================================================
# 5. CREATE SCORER
# ============================================================

scorer = AnomalyScorer(
    model=model,
    device=DEVICE,
)


# ============================================================
# 6. WINDOW GENERATOR
# ============================================================

window_generator = (
    TimeSeriesWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
)


# ============================================================
# 7. SCORE VALIDATION WINDOWS
# ============================================================

normal_scores = []

anomaly_scores = []


print(
    "\nProcessing validation instances..."
)


for i, row in enumerate(
    validation_metadata.itertuples(
        index=False
    ),
    start=1,
):

    print(
        f"{i}/{len(validation_metadata)} "
        f"{Path(row.path).name}"
    )


    df = pd.read_parquet(
        row.path
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Keep the original timeline.
    #
    # We preprocess the complete instance first.
    # --------------------------------------------------------

    processed = (
        preprocessor.transform(
            df
        )
    )


    # --------------------------------------------------------
    # Create temporal windows
    # --------------------------------------------------------

    windows = (
        window_generator.create_windows(
            processed
        )
    )


    if len(windows) == 0:
        continue


    # --------------------------------------------------------
    # Label each window
    #
    # A window is anomalous if ANY timestamp in that
    # window has a non-zero class.
    # --------------------------------------------------------

    class_values = (
        df["class"]
        .fillna(0)
        .to_numpy()
    )


    labels = []


    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STRIDE,
    ):

        end = (
            start
            + WINDOW_SIZE
        )


        window_labels = (
            class_values[start:end]
        )


        is_anomaly = np.any(
            window_labels != 0
        )


        labels.append(
            int(is_anomaly)
        )


    labels = np.asarray(
        labels[:len(windows)]
    )


    # --------------------------------------------------------
    # Calculate reconstruction score
    # --------------------------------------------------------

    scores = scorer.score(
        windows
    )


    scores = np.asarray(
        scores
    )


    # --------------------------------------------------------
    # Separate normal and anomaly scores
    # --------------------------------------------------------

    normal_scores.extend(
        scores[labels == 0]
    )


    anomaly_scores.extend(
        scores[labels == 1]
    )


# ============================================================
# 8. CONVERT TO NUMPY
# ============================================================

normal_scores = np.asarray(
    normal_scores,
    dtype=np.float32,
)

anomaly_scores = np.asarray(
    anomaly_scores,
    dtype=np.float32,
)


if len(normal_scores) == 0:
    raise RuntimeError(
        "No normal validation windows found."
    )


if len(anomaly_scores) == 0:
    raise RuntimeError(
        "No anomaly validation windows found."
    )


# ============================================================
# 9. SCORE DISTRIBUTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "VALIDATION SCORE DISTRIBUTION"
)

print(
    "=" * 70
)


print(
    "\nNormal windows:",
    len(normal_scores)
)

print(
    "Anomaly windows:",
    len(anomaly_scores)
)


print("\nNORMAL")

print(
    "Min:",
    normal_scores.min()
)

print(
    "Max:",
    normal_scores.max()
)

print(
    "Mean:",
    normal_scores.mean()
)

print(
    "Median:",
    np.median(normal_scores)
)

print(
    "95th percentile:",
    np.percentile(
        normal_scores,
        95,
    )
)

print(
    "99th percentile:",
    np.percentile(
        normal_scores,
        99,
    )
)


print("\nANOMALY")

print(
    "Min:",
    anomaly_scores.min()
)

print(
    "Max:",
    anomaly_scores.max()
)

print(
    "Mean:",
    anomaly_scores.mean()
)

print(
    "Median:",
    np.median(anomaly_scores)
)

print(
    "5th percentile:",
    np.percentile(
        anomaly_scores,
        5,
    )
)


# ============================================================
# 10. COMBINE SCORES
# ============================================================

all_scores = np.concatenate(
    [
        normal_scores,
        anomaly_scores,
    ]
)


true_labels = np.concatenate(
    [
        np.zeros(
            len(normal_scores),
            dtype=int,
        ),
        np.ones(
            len(anomaly_scores),
            dtype=int,
        ),
    ]
)


# ============================================================
# 11. SEARCH FOR BEST THRESHOLD
# ============================================================

candidate_thresholds = np.percentile(
    all_scores,
    np.linspace(
        80,
        99.9,
        100,
    ),
)


best_threshold = None
best_f1 = -1.0
best_metrics = None


for threshold in candidate_thresholds:

    predictions = (
        all_scores >= threshold
    ).astype(int)


    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0,
    )


    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0,
    )


    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0,
    )


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = (
            float(threshold)
        )

        best_metrics = (
            precision,
            recall,
            f1,
        )


# ============================================================
# 12. FINAL METRICS
# ============================================================

precision, recall, f1 = (
    best_metrics
)


predictions = (
    all_scores >= best_threshold
).astype(int)


tn, fp, fn, tp = (
    confusion_matrix(
        true_labels,
        predictions,
    ).ravel()
)


false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)


# ============================================================
# 13. RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SELECTED THRESHOLD"
)

print(
    "=" * 70
)


print(
    "\nThreshold:",
    best_threshold
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1:",
    f1
)

print(
    "False Positive Rate:",
    false_positive_rate
)


print(
    "\nCONFUSION MATRIX"
)


print(
    "True Negatives:",
    tn
)

print(
    "False Positives:",
    fp
)

print(
    "False Negatives:",
    fn
)

print(
    "True Positives:",
    tp
)


print(
    "\n" + "=" * 70
)

print(
    "CALIBRATION COMPLETE"
)

print(
    "=" * 70
)