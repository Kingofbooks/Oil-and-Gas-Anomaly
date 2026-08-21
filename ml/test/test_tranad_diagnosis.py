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

# Exact live event we already reproduced
DRAWN_FILE = (
    DATASET_ROOT
    / "1"
    / "DRAWN_00001.parquet"
)

LIVE_TIMESTAMP = pd.Timestamp(
    "2018-09-06 01:04:00"
)

LIVE_SCORE = 0.020082


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(
    scores,
    labels,
    threshold,
):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_metrics(result):
    print(
        f"{result['threshold']:<12.6f}"
        f"{result['precision']:<14.4f}"
        f"{result['recall']:<14.4f}"
        f"{result['f1']:<12.4f}"
        f"{result['fpr']:<12.4f}"
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("TRANAD COMPLETE DIAGNOSTIC TEST")
print("=" * 80)

print("Device:", DEVICE)

print("\nThis test will evaluate:")
print("1. Score distributions")
print("2. Threshold sweep")
print("3. Per-event-class detection")
print("4. Event-level detection")
print("5. Exact live-window reproduction")
print("6. Final diagnosis")


# ============================================================
# 1. BUILD DATASET INDEX
# ============================================================

print("\n" + "=" * 80)
print("1. BUILDING DATASET INDEX")
print("=" * 80)

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

train_metadata, validation_metadata, test_metadata = (
    split_real_instances(
        metadata_df
    )
)

print(
    "\nTrain:",
    len(train_metadata)
)

print(
    "Validation:",
    len(validation_metadata)
)

print(
    "Test:",
    len(test_metadata)
)

print(
    "Validation wells:",
    validation_metadata["well_id"].nunique()
)


# ============================================================
# 2. COLLECT NORMAL TRAINING DATA
# ============================================================

print("\n" + "=" * 80)
print("2. FIT PREPROCESSOR")
print("=" * 80)

training_frames = []

for i, row in enumerate(
    train_metadata.itertuples(index=False),
    start=1,
):

    df = pd.read_parquet(
        row.path
    )

    normal = df[
        df["class"] == 0
    ].copy()

    if not normal.empty:
        training_frames.append(
            normal
        )

    if i % 50 == 0:
        print(
            f"Processed {i}/{len(train_metadata)}"
        )


training_data = pd.concat(
    training_frames,
    axis=0,
)

print(
    "\nTraining rows:",
    len(training_data)
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
# 3. LOAD TRANAD
# ============================================================

print("\n" + "=" * 80)
print("3. LOAD TRANAD")
print("=" * 80)

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

scorer = AnomalyScorer(
    model=model,
    device=DEVICE,
)

print(
    "Model loaded:",
    MODEL_PATH
)


# ============================================================
# 4. PROCESS VALIDATION DATA
# ============================================================

print("\n" + "=" * 80)
print("4. PROCESS VALIDATION DATA")
print("=" * 80)

window_generator = (
    TimeSeriesWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
)

all_scores = []
all_labels = []

# For event-class analysis
class_scores = {}
class_window_counts = {}

# For event-level analysis
event_results = []

for i, row in enumerate(
    validation_metadata.itertuples(index=False),
    start=1,
):

    path = Path(row.path)

    df = pd.read_parquet(
        path
    )

    # --------------------------------------------------------
    # Original timeline is preserved
    # --------------------------------------------------------

    processed = (
        preprocessor.transform(
            df
        )
    )

    windows = (
        window_generator.create_windows(
            processed
        )
    )

    if len(windows) == 0:
        continue

    # --------------------------------------------------------
    # Window labels
    # --------------------------------------------------------

    class_values = (
        df["class"]
        .fillna(0)
        .to_numpy()
    )

    labels = []

    window_classes = []

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

        # Any non-zero class = anomaly
        is_anomaly = np.any(
            window_labels != 0
        )

        labels.append(
            int(is_anomaly)
        )

        # Find actual anomaly class
        anomaly_classes = (
            window_labels[
                window_labels != 0
            ]
        )

        if len(anomaly_classes) > 0:

            # Most common anomaly class
            values, counts = np.unique(
                anomaly_classes,
                return_counts=True,
            )

            dominant_class = int(
                values[
                    np.argmax(counts)
                ]
            )

        else:

            dominant_class = 0

        window_classes.append(
            dominant_class
        )

    labels = np.asarray(
        labels[:len(windows)]
    )

    window_classes = np.asarray(
        window_classes[
            :len(windows)
        ]
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    scores = scorer.score(
        windows
    )

    scores = np.asarray(
        scores
    )

    all_scores.extend(
        scores.tolist()
    )

    all_labels.extend(
        labels.tolist()
    )

    # --------------------------------------------------------
    # Per-event-class scores
    # --------------------------------------------------------

    for score, label, event_class in zip(
        scores,
        labels,
        window_classes,
    ):

        if label == 1:

            class_scores.setdefault(
                event_class,
                [],
            )

            class_scores[
                event_class
            ].append(
                float(score)
            )

            class_window_counts[
                event_class
            ] = (
                class_window_counts.get(
                    event_class,
                    0,
                )
                + 1
            )

    # --------------------------------------------------------
    # Event-level result
    #
    # This asks:
    #
    # "Did at least ONE window detect
    # this event?"
    # --------------------------------------------------------

    if np.any(labels == 1):

        anomaly_scores = scores[
            labels == 1
        ]

        event_results.append(
            {
                "file": path.name,
                "event_class": row.event_class,
                "max_score": float(
                    anomaly_scores.max()
                ),
            }
        )

    if i % 20 == 0:

        print(
            f"Processed "
            f"{i}/{len(validation_metadata)}"
        )


all_scores = np.asarray(
    all_scores
)

all_labels = np.asarray(
    all_labels
)


print(
    "\nValidation windows:",
    len(all_scores)
)

print(
    "Normal windows:",
    np.sum(all_labels == 0)
)

print(
    "Anomaly windows:",
    np.sum(all_labels == 1)
)


# ============================================================
# 5. SCORE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("5. SCORE DISTRIBUTIONS")
print("=" * 80)

normal_scores = (
    all_scores[
        all_labels == 0
    ]
)

anomaly_scores = (
    all_scores[
        all_labels == 1
    ]
)

print("\nNORMAL")

print(
    "Min:",
    normal_scores.min()
)

print(
    "Median:",
    np.median(
        normal_scores
    )
)

print(
    "95%:",
    np.percentile(
        normal_scores,
        95,
    )
)

print(
    "99%:",
    np.percentile(
        normal_scores,
        99,
    )
)

print(
    "Max:",
    normal_scores.max()
)


print("\nANOMALY")

print(
    "Min:",
    anomaly_scores.min()
)

print(
    "Median:",
    np.median(
        anomaly_scores
    )
)

print(
    "5%:",
    np.percentile(
        anomaly_scores,
        5,
    )
)

print(
    "25%:",
    np.percentile(
        anomaly_scores,
        25,
    )
)

print(
    "75%:",
    np.percentile(
        anomaly_scores,
        75,
    )
)

print(
    "Max:",
    anomaly_scores.max()
)


# ============================================================
# 6. THRESHOLD SWEEP
# ============================================================

print("\n" + "=" * 80)
print("6. THRESHOLD SWEEP")
print("=" * 80)

# We deliberately include thresholds
# around the live anomaly score.

thresholds = [
    0.100,
    0.075,
    0.050,
    0.040,
    0.030,
    0.0296178231,
    0.025,
    0.020,
    0.015,
    0.010,
    0.0075,
    0.005,
    0.0025,
    0.001,
]

print(
    "\n"
    f"{'Threshold':<12}"
    f"{'Precision':<14}"
    f"{'Recall':<14}"
    f"{'F1':<12}"
    f"{'FPR':<12}"
)

print("-" * 64)

threshold_results = []

for threshold in thresholds:

    result = calculate_metrics(
        all_scores,
        all_labels,
        threshold,
    )

    threshold_results.append(
        result
    )

    print_metrics(
        result
    )


# ============================================================
# 7. BEST THRESHOLDS
# ============================================================

print("\n" + "=" * 80)
print("7. BEST THRESHOLDS")
print("=" * 80)

best_f1 = max(
    threshold_results,
    key=lambda x: x["f1"],
)

best_recall = max(
    threshold_results,
    key=lambda x: x["recall"],
)

best_precision = max(
    threshold_results,
    key=lambda x: x["precision"],
)


print("\nBest F1:")

print_metrics(
    best_f1
)

print("\nBest Recall:")

print_metrics(
    best_recall
)

print("\nBest Precision:")

print_metrics(
    best_precision
)


# ============================================================
# 8. PER EVENT CLASS
# ============================================================

print("\n" + "=" * 80)
print("8. PER-EVENT-CLASS ANALYSIS")
print("=" * 80)

# Use the current threshold
CURRENT_THRESHOLD = 0.029617823120545266

print(
    "\n"
    f"{'Class':<10}"
    f"{'Windows':<12}"
    f"{'Median':<14}"
    f"{'Max':<14}"
    f"{'Detected':<12}"
    f"{'Recall':<10}"
)

print("-" * 72)

for event_class in sorted(
    class_scores.keys()
):

    scores = np.asarray(
        class_scores[event_class]
    )

    detected = np.sum(
        scores >= CURRENT_THRESHOLD
    )

    total = len(scores)

    recall = (
        detected / total
        if total > 0
        else 0
    )

    print(
        f"{event_class:<10}"
        f"{total:<12}"
        f"{np.median(scores):<14.6f}"
        f"{scores.max():<14.6f}"
        f"{detected:<12}"
        f"{recall:<10.4f}"
    )


# ============================================================
# 9. EVENT-LEVEL DETECTION
# ============================================================

print("\n" + "=" * 80)
print("9. EVENT-LEVEL DETECTION")
print("=" * 80)

print(
    "\nQuestion:"
)

print(
    "Did TranAD detect AT LEAST ONE window"
)

print(
    "inside an anomalous event?"
)

event_detected = 0
event_missed = 0

for event in event_results:

    detected = (
        event["max_score"]
        >= CURRENT_THRESHOLD
    )

    if detected:
        event_detected += 1
    else:
        event_missed += 1

    print(
        f"\n{event['file']}"
    )

    print(
        "Event class:",
        event["event_class"]
    )

    print(
        "Maximum score:",
        event["max_score"]
    )

    print(
        "Detected:",
        detected
    )


total_events = (
    event_detected
    + event_missed
)

event_recall = (
    event_detected / total_events
    if total_events > 0
    else 0
)

print("\nEVENT SUMMARY")

print(
    "Total events:",
    total_events
)

print(
    "Detected events:",
    event_detected
)

print(
    "Missed events:",
    event_missed
)

print(
    "Event recall:",
    event_recall
)


# ============================================================
# 10. EXACT LIVE WINDOW
# ============================================================

print("\n" + "=" * 80)
print("10. EXACT LIVE WINDOW")
print("=" * 80)

print(
    "File:",
    DRAWN_FILE
)

drawn_df = pd.read_parquet(
    DRAWN_FILE
)

target_position = (
    drawn_df.index.get_indexer(
        [LIVE_TIMESTAMP],
        method="nearest",
    )[0]
)

actual_timestamp = (
    drawn_df.index[
        target_position
    ]
)

start = (
    target_position
    - WINDOW_SIZE
    + 1
)

end = (
    target_position
    + 1
)

live_window = (
    drawn_df.iloc[
        start:end
    ]
)

live_features = (
    live_window[
        preprocessor.feature_columns
    ]
)

live_result = None

try:

    processed_live = (
        preprocessor.transform(
            live_features
        )
    )

    values = (
        processed_live
        .to_numpy(
            dtype=np.float32
        )
    )

    values = np.expand_dims(
        values,
        axis=0,
    )

    live_scores = scorer.score(
        values
    )

    live_score_offline = float(
        live_scores[0]
    )

    live_result = live_score_offline

except Exception as e:

    print(
        "Live window scoring failed:",
        e,
    )


print(
    "\nTimestamp:",
    live_window.index[0],
    "→",
    live_window.index[-1],
)

print(
    "Rows:",
    len(live_window)
)

print(
    "Ground truth classes:"
)

print(
    live_window["class"]
    .value_counts(
        dropna=False
    )
    .to_dict()
)

if live_result is not None:

    print(
        "\nOffline score:",
        live_result
    )

    print(
        "Previously observed live score:",
        LIVE_SCORE
    )

    print(
        "Difference:",
        abs(
            live_result
            - LIVE_SCORE
        )
    )

    print(
        "Current threshold:",
        CURRENT_THRESHOLD
    )

    print(
        "Current prediction:",
        "ANOMALY"
        if live_result
        >= CURRENT_THRESHOLD
        else "NORMAL"
    )


# ============================================================
# 11. SIMPLE DIAGNOSIS
# ============================================================

print("\n" + "=" * 80)
print("11. INITIAL DIAGNOSIS")
print("=" * 80)

print(
    "\nThis section is NOT automatically"
)

print(
    "changing the model."
)

print(
    "It only interprets the measurements."
)


live_detected = (
    live_result is not None
    and live_result
    >= CURRENT_THRESHOLD
)

lower_threshold_detects = (
    live_result is not None
    and live_result
    >= 0.020
)


print("\nLive anomaly:")

print(
    "Ground truth = ANOMALY"
)

print(
    "Current prediction =",
    "ANOMALY"
    if live_detected
    else "NORMAL",
)

print(
    "Live score =",
    live_result,
)

print(
    "Current threshold =",
    CURRENT_THRESHOLD,
)


print("\nInterpretation:")

if best_f1["f1"] > 0.5:

    print(
        "→ Threshold calibration may be a major issue."
    )

else:

    print(
        "→ Score separation appears weak."
    )


if best_recall["recall"] < 0.5:

    print(
        "→ Even threshold changes may not provide strong recall."
    )

else:

    print(
        "→ Lower thresholds can recover substantial recall."
    )


print(
    "\nIMPORTANT:"
)

print(
    "Do NOT retrain or change the live threshold"
)

print(
    "until we inspect these results."
)


print("\n" + "=" * 80)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 80)