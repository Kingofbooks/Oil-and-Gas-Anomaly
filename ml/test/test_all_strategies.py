from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances
from ml.preprocessing import Preprocessor
from ml.model import TranADNetwork
from ml.scoring import AnomalyScorer


# ============================================================
# FINAL ALL-IN-ONE EXPERIMENT
# ============================================================
#
# This file compares the decision layer on the SAME V1 model:
#
# 1. Existing global threshold
# 2. Adaptive threshold from normal training-score percentiles
# 3. Score smoothing
# 4. Persistence rule
# 5. Adaptive + smoothing
# 6. Adaptive + smoothing + persistence
#
# It reports:
# - window precision / recall / F1 / FPR
# - event recall
# - false alarm rate on normal instances
# - detection delay
#
# IMPORTANT:
# Thresholds are calibrated ONLY from training-normal data.
# Validation is used to compare strategies.
# Do NOT use validation results to retrain the model.
#
# After this experiment, the best strategy should be frozen
# and evaluated ONCE on the untouched TEST split.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

MODEL_PATH = Path("artifacts/tranad_v1.pt")

WINDOW_SIZE = 120
STEP = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Existing calibrated threshold from calibrate.py
GLOBAL_THRESHOLD = 0.029617823120545266

# Training-normal quantiles for adaptive thresholds
QUANTILES = [
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
]

# Smoothing windows are measured in MODEL WINDOWS.
# 3 means approximately 3 * 60 seconds = 3 minutes
# because STEP=60 seconds.
SMOOTHING_WINDOWS = [3, 5]

# Number of consecutive positive windows required.
PERSISTENCE_VALUES = [2, 3]


# ============================================================
# HELPERS
# ============================================================

def calculate_window_metrics(labels, predictions):
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)

    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def rolling_mean(scores, window):
    scores = np.asarray(scores, dtype=float)

    if window <= 1:
        return scores.copy()

    result = np.empty_like(scores)

    for i in range(len(scores)):
        start = max(0, i - window + 1)
        result[i] = scores[start:i + 1].mean()

    return result


def apply_persistence(binary_predictions, required):
    """
    Require N consecutive positive windows.

    Example with required=3:

        0 1 1 1 1 0
          -> 0 0 1 1 0 0

    This reduces isolated false alarms.
    """

    binary_predictions = np.asarray(
        binary_predictions
    ).astype(int)

    result = np.zeros_like(binary_predictions)

    run = 0

    for i, value in enumerate(binary_predictions):

        if value == 1:
            run += 1
        else:
            run = 0

        if run >= required:
            result[i] = 1

    return result


def evaluate_instance(
    instance,
    scores,
    labels,
    timestamps,
    strategy_name,
    predictions,
):
    """
    Evaluate one validation instance.

    Event-level definition:

    - An instance is an anomaly event if ANY timestamp has
      class != 0.
    - An event is detected if ANY prediction is positive.
    - For anomalous instances, detection delay is measured from
      the first ground-truth anomaly timestamp to the first
      positive prediction timestamp.
    """

    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)

    has_anomaly = bool(np.any(labels == 1))
    detected = bool(np.any(predictions == 1))

    event_recall = (
        int(detected)
        if has_anomaly
        else None
    )

    delay_seconds = None

    if has_anomaly and detected:

        first_anomaly_idx = np.where(labels == 1)[0][0]
        first_prediction_idx = np.where(predictions == 1)[0][0]

        anomaly_time = pd.Timestamp(
            timestamps[first_anomaly_idx]
        )

        prediction_time = pd.Timestamp(
            timestamps[first_prediction_idx]
        )

        delay_seconds = (
            prediction_time - anomaly_time
        ).total_seconds()

    return {
        "strategy": strategy_name,
        "has_anomaly": has_anomaly,
        "detected": detected,
        "event_recall": event_recall,
        "delay_seconds": delay_seconds,
    }


def print_strategy_result(name, metrics, event_stats):
    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    print(
        f"Window Precision : {metrics['precision']:.4f}"
    )
    print(
        f"Window Recall    : {metrics['recall']:.4f}"
    )
    print(
        f"Window F1        : {metrics['f1']:.4f}"
    )
    print(
        f"Window FPR       : {metrics['fpr']:.4f}"
    )

    print(
        f"Anomaly events   : {event_stats['anomaly_events']}"
    )
    print(
        f"Detected events  : {event_stats['detected_events']}"
    )
    print(
        f"Event Recall     : {event_stats['event_recall']:.4f}"
    )
    print(
        f"Normal events    : {event_stats['normal_events']}"
    )
    print(
        f"False alarms     : {event_stats['false_alarm_events']}"
    )
    print(
        f"Event FPR        : {event_stats['event_fpr']:.4f}"
    )

    if event_stats["median_delay"] is None:
        print("Median Delay     : N/A")
    else:
        print(
            f"Median Delay     : "
            f"{event_stats['median_delay']:.1f} sec"
        )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FINAL TRANAD DECISION-LAYER EXPERIMENT")
print("=" * 80)

print("Device:", DEVICE)
print("Model:", MODEL_PATH)
print("Window:", WINDOW_SIZE)
print("Step:", STEP)

print("\nThis is the FINAL validation experiment.")
print("We will not train another model here.")


# ============================================================
# DATASET + SAME SPLIT
# ============================================================

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

train_metadata, validation_metadata, test_metadata = (
    split_real_instances(metadata_df)
)

print("\nDataset split:")
print("Train:", len(train_metadata))
print("Validation:", len(validation_metadata))
print("Test:", len(test_metadata))


# ============================================================
# FIT PREPROCESSOR ON TRAINING NORMAL DATA ONLY
# ============================================================

print("\nCollecting NORMAL training data...")

training_frames = []

for row in train_metadata.itertuples(index=False):

    df = pd.read_parquet(row.path)

    if "class" not in df.columns:
        continue

    normal = df[df["class"] == 0].copy()

    if len(normal) > 0:
        training_frames.append(normal)


if not training_frames:
    raise RuntimeError(
        "No normal training data found."
    )


training_data = pd.concat(
    training_frames,
    axis=0,
    ignore_index=True,
)

print(
    "Training normal rows:",
    f"{len(training_data):,}",
)

print("\nFitting preprocessor...")

preprocessor = Preprocessor()

preprocessor.fit(training_data)

print(
    "Features:",
    len(preprocessor.feature_columns),
)


# ============================================================
# LOAD V1
# ============================================================

print("\nLoading TranAD V1...")

model = TranADNetwork(
    input_size=len(preprocessor.feature_columns),
    hidden_size=64,
    num_heads=4,
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

scorer = AnomalyScorer(
    model=model,
    device=DEVICE,
)

print("Model loaded.")


# ============================================================
# CALIBRATE ADAPTIVE THRESHOLDS FROM NORMAL TRAINING
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATING ADAPTIVE THRESHOLDS")
print("=" * 80)

normal_training_scores = []

normal_window_count = 0

for i, row in enumerate(
    train_metadata.itertuples(index=False),
    start=1,
):

    df = pd.read_parquet(row.path)

    if "class" not in df.columns:
        continue

    normal = df[df["class"] == 0].copy()

    if len(normal) < WINDOW_SIZE:
        continue

    features = normal[
        preprocessor.feature_columns
    ].copy()

    processed = preprocessor.transform(
        features
    )

    values = processed.to_numpy(
        dtype=np.float32
    )

    windows = []

    for start in range(
        0,
        len(values) - WINDOW_SIZE + 1,
        STEP,
    ):

        window = values[
            start:start + WINDOW_SIZE
        ]

        windows.append(window)

    if not windows:
        continue

    windows = np.asarray(
        windows,
        dtype=np.float32,
    )

    scores = scorer.score(windows)

    normal_training_scores.extend(scores)

    normal_window_count += len(scores)

    if i % 25 == 0:
        print(
            f"Calibration files: "
            f"{i}/{len(train_metadata)}"
        )


normal_training_scores = np.asarray(
    normal_training_scores,
    dtype=np.float32,
)

print(
    "\nNormal calibration windows:",
    f"{len(normal_training_scores):,}",
)

print(
    "Normal median:",
    float(np.median(normal_training_scores)),
)

print(
    "Normal 95%:",
    float(np.quantile(normal_training_scores, 0.95)),
)

print(
    "Normal 99%:",
    float(np.quantile(normal_training_scores, 0.99)),
)

print(
    "Normal 99.5%:",
    float(np.quantile(normal_training_scores, 0.995)),
)


adaptive_thresholds = {
    q: float(
        np.quantile(
            normal_training_scores,
            q,
        )
    )
    for q in QUANTILES
}

print("\nAdaptive thresholds:")

for q, threshold in adaptive_thresholds.items():
    print(
        f"{q * 100:5.1f}% -> "
        f"{threshold:.8f}"
    )

print(
    f"\nExisting global threshold -> "
    f"{GLOBAL_THRESHOLD:.8f}"
)


# ============================================================
# SCORE VALIDATION INSTANCE-BY-INSTANCE
# ============================================================

print("\n" + "=" * 80)
print("SCORING VALIDATION SET")
print("=" * 80)

validation_instances = []

for i, row in enumerate(
    validation_metadata.itertuples(index=False),
    start=1,
):

    path = Path(row.path)

    try:
        df = pd.read_parquet(path)

        if "class" not in df.columns:
            continue

        if len(df) < WINDOW_SIZE:
            continue

        features = df[
            preprocessor.feature_columns
        ].copy()

        processed = preprocessor.transform(
            features
        )

        values = processed.to_numpy(
            dtype=np.float32
        )

        windows = []
        labels = []
        timestamps = []

        for start in range(
            0,
            len(values) - WINDOW_SIZE + 1,
            STEP,
        ):

            end = start + WINDOW_SIZE

            windows.append(
                values[start:end]
            )

            window_classes = (
                df["class"]
                .iloc[start:end]
                .fillna(0)
                .to_numpy()
            )

            labels.append(
                int(
                    np.any(
                        window_classes != 0
                    )
                )
            )

            # Use the END timestamp of the model window.
            timestamps.append(
                df.index[end - 1]
            )

        if not windows:
            continue

        windows = np.asarray(
            windows,
            dtype=np.float32,
        )

        scores = scorer.score(windows)

        validation_instances.append(
            {
                "name": path.name,
                "path": path,
                "scores": np.asarray(
                    scores,
                    dtype=np.float32,
                ),
                "labels": np.asarray(
                    labels,
                    dtype=int,
                ),
                "timestamps": np.asarray(
                    timestamps,
                ),
            }
        )

        if i % 10 == 0:
            print(
                f"Processed "
                f"{i}/{len(validation_metadata)}"
            )

    except Exception as e:
        print(
            f"Skipping {path.name}: {e}"
        )


print(
    "\nValidation files successfully scored:",
    len(validation_instances),
)


# ============================================================
# STRATEGY GENERATION
# ============================================================

strategies = {}


# ------------------------------------------------------------
# 1. CURRENT GLOBAL
# ------------------------------------------------------------

strategies["GLOBAL"] = {
    "threshold": GLOBAL_THRESHOLD,
    "smoothing": 1,
    "persistence": 1,
}


# ------------------------------------------------------------
# 2. ADAPTIVE THRESHOLDS
# ------------------------------------------------------------

for q, threshold in adaptive_thresholds.items():

    strategies[
        f"ADAPTIVE_{q * 100:.1f}%"
    ] = {
        "threshold": threshold,
        "smoothing": 1,
        "persistence": 1,
    }


# ------------------------------------------------------------
# 3. GLOBAL + SMOOTHING
# ------------------------------------------------------------

for smoothing in SMOOTHING_WINDOWS:

    strategies[
        f"GLOBAL_SMOOTH_{smoothing}"
    ] = {
        "threshold": GLOBAL_THRESHOLD,
        "smoothing": smoothing,
        "persistence": 1,
    }


# ------------------------------------------------------------
# 4. GLOBAL + PERSISTENCE
# ------------------------------------------------------------

for persistence in PERSISTENCE_VALUES:

    strategies[
        f"GLOBAL_PERSIST_{persistence}"
    ] = {
        "threshold": GLOBAL_THRESHOLD,
        "smoothing": 1,
        "persistence": persistence,
    }


# ------------------------------------------------------------
# 5. ADAPTIVE + SMOOTHING
# ------------------------------------------------------------

for q, threshold in adaptive_thresholds.items():

    for smoothing in SMOOTHING_WINDOWS:

        strategies[
            f"ADAPTIVE_{q * 100:.1f}_SMOOTH_{smoothing}"
        ] = {
            "threshold": threshold,
            "smoothing": smoothing,
            "persistence": 1,
        }


# ------------------------------------------------------------
# 6. ADAPTIVE + SMOOTHING + PERSISTENCE
# ------------------------------------------------------------

for q, threshold in adaptive_thresholds.items():

    for smoothing in SMOOTHING_WINDOWS:

        for persistence in PERSISTENCE_VALUES:

            strategies[
                (
                    f"ADAPTIVE_{q * 100:.1f}"
                    f"_SMOOTH_{smoothing}"
                    f"_PERSIST_{persistence}"
                )
            ] = {
                "threshold": threshold,
                "smoothing": smoothing,
                "persistence": persistence,
            }


# ============================================================
# EVALUATE ALL STRATEGIES
# ============================================================

print("\n" + "=" * 80)
print("RUNNING ALL STRATEGIES")
print("=" * 80)

summary_rows = []

for strategy_name, config in strategies.items():

    all_labels = []
    all_predictions = []

    event_results = []

    for instance in validation_instances:

        raw_scores = instance["scores"]

        threshold = config["threshold"]
        smoothing = config["smoothing"]
        persistence = config["persistence"]

        scores_for_decision = rolling_mean(
            raw_scores,
            smoothing,
        )

        raw_predictions = (
            scores_for_decision >= threshold
        ).astype(int)

        predictions = apply_persistence(
            raw_predictions,
            persistence,
        )

        all_labels.extend(
            instance["labels"]
        )

        all_predictions.extend(
            predictions
        )

        event_results.append(
            evaluate_instance(
                instance=instance,
                scores=raw_scores,
                labels=instance["labels"],
                timestamps=instance["timestamps"],
                strategy_name=strategy_name,
                predictions=predictions,
            )
        )

    metrics = calculate_window_metrics(
        all_labels,
        all_predictions,
    )

    anomaly_events = [
        x
        for x in event_results
        if x["has_anomaly"]
    ]

    normal_events = [
        x
        for x in event_results
        if not x["has_anomaly"]
    ]

    detected_anomalies = [
        x
        for x in anomaly_events
        if x["detected"]
    ]

    false_alarm_events = [
        x
        for x in normal_events
        if x["detected"]
    ]

    delays = [
        x["delay_seconds"]
        for x in detected_anomalies
        if x["delay_seconds"] is not None
    ]

    event_recall = (
        len(detected_anomalies)
        / len(anomaly_events)
        if anomaly_events
        else 0.0
    )

    event_fpr = (
        len(false_alarm_events)
        / len(normal_events)
        if normal_events
        else 0.0
    )

    median_delay = (
        float(np.median(delays))
        if delays
        else None
    )

    summary_rows.append(
        {
            "Strategy": strategy_name,
            "Threshold": config["threshold"],
            "Smooth": config["smoothing"],
            "Persist": config["persistence"],
            "WindowPrecision": metrics["precision"],
            "WindowRecall": metrics["recall"],
            "WindowF1": metrics["f1"],
            "WindowFPR": metrics["fpr"],
            "EventRecall": event_recall,
            "EventFPR": event_fpr,
            "MedianDelaySec": median_delay,
            "AnomalyEvents": len(anomaly_events),
            "DetectedEvents": len(detected_anomalies),
            "FalseAlarmEvents": len(false_alarm_events),
        }
    )


# ============================================================
# RESULTS
# ============================================================

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    by=[
        "EventRecall",
        "EventFPR",
        "WindowF1",
    ],
    ascending=[
        False,
        True,
        False,
    ],
)

print("\n" + "=" * 80)
print("FINAL STRATEGY RANKING")
print("=" * 80)

display_columns = [
    "Strategy",
    "Threshold",
    "Smooth",
    "Persist",
    "WindowPrecision",
    "WindowRecall",
    "WindowF1",
    "WindowFPR",
    "EventRecall",
    "EventFPR",
    "MedianDelaySec",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)


# ============================================================
# BEST CANDIDATES
# ============================================================

print("\n" + "=" * 80)
print("BEST CANDIDATES")
print("=" * 80)

print("\nBest Event Recall:")
print(
    summary.sort_values(
        ["EventRecall", "EventFPR"],
        ascending=[False, True],
    ).head(5)[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

print("\nBest Window F1:")
print(
    summary.sort_values(
        ["WindowF1", "WindowFPR"],
        ascending=[False, True],
    ).head(5)[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

print("\nBest low-false-alarm candidates (Event FPR <= 10%):")

low_fpr = summary[
    summary["EventFPR"] <= 0.10
]

if len(low_fpr) == 0:
    print("No strategy achieved Event FPR <= 10%.")

else:
    print(
        low_fpr.sort_values(
            ["EventRecall", "WindowF1"],
            ascending=[False, False],
        ).head(10)[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = Path(
    "artifacts/final_strategy_comparison.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    output_path,
    index=False,
)

print(
    f"\nFull results saved to: {output_path}"
)


# ============================================================
# FINAL DECISION RULE
# ============================================================

print("\n" + "=" * 80)
print("DECISION")
print("=" * 80)

print(
    """
Do NOT automatically choose the first row.

Use this order:

1. Prefer high EVENT RECALL.
2. Among similar event recall, prefer lower EVENT FPR.
3. Then prefer higher WINDOW F1.
4. Then prefer lower detection delay.

The validation result is used ONLY to choose the strategy.

Once chosen:
    validation -> freeze strategy
    test       -> final unbiased evaluation
    MQTT       -> live inference
"""
)

print("\n" + "=" * 80)
print("FINAL EXPERIMENT COMPLETE")
print("=" * 80)
