from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
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

V1_MODEL = Path(
    "artifacts/tranad_v1.pt"
)

V2_MODEL = Path(
    "artifacts/tranad_v2.pt"
)

WINDOW_SIZE = 120
STRIDE = 60

HIDDEN_SIZE = 64
NUM_HEADS = 4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("TRANAD V1 vs V2 - FULL VALIDATION COMPARISON")
print("=" * 80)

print("Device:", DEVICE)


# ============================================================
# 1. BUILD DATASET INDEX
# ============================================================

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
# 2. SAME VALIDATION SPLIT
# ============================================================

_, validation_metadata, _ = (
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
# 3. FIT PREPROCESSOR ON TRAINING DATA ONLY
# ============================================================

print("\nCollecting training data for preprocessor...")


train_metadata, _, _ = (
    split_real_instances(
        metadata_df
    )
)


training_frames = []


for row in train_metadata.itertuples(
    index=False
):

    df = pd.read_parquet(
        row.path
    )


    normal = df[
        df["class"] == 0
    ].copy()


    if normal.empty:
        continue


    # Same preprocessing basis used by
    # the existing training pipeline.
    normal = normal.iloc[
        :2000
    ]


    training_frames.append(
        normal
    )


if not training_frames:

    raise RuntimeError(
        "No training data found."
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


# ============================================================
# 4. LOAD BOTH MODELS
# ============================================================

print("\nLoading models...")


def load_model(
    model_path: Path
):

    model = TranADNetwork(
        input_size=len(
            preprocessor.feature_columns
        ),
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
    )


    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )


    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()


    return model


v1_model = load_model(
    V1_MODEL
)

v2_model = load_model(
    V2_MODEL
)


v1_scorer = AnomalyScorer(
    model=v1_model,
    device=DEVICE
)

v2_scorer = AnomalyScorer(
    model=v2_model,
    device=DEVICE
)


print(
    "V1 loaded:",
    V1_MODEL
)

print(
    "V2 loaded:",
    V2_MODEL
)


# ============================================================
# 5. WINDOW GENERATOR
# ============================================================

window_generator = (
    TimeSeriesWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )
)


# ============================================================
# 6. COLLECT VALIDATION SCORES
# ============================================================

v1_scores = []
v2_scores = []
labels = []


normal_count = 0
anomaly_count = 0


print("\n" + "=" * 80)
print("PROCESSING VALIDATION DATA")
print("=" * 80)


for index, row in enumerate(
    validation_metadata.itertuples(
        index=False
    ),
    start=1
):

    df = pd.read_parquet(
        row.path
    )


    # --------------------------------------------------------
    # Keep entire timeline.
    #
    # IMPORTANT:
    # Do NOT remove anomaly rows.
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
    # Create window labels
    # --------------------------------------------------------

    class_values = (
        df["class"]
        .fillna(0)
        .to_numpy()
    )


    window_labels = []


    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STRIDE
    ):

        end = (
            start
            + WINDOW_SIZE
        )


        values = (
            class_values[
                start:end
            ]
        )


        is_anomaly = np.any(
            values != 0
        )


        window_labels.append(
            int(is_anomaly)
        )


    window_labels = np.asarray(
        window_labels[:len(windows)]
    )


    # --------------------------------------------------------
    # Score BOTH models
    # --------------------------------------------------------

    scores_v1 = v1_scorer.score(
        windows
    )

    scores_v2 = v2_scorer.score(
        windows
    )


    scores_v1 = np.asarray(
        scores_v1
    )

    scores_v2 = np.asarray(
        scores_v2
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    v1_scores.extend(
        scores_v1
    )

    v2_scores.extend(
        scores_v2
    )

    labels.extend(
        window_labels
    )


    normal_count += np.sum(
        window_labels == 0
    )

    anomaly_count += np.sum(
        window_labels == 1
    )


    if index % 10 == 0:

        print(
            f"Processed "
            f"{index}/"
            f"{len(validation_metadata)}"
        )


# ============================================================
# 7. NUMPY
# ============================================================

v1_scores = np.asarray(
    v1_scores,
    dtype=np.float64
)

v2_scores = np.asarray(
    v2_scores,
    dtype=np.float64
)

labels = np.asarray(
    labels,
    dtype=int
)


if len(labels) == 0:

    raise RuntimeError(
        "No validation windows were created."
    )


# ============================================================
# 8. BASIC DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION DATA")
print("=" * 80)


print(
    "Total windows:",
    len(labels)
)

print(
    "Normal windows:",
    normal_count
)

print(
    "Anomaly windows:",
    anomaly_count
)


# ============================================================
# 9. DISTRIBUTION FUNCTION
# ============================================================

def print_distribution(
    name,
    scores
):

    normal = scores[
        labels == 0
    ]

    anomaly = scores[
        labels == 1
    ]


    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)


    print(
        "\nNORMAL"
    )

    print(
        "Min:",
        normal.min()
    )

    print(
        "Median:",
        np.median(normal)
    )

    print(
        "Mean:",
        normal.mean()
    )

    print(
        "95th percentile:",
        np.percentile(
            normal,
            95
        )
    )

    print(
        "99th percentile:",
        np.percentile(
            normal,
            99
        )
    )


    print(
        "\nANOMALY"
    )

    print(
        "Min:",
        anomaly.min()
    )

    print(
        "Median:",
        np.median(anomaly)
    )

    print(
        "Mean:",
        anomaly.mean()
    )

    print(
        "5th percentile:",
        np.percentile(
            anomaly,
            5
        )
    )

    print(
        "Max:",
        anomaly.max()
    )


print_distribution(
    "TRANAD V1",
    v1_scores
)

print_distribution(
    "TRANAD V2",
    v2_scores
)


# ============================================================
# 10. ROC-AUC
# ============================================================

print("\n" + "=" * 80)
print("ROC-AUC")
print("=" * 80)


v1_auc = roc_auc_score(
    labels,
    v1_scores
)

v2_auc = roc_auc_score(
    labels,
    v2_scores
)


print(
    "V1 ROC-AUC:",
    v1_auc
)

print(
    "V2 ROC-AUC:",
    v2_auc
)


# ============================================================
# 11. THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    scores,
    labels
):

    # --------------------------------------------------------
    # Use score percentiles as candidate thresholds.
    # --------------------------------------------------------

    thresholds = np.unique(
        np.percentile(
            scores,
            np.linspace(
                80,
                99.9,
                200
            )
        )
    )


    best = None


    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)


        precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            labels,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )


        tn, fp, fn, tp = (
            confusion_matrix(
                labels,
                predictions,
                labels=[0, 1]
            ).ravel()
        )


        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )


        candidate = {
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


        if (
            best is None
            or f1 > best["f1"]
        ):

            best = candidate


    return best


# ============================================================
# 12. BEST V1
# ============================================================

v1_best = find_best_threshold(
    v1_scores,
    labels
)


# ============================================================
# 13. BEST V2
# ============================================================

v2_best = find_best_threshold(
    v2_scores,
    labels
)


# ============================================================
# 14. PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics
):

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)


    print(
        "Threshold:",
        metrics["threshold"]
    )

    print(
        "Precision:",
        metrics["precision"]
    )

    print(
        "Recall:",
        metrics["recall"]
    )

    print(
        "F1:",
        metrics["f1"]
    )

    print(
        "False Positive Rate:",
        metrics["fpr"]
    )

    print(
        "True Negatives:",
        metrics["tn"]
    )

    print(
        "False Positives:",
        metrics["fp"]
    )

    print(
        "False Negatives:",
        metrics["fn"]
    )

    print(
        "True Positives:",
        metrics["tp"]
    )


print(
    "\n" + "=" * 80
)

print(
    "BEST THRESHOLD RESULTS"
)

print(
    "=" * 80
)


print_metrics(
    "TRANAD V1",
    v1_best
)

print_metrics(
    "TRANAD V2",
    v2_best
)


# ============================================================
# 15. NORMAL / ANOMALY SEPARATION
# ============================================================

print("\n" + "=" * 80)
print("SEPARATION CHECK")
print("=" * 80)


v1_normal_median = np.median(
    v1_scores[labels == 0]
)

v1_anomaly_median = np.median(
    v1_scores[labels == 1]
)


v2_normal_median = np.median(
    v2_scores[labels == 0]
)

v2_anomaly_median = np.median(
    v2_scores[labels == 1]
)


print(
    "\nV1 anomaly/normal median ratio:",
    v1_anomaly_median
    / v1_normal_median
    if v1_normal_median != 0
    else float("inf")
)


print(
    "V2 anomaly/normal median ratio:",
    v2_anomaly_median
    / v2_normal_median
    if v2_normal_median != 0
    else float("inf")
)


# ============================================================
# 16. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("FINAL V1 vs V2")
print("=" * 80)


print(
    "\n                    V1              V2"
)

print(
    "ROC-AUC:       "
    f"{v1_auc:.6f}      "
    f"{v2_auc:.6f}"
)

print(
    "Precision:     "
    f"{v1_best['precision']:.6f}      "
    f"{v2_best['precision']:.6f}"
)

print(
    "Recall:        "
    f"{v1_best['recall']:.6f}      "
    f"{v2_best['recall']:.6f}"
)

print(
    "F1:            "
    f"{v1_best['f1']:.6f}      "
    f"{v2_best['f1']:.6f}"
)

print(
    "FPR:           "
    f"{v1_best['fpr']:.6f}      "
    f"{v2_best['fpr']:.6f}"
)


# ============================================================
# 17. VERDICT
# ============================================================

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)


if v2_auc > v1_auc:

    print(
        "\nV2 has better ROC-AUC."
    )

else:

    print(
        "\nV1 has better ROC-AUC."
    )


if v2_best["f1"] > v1_best["f1"]:

    print(
        "V2 has better F1."
    )

else:

    print(
        "V1 has better F1."
    )


if v2_best["recall"] > v1_best["recall"]:

    print(
        "V2 detects more anomalies."
    )

else:

    print(
        "V1 detects more anomalies."
    )


print(
    "\nThis experiment compares the models "
    "on the SAME validation set."
)

print(
    "Thresholds are calibrated independently "
    "for each model."
)


print("\n" + "=" * 80)
print("FULL COMPARISON COMPLETE")
print("=" * 80)