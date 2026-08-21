from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ml.dataset import ThreeWDataset
from ml.preprocessing import Preprocessor
from ml.model import TranADNetwork
from ml.scoring import AnomalyScorer


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

MODEL_PATH = Path("artifacts/tranad_v1.pt")

WINDOW_SIZE = 120
STEP = 60

DEVICE = "cpu"

# Existing threshold
GLOBAL_THRESHOLD = 0.029617823120545266

# Quantiles for adaptive threshold
NORMAL_QUANTILES = [
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
]


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(y_true, scores, threshold):
    """
    Calculate binary classification metrics for one threshold.
    """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(name, metrics):

    print(f"\n{name}")
    print("-" * 70)

    print(f"Threshold : {metrics['threshold']:.8f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"FPR       : {metrics['fpr']:.4f}")

    print()
    print(f"TP: {metrics['tp']}")
    print(f"TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("TRANAD THRESHOLD STRATEGY COMPARISON")
print("=" * 80)

print(f"Device: {DEVICE}")

print("\nBuilding dataset index...")

dataset = ThreeWDataset(DATASET_ROOT)

all_instances = dataset.build_index()

train_instances = [inst for inst in all_instances if inst.folder_type == 0]
validation_instances = [inst for inst in all_instances if inst.folder_type != 0]

print(f"Validation instances: {len(validation_instances)}")

# ============================================================
# COLLECT NORMAL TRAINING DATA
# ============================================================

print("\nCollecting normal training data for threshold calibration...")

normal_frames = []

for instance in train_instances:

    try:
        df = pd.read_parquet(instance.path)

        if "class" not in df.columns:
            continue

        normal = df[df["class"] == 0].copy()

        if len(normal) > 0:
            normal_frames.append(normal)

    except Exception as e:
        print(f"Skipping {instance.path.name}: {e}")


if not normal_frames:
    raise RuntimeError("No normal training data found.")


normal_df = pd.concat(normal_frames, ignore_index=True)

print(f"Normal training rows: {len(normal_df):,}")


# ============================================================
# FIT PREPROCESSOR
# ============================================================

print("\nFitting preprocessor...")

preprocessor = Preprocessor()

preprocessor.fit(normal_df)

print(f"Features: {len(preprocessor.feature_columns)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading TranAD...")

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
# SCORE NORMAL TRAINING WINDOWS
# ============================================================

print("\nScoring NORMAL training windows...")
print("These scores will be used to build adaptive thresholds.")


normal_scores = []

window_count = 0

for instance in train_instances:

    try:

        df = pd.read_parquet(instance.path)

        if "class" not in df.columns:
            continue

        df = df[df["class"] == 0].copy()

        if len(df) < WINDOW_SIZE:
            continue

        features = df[preprocessor.feature_columns].copy()

        processed = preprocessor.transform(features)

        values = processed.to_numpy(dtype=np.float32)

        for start in range(
            0,
            len(values) - WINDOW_SIZE + 1,
            STEP,
        ):

            window = values[start:start + WINDOW_SIZE]

            window = np.expand_dims(
                window,
                axis=0,
            )

            score = float(
                scorer.score(window)[0]
            )

            normal_scores.append(score)

            window_count += 1

            if window_count % 1000 == 0:
                print(
                    f"Normal windows scored: {window_count}"
                )

    except Exception as e:

        print(
            f"Skipping {instance.path.name}: {e}"
        )


normal_scores = np.asarray(normal_scores)

print()
print(f"Normal windows scored: {len(normal_scores):,}")

print()
print("NORMAL SCORE DISTRIBUTION")
print("-" * 70)

print(f"Min      : {normal_scores.min():.8f}")
print(f"Median   : {np.median(normal_scores):.8f}")
print(f"Mean     : {normal_scores.mean():.8f}")
print(f"95%      : {np.percentile(normal_scores, 95):.8f}")
print(f"99%      : {np.percentile(normal_scores, 99):.8f}")
print(f"99.5%    : {np.percentile(normal_scores, 99.5):.8f}")
print(f"Max      : {normal_scores.max():.8f}")


# ============================================================
# BUILD ADAPTIVE THRESHOLDS
# ============================================================

adaptive_thresholds = {}

for q in NORMAL_QUANTILES:

    threshold = float(
        np.quantile(normal_scores, q)
    )

    adaptive_thresholds[q] = threshold


print()
print("=" * 80)
print("ADAPTIVE THRESHOLDS")
print("=" * 80)

for q, threshold in adaptive_thresholds.items():

    print(
        f"Normal {q * 100:5.1f}% percentile "
        f"-> threshold = {threshold:.8f}"
    )


# ============================================================
# SCORE VALIDATION WINDOWS
# ============================================================

print()
print("=" * 80)
print("SCORING VALIDATION WINDOWS")
print("=" * 80)

validation_scores = []
validation_labels = []

processed_instances = 0

for instance in validation_instances:

    try:

        df = pd.read_parquet(instance.path)

        if "class" not in df.columns:
            continue

        if len(df) < WINDOW_SIZE:
            continue

        features = df[
            preprocessor.feature_columns
        ].copy()

        processed_features = preprocessor.transform(
            features
        )

        values = processed_features.to_numpy(
            dtype=np.float32
        )

        classes = df["class"].to_numpy()

        for start in range(
            0,
            len(values) - WINDOW_SIZE + 1,
            STEP,
        ):

            end = start + WINDOW_SIZE

            window = values[start:end]

            window = np.expand_dims(
                window,
                axis=0,
            )

            score = float(
                scorer.score(window)[0]
            )

            # ------------------------------------------------
            # Window is anomalous if ANY row is non-zero.
            # ------------------------------------------------

            window_classes = classes[start:end]

            is_anomaly = int(
                np.any(window_classes != 0)
            )

            validation_scores.append(score)
            validation_labels.append(is_anomaly)

        processed_instances += 1

        if processed_instances % 10 == 0:

            print(
                f"Processed "
                f"{processed_instances}/"
                f"{len(validation_instances)}"
            )

    except Exception as e:

        print(
            f"Skipping {instance.path.name}: {e}"
        )


validation_scores = np.asarray(
    validation_scores
)

validation_labels = np.asarray(
    validation_labels
)

print()
print(f"Validation windows: {len(validation_scores):,}")

print(
    f"Normal windows: "
    f"{(validation_labels == 0).sum():,}"
)

print(
    f"Anomaly windows: "
    f"{(validation_labels == 1).sum():,}"
)


# ============================================================
# GLOBAL THRESHOLD
# ============================================================

print()
print("=" * 80)
print("1. CURRENT GLOBAL THRESHOLD")
print("=" * 80)

global_result = calculate_metrics(
    validation_labels,
    validation_scores,
    GLOBAL_THRESHOLD,
)

print_metrics(
    "GLOBAL THRESHOLD",
    global_result,
)


# ============================================================
# ADAPTIVE THRESHOLD RESULTS
# ============================================================

print()
print("=" * 80)
print("2. ADAPTIVE THRESHOLD STRATEGIES")
print("=" * 80)

adaptive_results = []

for q, threshold in adaptive_thresholds.items():

    result = calculate_metrics(
        validation_labels,
        validation_scores,
        threshold,
    )

    adaptive_results.append(
        (q, result)
    )

    print()
    print(
        f"Adaptive threshold = "
        f"normal {q * 100:.1f}th percentile"
    )

    print_metrics(
        f"ADAPTIVE {q * 100:.1f}%",
        result,
    )


# ============================================================
# BEST F1
# ============================================================

print()
print("=" * 80)
print("3. BEST F1")
print("=" * 80)

best_f1 = max(
    adaptive_results,
    key=lambda x: x[1]["f1"],
)

best_q, best_result = best_f1

print(
    f"Best adaptive percentile: "
    f"{best_q * 100:.1f}%"
)

print_metrics(
    "BEST ADAPTIVE THRESHOLD",
    best_result,
)


# ============================================================
# BEST RECALL UNDER FPR CONSTRAINT
# ============================================================

print()
print("=" * 80)
print("4. BEST RECALL WITH FPR <= 10%")
print("=" * 80)

acceptable = [
    (q, result)
    for q, result in adaptive_results
    if result["fpr"] <= 0.10
]

if acceptable:

    best_recall = max(
        acceptable,
        key=lambda x: x[1]["recall"],
    )

    q, result = best_recall

    print(
        f"Best adaptive percentile: "
        f"{q * 100:.1f}%"
    )

    print_metrics(
        "BEST RECALL <= 10% FPR",
        result,
    )

else:

    print(
        "No tested adaptive threshold achieved "
        "FPR <= 10%."
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

print()
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

rows = []

rows.append(
    {
        "Strategy": "Global",
        "Percentile": "-",
        "Threshold": GLOBAL_THRESHOLD,
        "Precision": global_result["precision"],
        "Recall": global_result["recall"],
        "F1": global_result["f1"],
        "FPR": global_result["fpr"],
    }
)

for q, result in adaptive_results:

    rows.append(
        {
            "Strategy": "Adaptive",
            "Percentile": f"{q * 100:.1f}%",
            "Threshold": result["threshold"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
            "FPR": result["fpr"],
        }
    )


summary = pd.DataFrame(rows)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("THRESHOLD EXPERIMENT COMPLETE")
print("=" * 80)

print()
print("IMPORTANT:")
print(
    "These results DO NOT change the production threshold."
)

print(
    "We are only testing whether adaptive calibration "
    "is better than the current global threshold."
)

print()
print(
    "Next decision should be based on:"
)

print(
    "1. Recall"
)

print(
    "2. False-positive rate"
)

print(
    "3. F1"
)

print(
    "4. Event-level detection"
)

print(
    "5. Whether different anomaly classes need "
    "different thresholds"
)