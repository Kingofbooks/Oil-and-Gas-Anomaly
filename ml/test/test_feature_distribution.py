from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from ml.preprocessing import Preprocessor
from ml.tranad_detector import TranADDetector 

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas"
    r"\datasets\demo\oil_gas_demo_22features.csv"
)

MODEL_PATH = Path("artifacts/tranad_demo_v1.pt")

WINDOW_SIZE = 120

# Current threshold
CURRENT_THRESHOLD = 0.005179412059486259

# We care about high recall for anomaly detection.
MIN_RECALL = 0.90


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "ABER-CKGL",
    "ABER-CKP",
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "P-ANULAR",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKP",
    "P-PDG",
    "P-TPT",
    "QGL",
    "T-JUS-CKP",
    "T-MON-CKP",
    "T-PDG",
    "T-TPT",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TRANAD THRESHOLD SWEEP")
print("=" * 80)

print(f"Dataset: {DATASET_PATH}")
print(f"Model:   {MODEL_PATH}")
print(f"Window:  {WINDOW_SIZE}")
print()

df = pd.read_csv(DATASET_PATH)

print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CLEAN DATA
# ============================================================

print()
print("=" * 80)
print("CLEANING DATA")
print("=" * 80)

df["timestamp"] = pd.to_datetime(df["timestamp"])

for feature in FEATURE_COLUMNS:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .replace([np.inf, -np.inf], np.nan)
    .ffill()
    .bfill()
)

df["ground_truth"] = (
    pd.to_numeric(
        df["class"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
    .ne(0)
    .astype(int)
)

print(
    "Missing feature values:",
    int(df[FEATURE_COLUMNS].isna().sum().sum()),
)

print(
    "Ground truth:",
    df["ground_truth"].value_counts().sort_index().to_dict(),
)


# ============================================================
# PREPROCESSOR
# ============================================================

print()
print("=" * 80)
print("FITTING PREPROCESSOR")
print("=" * 80)

normal_data = df.loc[
    df["ground_truth"] == 0,
    FEATURE_COLUMNS,
].copy()

preprocessor = Preprocessor(
    feature_columns=FEATURE_COLUMNS
)

preprocessor.fit(normal_data)

print(f"Features: {len(preprocessor.feature_columns)}")


# ============================================================
# DETECTOR
# ============================================================

print()
print("=" * 80)
print("LOADING TRANAD")
print("=" * 80)

detector = TranADDetector(
    model_path=MODEL_PATH,
    preprocessor=preprocessor,
    threshold=CURRENT_THRESHOLD,
    device="cpu",
)

print("Model loaded successfully.")


# ============================================================
# GENERATE WINDOW SCORES
# ============================================================

print()
print("=" * 80)
print("GENERATING WINDOW SCORES")
print("=" * 80)

total_windows = len(df) - WINDOW_SIZE + 1

print(f"Total windows: {total_windows}")
print()

scores = []
labels = []
window_starts = []
window_ends = []
window_timestamps = []


for start in range(total_windows):

    end = start + WINDOW_SIZE

    window = df.iloc[start:end].copy()

    result = detector.detect(
        window[FEATURE_COLUMNS]
    )

    score = float(result.anomaly_score)

    # A window is considered anomalous if ANY
    # row inside the window is anomalous.
    ground_truth = int(
        window["ground_truth"].max()
    )

    scores.append(score)
    labels.append(ground_truth)

    window_starts.append(start)
    window_ends.append(end - 1)

    window_timestamps.append(
        window.iloc[-1]["timestamp"]
    )

    if (start + 1) % 1000 == 0:

        print(
            f"Processed "
            f"{start + 1}/{total_windows}"
        )


scores = np.asarray(scores)
labels = np.asarray(labels)

window_starts = np.asarray(window_starts)
window_ends = np.asarray(window_ends)

print()
print(
    f"Scores generated: {len(scores)}"
)


# ============================================================
# ROC-AUC
# ============================================================

print()
print("=" * 80)
print("ROC-AUC")
print("=" * 80)

roc_auc = roc_auc_score(
    labels,
    scores,
)

print(
    f"ROC-AUC: {roc_auc:.6f}"
)


# ============================================================
# THRESHOLD LIST
# ============================================================

# We deliberately test a wide range.
#
# The exact values don't matter beforehand.
# The goal is to observe the tradeoff.

thresholds = sorted(
    set(
        [
            0.001,
            0.0015,
            0.002,
            0.0025,
            0.003,
            0.0035,
            0.004,
            0.0045,
            0.005,
            0.005179412059486259,
            0.0055,
            0.006,
            0.007,
            0.008,
            0.009,
            0.010,
            0.012,
            0.015,
            0.020,
            0.025,
            0.030,
            0.040,
            0.050,
            0.075,
            0.100,
            0.150,
            0.200,
            0.300,
            0.500,
            1.000,
        ]
    )
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

results = []


for threshold in thresholds:

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

    tn = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )

    tp = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    results.append(
        {
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
    )


results_df = pd.DataFrame(results)


# ============================================================
# FULL RESULTS
# ============================================================

print()
print("=" * 80)
print("THRESHOLD RESULTS")
print("=" * 80)

print(
    results_df[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "fpr",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]


print()
print("=" * 80)
print("BEST F1 THRESHOLD")
print("=" * 80)

print(
    f"Threshold : {best_f1['threshold']:.8f}"
)

print(
    f"Precision : {best_f1['precision']:.4f}"
)

print(
    f"Recall    : {best_f1['recall']:.4f}"
)

print(
    f"F1        : {best_f1['f1']:.4f}"
)

print(
    f"FPR       : {best_f1['fpr']:.4f}"
)

print(
    f"TP        : {int(best_f1['tp'])}"
)

print(
    f"TN        : {int(best_f1['tn'])}"
)

print(
    f"FP        : {int(best_f1['fp'])}"
)

print(
    f"FN        : {int(best_f1['fn'])}"
)


# ============================================================
# BEST HIGH-RECALL THRESHOLD
# ============================================================

high_recall = results_df[
    results_df["recall"] >= MIN_RECALL
].copy()


print()
print("=" * 80)
print(
    f"BEST THRESHOLD WITH RECALL >= {MIN_RECALL:.0%}"
)
print("=" * 80)


if len(high_recall) == 0:

    print(
        "No tested threshold reaches "
        f"{MIN_RECALL:.0%} recall."
    )

else:

    # Among thresholds maintaining high recall,
    # choose the one with highest F1.
    best_high_recall = high_recall.loc[
        high_recall["f1"].idxmax()
    ]

    print(
        f"Threshold : "
        f"{best_high_recall['threshold']:.8f}"
    )

    print(
        f"Precision : "
        f"{best_high_recall['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_high_recall['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_high_recall['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{best_high_recall['fpr']:.4f}"
    )

    print(
        f"TP        : "
        f"{int(best_high_recall['tp'])}"
    )

    print(
        f"TN        : "
        f"{int(best_high_recall['tn'])}"
    )

    print(
        f"FP        : "
        f"{int(best_high_recall['fp'])}"
    )

    print(
        f"FN        : "
        f"{int(best_high_recall['fn'])}"
    )


# ============================================================
# CURRENT THRESHOLD
# ============================================================

current = results_df.iloc[
    (
        results_df["threshold"]
        - CURRENT_THRESHOLD
    ).abs().argmin()
]


print()
print("=" * 80)
print("CURRENT THRESHOLD")
print("=" * 80)

print(
    f"Threshold : "
    f"{current['threshold']:.8f}"
)

print(
    f"Precision : "
    f"{current['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{current['recall']:.4f}"
)

print(
    f"F1        : "
    f"{current['f1']:.4f}"
)

print(
    f"FPR       : "
    f"{current['fpr']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = Path(
    "artifacts/threshold_sweep_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print()
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(
    f"Saved to: {output_path}"
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print()

print(
    "BEST F1 THRESHOLD:"
)

print(
    f"  {best_f1['threshold']:.8f}"
)

print()

if len(high_recall) > 0:

    print(
        "BEST HIGH-RECALL THRESHOLD:"
    )

    print(
        f"  {best_high_recall['threshold']:.8f}"
    )

    print()

    print(
        "For oil/gas anomaly detection, "
        "prefer the HIGH-RECALL threshold "
        "if its false-positive rate is acceptable."
    )

else:

    print(
        "No tested threshold satisfies "
        "the high-recall requirement."
    )

print()
print("=" * 80)
print("THRESHOLD SWEEP COMPLETE")
print("=" * 80)