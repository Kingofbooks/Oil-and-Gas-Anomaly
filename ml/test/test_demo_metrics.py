from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from ml.preprocessing import Preprocessor
from ml.tranad_detector import TranADDetector

# ============================================================
# CONFIG
# ============================================================
DATASET = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\demo\oil_gas_demo_22features.csv"
)

ARTIFACT_DIR = Path("artifacts")

MODEL_PATH = (
    ARTIFACT_DIR
    / "tranad_demo_v1.pt"
)

WINDOW_SIZE = 120

THRESHOLD = 0.005179412059486259


FEATURES = [
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
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

print_header("FINAL TRANAD DEMO METRICS TEST")

print(f"Dataset: {DATASET}")
print(f"Model:   {MODEL_PATH}")
print(f"Window:  {WINDOW_SIZE}")
print(f"Threshold: {THRESHOLD}")

df = pd.read_csv(DATASET)

print()
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

if "class" in df.columns and "ground_truth" not in df.columns:
    df["ground_truth"] = (pd.to_numeric(df["class"], errors="coerce").fillna(0) > 0).astype(int)

# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

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

print()
print("Ground truth distribution:")
print(df["ground_truth"].value_counts().sort_index())


# ============================================================
# CLEAN SENSOR DATA
# ============================================================

print_header("CLEANING DATA")

for feature in FEATURES:
    if feature not in df.columns:
        raise ValueError(f"Missing feature: {feature}")

    df[feature] = pd.to_numeric(df[feature], errors="coerce")

df[FEATURES] = (
    df[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .ffill()
    .bfill()
    .fillna(0.0)
)

print("Missing values after cleaning:")
print(df[FEATURES].isna().sum().sum())


# ============================================================
# FIT PREPROCESSOR
# ============================================================

print_header("FITTING PREPROCESSOR")

preprocessor = Preprocessor(
    feature_columns=FEATURES
)

preprocessor.fit(df[FEATURES])

print(f"Features: {len(preprocessor.feature_columns)}")


# ============================================================
# LOAD MODEL
# ============================================================

print_header("LOADING TRANAD")

detector = TranADDetector(
    model_path=MODEL_PATH,
    preprocessor=preprocessor,
    threshold=THRESHOLD,
    device="cpu",
)

print("Model loaded successfully.")


# ============================================================
# SLIDING WINDOW EVALUATION
# ============================================================

print_header("RUNNING SLIDING WINDOW EVALUATION")

y_true = []
y_pred = []
scores = []

timestamps = []

window_records = []

total_windows = len(df) - WINDOW_SIZE + 1

print(f"Total windows: {total_windows}")

for start in range(total_windows):

    end = start + WINDOW_SIZE

    window = df.iloc[start:end].copy()

    # --------------------------------------------------------
    # Ground truth for this window
    #
    # If ANY anomalous row exists inside the 120-row window,
    # classify the entire window as anomalous.
    # --------------------------------------------------------

    truth = int(window["ground_truth"].max())

    result = detector.detect(window[FEATURES])

    prediction = int(result.is_anomaly)
    score = float(result.anomaly_score)

    y_true.append(truth)
    y_pred.append(prediction)
    scores.append(score)

    timestamps.append(window.iloc[-1]["timestamp"]
                      if "timestamp" in window.columns
                      else end - 1)

    window_records.append(
        {
            "start": start,
            "end": end - 1,
            "truth": truth,
            "prediction": prediction,
            "score": score,
        }
    )

    if (start + 1) % 100 == 0:
        print(
            f"Processed {start + 1}/{total_windows}"
        )


# ============================================================
# METRICS
# ============================================================

y_true = np.array(y_true)
y_pred = np.array(y_pred)
scores = np.array(scores)

print_header("FINAL METRICS")

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")

print()
print("CONFUSION MATRIX")
print("----------------")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")


# ============================================================
# ROC-AUC
# ============================================================

print_header("ROC-AUC")

if len(np.unique(y_true)) == 2:

    auc = roc_auc_score(
        y_true,
        scores,
    )

    print(f"ROC-AUC: {auc:.4f}")

else:

    print("ROC-AUC cannot be calculated.")
    print("Only one class exists in the evaluated windows.")


# ============================================================
# FALSE ALARM WINDOWS
# ============================================================

print_header("FALSE ALARMS")

false_alarm_indices = np.where(
    (y_true == 0) &
    (y_pred == 1)
)[0]

print(
    f"False alarm windows: "
    f"{len(false_alarm_indices)}"
)

if len(false_alarm_indices) > 0:

    print()
    print("First false alarms:")

    for idx in false_alarm_indices[:10]:

        record = window_records[idx]

        print(
            f"Window {record['start']} → "
            f"{record['end']} | "
            f"Score={record['score']:.6f}"
        )

else:

    print("No false alarm windows.")


# ============================================================
# MISSED ANOMALIES
# ============================================================

print_header("MISSED ANOMALIES")

missed_indices = np.where(
    (y_true == 1) &
    (y_pred == 0)
)[0]

print(
    f"Missed anomaly windows: "
    f"{len(missed_indices)}"
)

if len(missed_indices) > 0:

    print()
    print("First missed anomalies:")

    for idx in missed_indices[:10]:

        record = window_records[idx]

        print(
            f"Window {record['start']} → "
            f"{record['end']} | "
            f"Score={record['score']:.6f}"
        )

else:

    print("No missed anomaly windows.")


# ============================================================
# EVENT DETECTION
# ============================================================

print_header("ANOMALY EVENT DETECTION")

# Find contiguous ground-truth anomaly regions
truth_regions = []

inside = False
start = None

for i, value in enumerate(df["ground_truth"].values):

    if value == 1 and not inside:

        inside = True
        start = i

    elif value == 0 and inside:

        truth_regions.append(
            (start, i - 1)
        )

        inside = False

if inside:
    truth_regions.append(
        (start, len(df) - 1)
    )


print(
    f"Ground-truth anomaly events: "
    f"{len(truth_regions)}"
)


# ------------------------------------------------------------
# Detection delay
# ------------------------------------------------------------

delays = []

for event_start, event_end in truth_regions:

    detected_windows = [
        record
        for record in window_records
        if (
            record["prediction"] == 1
            and record["end"] >= event_start
            and record["start"] <= event_end
        )
    ]

    if not detected_windows:

        print(
            f"Event {event_start} → {event_end}: "
            f"NOT DETECTED"
        )

        continue

    first_detection = detected_windows[0]

    detection_row = first_detection["end"]

    delay = max(
        0,
        detection_row - event_start
    )

    delays.append(delay)

    print(
        f"Event {event_start} → {event_end} | "
        f"Detected at row {detection_row} | "
        f"Delay: {delay} rows"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("FINAL VERDICT")

print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision:           {precision:.4f}")
print(f"Recall:              {recall:.4f}")
print(f"F1:                  {f1:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")

print()
print(f"TP: {tp}")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")

if delays:

    print()
    print(
        f"Average detection delay: "
        f"{np.mean(delays):.2f} rows"
    )

    print(
        f"Maximum detection delay: "
        f"{np.max(delays):.2f} rows"
    )

print()
print("=" * 80)
print("FINAL TEST COMPLETE")
print("=" * 80)