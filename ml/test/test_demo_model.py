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

from ml.preprocessing import Preprocessor
from ml.windowing import TimeSeriesWindowGenerator
from ml.model import TranADNetwork
from ml.scoring import AnomalyScorer


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\demo\oil_gas_demo_22features.csv"
)

MODEL_PATH = Path(
    "artifacts/tranad_demo_v1.pt"
)

WINDOW_SIZE = 120
STRIDE = 30

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
# HELPERS
# ============================================================

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    for feature in FEATURE_COLUMNS:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    for feature in FEATURE_COLUMNS:

        if df[feature].isna().any():

            median = df[feature].median()

            df[feature] = (
                df[feature]
                .fillna(median)
            )

    return df


def evaluate_threshold(
    scores,
    labels,
    threshold,
):

    predictions = (
        np.asarray(scores)
        >= threshold
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
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
        fpr,
        tn,
        fp,
        fn,
        tp,
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("TRANAD DEMO MODEL - FULL EVALUATION")
print("=" * 80)

print("Device:", DEVICE)


# ============================================================
# 1. LOAD DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


df = pd.read_csv(
    DATA_PATH
)

print(
    "\nDataset rows:",
    len(df)
)


# ============================================================
# 2. PREPARE DATA
# ============================================================

df = prepare_dataframe(
    df
)


print(
    "Normal rows:",
    int((df["class"] == 0).sum())
)

print(
    "Anomaly rows:",
    int((df["class"] != 0).sum())
)


# ============================================================
# 3. FIT PREPROCESSOR ON NORMAL DATA ONLY
# ============================================================

print("\nFitting preprocessor...")

normal_df = df[
    df["class"] == 0
].copy()

preprocessor = Preprocessor()

preprocessor.fit(
    normal_df
)

print(
    "Features:",
    len(preprocessor.feature_columns)
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\nLoading TranAD model...")

model = TranADNetwork(
    input_size=22,
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
# 5. WINDOW GENERATION
# ============================================================

window_generator = (
    TimeSeriesWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
)


processed = preprocessor.transform(
    df
)

windows = (
    window_generator.create_windows(
        processed
    )
)


print("\nTotal windows:", len(windows))


# ============================================================
# 6. CREATE WINDOW LABELS
# ============================================================

labels = []
timestamps = []
window_ranges = []


for start in range(
    0,
    len(df) - WINDOW_SIZE + 1,
    STRIDE,
):

    end = start + WINDOW_SIZE

    classes = (
        df["class"]
        .iloc[start:end]
        .to_numpy()
    )

    # Window is anomalous if ANY
    # timestamp contains an anomaly.

    label = int(
        np.any(classes != 0)
    )

    labels.append(
        label
    )

    if "timestamp" in df.columns:

        timestamps.append(
            df["timestamp"]
            .iloc[end - 1]
        )

    else:

        timestamps.append(
            end - 1
        )

    window_ranges.append(
        (
            start,
            end - 1,
        )
    )


labels = np.asarray(
    labels[:len(windows)]
)

timestamps = timestamps[
    :len(windows)
]

window_ranges = window_ranges[
    :len(windows)
]


print(
    "Normal windows:",
    int((labels == 0).sum())
)

print(
    "Anomaly windows:",
    int((labels == 1).sum())
)


# ============================================================
# 7. SCORE WINDOWS
# ============================================================

print("\nScoring windows...")

scores = []

BATCH_SIZE = 128


for start in range(
    0,
    len(windows),
    BATCH_SIZE,
):

    batch = windows[
        start:start + BATCH_SIZE
    ]

    batch_scores = scorer.score(
        batch
    )

    scores.extend(
        np.asarray(
            batch_scores
        ).tolist()
    )


scores = np.asarray(
    scores,
    dtype=np.float64,
)


print(
    "Scores generated:",
    len(scores)
)


# ============================================================
# 8. SCORE DISTRIBUTION
# ============================================================

normal_scores = scores[
    labels == 0
]

anomaly_scores = scores[
    labels == 1
]


print("\n" + "=" * 80)
print("SCORE DISTRIBUTION")
print("=" * 80)


print("\nNORMAL WINDOWS")

print(
    "Count:",
    len(normal_scores)
)

print(
    "Min:",
    normal_scores.min()
)

print(
    "Median:",
    np.median(normal_scores)
)

print(
    "Mean:",
    normal_scores.mean()
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

print(
    "Max:",
    normal_scores.max()
)


print("\nANOMALY WINDOWS")

print(
    "Count:",
    len(anomaly_scores)
)

print(
    "Min:",
    anomaly_scores.min()
)

print(
    "Median:",
    np.median(anomaly_scores)
)

print(
    "Mean:",
    anomaly_scores.mean()
)

print(
    "5th percentile:",
    np.percentile(
        anomaly_scores,
        5,
    )
)

print(
    "Max:",
    anomaly_scores.max()
)


# ============================================================
# 9. ROC-AUC
# ============================================================

print("\n" + "=" * 80)
print("ROC-AUC")
print("=" * 80)

if (
    len(np.unique(labels))
    == 2
):

    auc = roc_auc_score(
        labels,
        scores,
    )

    print(
        "ROC-AUC:",
        auc
    )

else:

    auc = None

    print(
        "ROC-AUC unavailable: "
        "only one class present."
    )


# ============================================================
# 10. THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD SEARCH")
print("=" * 80)


# Candidate thresholds from
# observed score distribution.

candidate_thresholds = np.unique(
    np.percentile(
        scores,
        np.linspace(
            80,
            99.9,
            200,
        ),
    )
)


best_threshold = None
best_f1 = -1
best_result = None


for threshold in candidate_thresholds:

    result = evaluate_threshold(
        scores,
        labels,
        threshold,
    )

    (
        precision,
        recall,
        f1,
        fpr,
        tn,
        fp,
        fn,
        tp,
    ) = result


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = (
            threshold
        )

        best_result = result


(
    precision,
    recall,
    f1,
    fpr,
    tn,
    fp,
    fn,
    tp,
) = best_result


print(
    "\nBest threshold:",
    best_threshold,
)

print(
    "Precision:",
    precision,
)

print(
    "Recall:",
    recall,
)

print(
    "F1:",
    f1,
)

print(
    "False Positive Rate:",
    fpr,
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    "True Negatives:",
    tn,
)

print(
    "False Positives:",
    fp,
)

print(
    "False Negatives:",
    fn,
)

print(
    "True Positives:",
    tp,
)


# ============================================================
# 12. CONTROLLED EVENT DETECTION
# ============================================================

print("\n" + "=" * 80)
print("CONTROLLED ANOMALY EVENTS")
print("=" * 80)


event_rows = [
    (6000, 6180),
    (12000, 12240),
    (18000, 18300),
    (24000, 24120),
    (27000, 27300),
]


for event_start, event_end in event_rows:

    matching_indices = []

    for i, (
        start,
        end,
    ) in enumerate(
        window_ranges
    ):

        # Window overlaps event.

        if (
            start <= event_end
            and end >= event_start
        ):

            matching_indices.append(
                i
            )


    print(
        f"\nEvent: "
        f"rows {event_start} → {event_end}"
    )

    print(
        "Overlapping windows:",
        len(matching_indices)
    )


    if not matching_indices:

        print(
            "No windows overlap event."
        )

        continue


    event_scores = scores[
        matching_indices
    ]


    max_index = (
        matching_indices[
            int(
                np.argmax(
                    event_scores
                )
            )
        ]
    )


    max_score = (
        scores[max_index]
    )


    prediction = (
        "ANOMALY"
        if max_score >= best_threshold
        else "NORMAL"
    )


    print(
        "Maximum score:",
        max_score
    )

    print(
        "Threshold:",
        best_threshold
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Window:",
        window_ranges[max_index]
    )

    print(
        "Window timestamp:",
        timestamps[max_index]
    )


# ============================================================
# 13. TOP ANOMALY WINDOWS
# ============================================================

print("\n" + "=" * 80)
print("TOP 10 HIGHEST-SCORING WINDOWS")
print("=" * 80)


top_indices = np.argsort(
    scores
)[-10:][::-1]


for rank, index in enumerate(
    top_indices,
    start=1,
):

    print(
        f"\n#{rank}"
    )

    print(
        "Score:",
        scores[index]
    )

    print(
        "Prediction:",
        (
            "ANOMALY"
            if scores[index]
            >= best_threshold
            else "NORMAL"
        )
    )

    print(
        "Ground truth:",
        (
            "ANOMALY"
            if labels[index] == 1
            else "NORMAL"
        )
    )

    print(
        "Rows:",
        window_ranges[index]
    )

    print(
        "Timestamp:",
        timestamps[index]
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("DEMO MODEL EVALUATION COMPLETE")
print("=" * 80)

print(
    "\nModel:",
    MODEL_PATH
)

if auc is not None:

    print(
        "ROC-AUC:",
        auc
    )

print(
    "Best threshold:",
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

print("=" * 80)