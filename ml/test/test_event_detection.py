from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

MODEL_PATH = Path("artifacts/tranad_v1.pt")

WINDOW_SIZE = 120
STRIDE = 60

THRESHOLD = 0.029617823120545266

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("TRANAD V1 - EVENT LEVEL DETECTION TEST")
print("=" * 80)

print("Device:", DEVICE)
print("Threshold:", THRESHOLD)


# ============================================================
# 1. BUILD DATASET INDEX
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


# ============================================================
# 2. USE TRAINING DATA ONLY FOR PREPROCESSOR
# ============================================================

train_metadata, _, _ = split_real_instances(
    metadata_df
)

print("\nTraining instances:", len(train_metadata))
print(
    "Training wells:",
    train_metadata["well_id"].nunique()
)


print("\nCollecting normal training data...")

training_frames = []

for row in train_metadata.itertuples(index=False):

    df = pd.read_parquet(row.path)

    normal = df[
        df["class"] == 0
    ].copy()

    if normal.empty:
        continue

    training_frames.append(normal)


training_data = pd.concat(
    training_frames,
    axis=0,
    ignore_index=False,
)


# ============================================================
# 3. FIT PREPROCESSOR
# ============================================================

print("\nFitting preprocessor...")

preprocessor = Preprocessor()

preprocessor.fit(training_data)

print(
    "Features:",
    len(preprocessor.feature_columns)
)

print(
    preprocessor.feature_columns
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\nLoading TranAD V1...")

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

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

scorer = AnomalyScorer(
    model=model,
    device=DEVICE,
)

print("Model loaded successfully.")


# ============================================================
# 5. WINDOW GENERATOR
# ============================================================

window_generator = TimeSeriesWindowGenerator(
    window_size=WINDOW_SIZE,
    stride=STRIDE,
)


# ============================================================
# 6. SELECT ANOMALY EVENTS
# ============================================================

# We only want real labelled anomaly events.
#
# source == real
# event_class != 0
#
# This excludes synthetic/drawn events for this first
# evaluation.

anomaly_metadata = metadata_df[
    (metadata_df["source"] == "real")
    & (metadata_df["event_class"] != 0)
].copy()


print("\n" + "=" * 80)
print("ANOMALY EVENTS")
print("=" * 80)

print(
    "Total anomaly files:",
    len(anomaly_metadata)
)


# ============================================================
# 7. PROCESS EVENTS
# ============================================================

results = []

for index, row in enumerate(
    anomaly_metadata.itertuples(index=False),
    start=1,
):

    path = Path(row.path)

    print(
        f"\n[{index}/{len(anomaly_metadata)}] "
        f"{path.name}"
    )

    df = pd.read_parquet(path)

    if len(df) < WINDOW_SIZE:
        print("Skipping: file too short")
        continue


    # --------------------------------------------------------
    # Transform entire timeline
    # --------------------------------------------------------

    processed = preprocessor.transform(df)


    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    windows = window_generator.create_windows(
        processed
    )

    if len(windows) == 0:
        print("Skipping: no windows")
        continue


    # --------------------------------------------------------
    # Score windows
    # --------------------------------------------------------

    scores = np.asarray(
        scorer.score(windows)
    )


    # --------------------------------------------------------
    # Determine which windows actually contain
    # anomaly labels.
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

        end = start + WINDOW_SIZE

        window_classes = (
            class_values[start:end]
        )

        labels.append(
            int(
                np.any(
                    window_classes != 0
                )
            )
        )


    labels = np.asarray(
        labels[:len(scores)]
    )


    # --------------------------------------------------------
    # Separate anomaly-containing windows
    # --------------------------------------------------------

    anomaly_scores = scores[
        labels == 1
    ]

    normal_scores = scores[
        labels == 0
    ]


    if len(anomaly_scores) == 0:

        print(
            "WARNING: no anomaly-containing windows"
        )

        continue


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    max_score = float(
        np.max(anomaly_scores)
    )

    median_score = float(
        np.median(anomaly_scores)
    )

    percentile_95 = float(
        np.percentile(
            anomaly_scores,
            95,
        )
    )

    detection_rate = float(
        np.mean(
            anomaly_scores >= THRESHOLD
        )
    )


    # --------------------------------------------------------
    # Event detected?
    # --------------------------------------------------------

    detected = (
        max_score >= THRESHOLD
    )


    # --------------------------------------------------------
    # Normal windows false positives
    # --------------------------------------------------------

    if len(normal_scores) > 0:

        false_positive_rate = float(
            np.mean(
                normal_scores >= THRESHOLD
            )
        )

    else:

        false_positive_rate = 0.0


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Event class: {row.event_class}"
    )

    print(
        f"Total windows: {len(scores)}"
    )

    print(
        f"Anomaly windows: {len(anomaly_scores)}"
    )

    print(
        f"Max score: {max_score:.6f}"
    )

    print(
        f"Median anomaly score: "
        f"{median_score:.6f}"
    )

    print(
        f"95th percentile: "
        f"{percentile_95:.6f}"
    )

    print(
        f"Detection rate: "
        f"{detection_rate:.2%}"
    )

    print(
        f"Normal-window FPR: "
        f"{false_positive_rate:.2%}"
    )

    print(
        "Event prediction:",
        "DETECTED" if detected else "MISSED"
    )


    results.append(
        {
            "file": path.name,
            "well_id": row.well_id,
            "event_class": row.event_class,
            "rows": len(df),
            "windows": len(scores),
            "anomaly_windows": len(anomaly_scores),
            "max_score": max_score,
            "median_score": median_score,
            "p95_score": percentile_95,
            "detection_rate": detection_rate,
            "normal_fpr": false_positive_rate,
            "detected": detected,
        }
    )


# ============================================================
# 8. SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


print("\n")
print("=" * 80)
print("EVENT LEVEL SUMMARY")
print("=" * 80)


if results_df.empty:

    print("No events were evaluated.")

else:

    print(
        results_df[
            [
                "file",
                "event_class",
                "max_score",
                "median_score",
                "p95_score",
                "detection_rate",
                "normal_fpr",
                "detected",
            ]
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Overall detection rate
    # --------------------------------------------------------

    event_detection_rate = (
        results_df["detected"].mean()
    )

    mean_window_detection = (
        results_df["detection_rate"].mean()
    )

    mean_fpr = (
        results_df["normal_fpr"].mean()
    )


    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    print(
        "Events evaluated:",
        len(results_df)
    )

    print(
        "Events detected:",
        int(
            results_df["detected"].sum()
        )
    )

    print(
        "Event detection rate:",
        f"{event_detection_rate:.2%}"
    )

    print(
        "Mean anomaly-window detection rate:",
        f"{mean_window_detection:.2%}"
    )

    print(
        "Mean normal-window FPR:",
        f"{mean_fpr:.2%}"
    )


    # --------------------------------------------------------
    # Best and worst events
    # --------------------------------------------------------

    best_event = results_df.loc[
        results_df["max_score"].idxmax()
    ]

    worst_event = results_df.loc[
        results_df["max_score"].idxmin()
    ]


    print("\n" + "=" * 80)
    print("HIGHEST SCORING EVENT")
    print("=" * 80)

    print(
        "File:",
        best_event["file"]
    )

    print(
        "Max score:",
        best_event["max_score"]
    )


    print("\n" + "=" * 80)
    print("LOWEST SCORING EVENT")
    print("=" * 80)

    print(
        "File:",
        worst_event["file"]
    )

    print(
        "Max score:",
        worst_event["max_score"]
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("EVENT LEVEL TEST COMPLETE")
print("=" * 80)