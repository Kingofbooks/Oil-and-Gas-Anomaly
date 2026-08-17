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
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

MODEL_PATH = Path(
    "artifacts/attention_reconstruction_v1.pt"
)

WINDOW_SIZE = 120
STRIDE = 60

MAX_ROWS_PER_INSTANCE = 2000


# ============================================================
# 1. BUILD DATASET METADATA
# ============================================================

print("=" * 70)
print("BUILDING DATASET INDEX")
print("=" * 70)

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
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_metadata, validation_metadata, test_metadata = (
    split_real_instances(metadata_df)
)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Train instances:      {len(train_metadata)}"
)

print(
    f"Validation instances: {len(validation_metadata)}"
)

print(
    f"Test instances:       {len(test_metadata)}"
)


# ============================================================
# 3. FIT PREPROCESSOR ON TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 70)
print("FITTING PREPROCESSOR")
print("=" * 70)


preprocessor = Preprocessor()

training_frames = []


for index, row in enumerate(
    train_metadata.itertuples(index=False),
    start=1,
):

    df = pd.read_parquet(row.path)

    # Training uses NORMAL data only.
    normal = df[
        df["class"] == 0
    ].copy()

    if normal.empty:
        continue

    # Same limit used during training.
    normal = normal.iloc[
        :MAX_ROWS_PER_INSTANCE
    ]

    training_frames.append(normal)


if not training_frames:
    raise RuntimeError(
        "No normal training data found."
    )


combined_training = pd.concat(
    training_frames,
    axis=0,
)


preprocessor.fit(
    combined_training
)


print(
    "Number of features:",
    len(preprocessor.feature_columns),
)

print(
    "Features:",
    preprocessor.feature_columns,
)


# ============================================================
# 4. WINDOW GENERATOR
# ============================================================

window_generator = TimeSeriesWindowGenerator(
    window_size=WINDOW_SIZE,
    stride=STRIDE,
)


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)


model = TranADNetwork(
    input_size=len(
        preprocessor.feature_columns
    ),
    hidden_size=64,
    num_heads=4,
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
    )
)


model.eval()


scorer = AnomalyScorer(
    model,
    device="cpu",
)


print(
    "Model loaded:",
    MODEL_PATH,
)


# ============================================================
# 6. BUILD VALIDATION WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING VALIDATION WINDOWS")
print("=" * 70)


normal_windows = []
normal_metadata = []

anomaly_windows = []
anomaly_metadata = []


for index, row in enumerate(
    validation_metadata.itertuples(index=False),
    start=1,
):

    print(
        f"{index}/{len(validation_metadata)} "
        f"{row.path.name}"
    )

    df = pd.read_parquet(row.path)


    # --------------------------------------------------------
    # NORMAL DATA
    # --------------------------------------------------------

    normal = df[
        df["class"] == 0
    ].copy()


    if len(normal) >= WINDOW_SIZE:

        processed_normal = (
            preprocessor.transform(normal)
        )


        windows = (
            window_generator.create_windows(
                processed_normal
            )
        )


        if len(windows) > 0:

            normal_windows.append(
                windows
            )


            # Preserve metadata for every window.
            for window_index in range(
                len(windows)
            ):

                start_position = (
                    window_index * STRIDE
                )

                end_position = (
                    start_position
                    + WINDOW_SIZE
                    - 1
                )


                if (
                    start_position
                    >= len(normal)
                ):
                    continue


                end_position = min(
                    end_position,
                    len(normal) - 1,
                )


                normal_metadata.append(
                    {
                        "file": row.path.name,
                        "well_id": row.well_id,
                        "event_class": row.event_class,
                        "window_index": window_index,
                        "start_time": normal.index[
                            start_position
                        ],
                        "end_time": normal.index[
                            end_position
                        ],
                    }
                )


    # --------------------------------------------------------
    # ANOMALY DATA
    # --------------------------------------------------------

    anomaly = df[
        df["class"] != 0
    ].copy()


    if len(anomaly) >= WINDOW_SIZE:

        processed_anomaly = (
            preprocessor.transform(
                anomaly
            )
        )


        windows = (
            window_generator.create_windows(
                processed_anomaly
            )
        )


        if len(windows) > 0:

            anomaly_windows.append(
                windows
            )


            for window_index in range(
                len(windows)
            ):

                start_position = (
                    window_index * STRIDE
                )

                end_position = (
                    start_position
                    + WINDOW_SIZE
                    - 1
                )


                if (
                    start_position
                    >= len(anomaly)
                ):
                    continue


                end_position = min(
                    end_position,
                    len(anomaly) - 1,
                )


                anomaly_metadata.append(
                    {
                        "file": row.path.name,
                        "well_id": row.well_id,
                        "event_class": row.event_class,
                        "window_index": window_index,
                        "start_time": anomaly.index[
                            start_position
                        ],
                        "end_time": anomaly.index[
                            end_position
                        ],
                    }
                )


# ============================================================
# 7. COMBINE WINDOWS
# ============================================================

if not normal_windows:

    raise RuntimeError(
        "No normal validation windows were created."
    )


if not anomaly_windows:

    raise RuntimeError(
        "No anomaly validation windows were created."
    )


normal_windows = np.concatenate(
    normal_windows,
    axis=0,
).astype(
    np.float32
)


anomaly_windows = np.concatenate(
    anomaly_windows,
    axis=0,
).astype(
    np.float32
)


normal_metadata = pd.DataFrame(
    normal_metadata
)

anomaly_metadata = pd.DataFrame(
    anomaly_metadata
)


print("\n" + "=" * 70)
print("VALIDATION WINDOWS")
print("=" * 70)


print(
    "Normal windows:",
    normal_windows.shape,
)


print(
    "Anomaly windows:",
    anomaly_windows.shape,
)


print(
    "Normal metadata:",
    len(normal_metadata),
)


print(
    "Anomaly metadata:",
    len(anomaly_metadata),
)


# ============================================================
# 8. SCORE WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("SCORING")
print("=" * 70)


print("Scoring normal windows...")


normal_scores = scorer.score(
    normal_windows
)


print("Scoring anomaly windows...")


anomaly_scores = scorer.score(
    anomaly_windows
)


# ============================================================
# 9. NORMAL SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("NORMAL SCORE DISTRIBUTION")
print("=" * 70)


print(
    "Minimum:",
    normal_scores.min(),
)


print(
    "Maximum:",
    normal_scores.max(),
)


print(
    "Mean:",
    normal_scores.mean(),
)


print(
    "Median:",
    np.median(normal_scores),
)


print(
    "90th percentile:",
    np.percentile(
        normal_scores,
        90,
    ),
)


print(
    "95th percentile:",
    np.percentile(
        normal_scores,
        95,
    ),
)


print(
    "99th percentile:",
    np.percentile(
        normal_scores,
        99,
    ),
)


# ============================================================
# 10. ANOMALY SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY SCORE DISTRIBUTION")
print("=" * 70)


print(
    "Minimum:",
    anomaly_scores.min(),
)


print(
    "Maximum:",
    anomaly_scores.max(),
)


print(
    "Mean:",
    anomaly_scores.mean(),
)


print(
    "Median:",
    np.median(anomaly_scores),
)


print(
    "5th percentile:",
    np.percentile(
        anomaly_scores,
        5,
    ),
)


print(
    "10th percentile:",
    np.percentile(
        anomaly_scores,
        10,
    ),
)


# ============================================================
# 11. TOP NORMAL OUTLIERS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 NORMAL SCORE OUTLIERS")
print("=" * 70)


top_normal_indices = np.argsort(
    normal_scores
)[-10:][::-1]


for rank, index in enumerate(
    top_normal_indices,
    start=1,
):

    metadata_row = (
        normal_metadata.iloc[index]
    )


    print(
        f"\n#{rank}"
    )


    print(
        "Score:",
        normal_scores[index],
    )


    print(
        "File:",
        metadata_row["file"],
    )


    print(
        "Well:",
        metadata_row["well_id"],
    )


    print(
        "Event class:",
        metadata_row["event_class"],
    )


    print(
        "Window:",
        metadata_row["window_index"],
    )


    print(
        "Start:",
        metadata_row["start_time"],
    )


    print(
        "End:",
        metadata_row["end_time"],
    )


# ============================================================
# 12. TOP ANOMALY SCORES
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 ANOMALY SCORE OUTLIERS")
print("=" * 70)


top_anomaly_indices = np.argsort(
    anomaly_scores
)[-10:][::-1]


for rank, index in enumerate(
    top_anomaly_indices,
    start=1,
):

    metadata_row = (
        anomaly_metadata.iloc[index]
    )


    print(
        f"\n#{rank}"
    )


    print(
        "Score:",
        anomaly_scores[index],
    )


    print(
        "File:",
        metadata_row["file"],
    )


    print(
        "Well:",
        metadata_row["well_id"],
    )


    print(
        "Event class:",
        metadata_row["event_class"],
    )


    print(
        "Window:",
        metadata_row["window_index"],
    )


    print(
        "Start:",
        metadata_row["start_time"],
    )


    print(
        "End:",
        metadata_row["end_time"],
    )


# ============================================================
# 13. INITIAL THRESHOLD
# ============================================================

threshold = np.percentile(
    normal_scores,
    95,
)


normal_predictions = (
    normal_scores > threshold
)


anomaly_predictions = (
    anomaly_scores > threshold
)


false_positive_rate = (
    normal_predictions.mean()
)


detection_rate = (
    anomaly_predictions.mean()
)


print("\n" + "=" * 70)
print("INITIAL THRESHOLD CHECK")
print("=" * 70)


print(
    "Threshold:",
    threshold,
)


print(
    "Normal false positive rate:",
    false_positive_rate,
)


print(
    "Anomaly detection rate:",
    detection_rate,
)


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)