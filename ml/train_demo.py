from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.preprocessing import Preprocessor
from ml.windowing import TimeSeriesWindowGenerator
from ml.torch_dataset import WindowDataset
from ml.model import TranADNetwork
from ml.trainer import Trainer


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\demo\oil_gas_demo_22features.csv"
)

ARTIFACT_DIR = Path("artifacts")

MODEL_PATH = (
    ARTIFACT_DIR
    / "tranad_demo_v1.pt"
)

WINDOW_SIZE = 120
STRIDE = 30

# Keep the experiment genuinely small.
MAX_TRAIN_WINDOWS = 2000

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3


# ============================================================
# EXPECTED FEATURES
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
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPER
# ============================================================

def print_separator():
    print("=" * 80)


# ============================================================
# START
# ============================================================

print_separator()
print("TRAINING TRANAD DEMO MODEL")
print_separator()

print("Device:", DEVICE)
print("Dataset:", DATA_PATH)


# ============================================================
# 1. CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDemo dataset not found:\n"
        f"{DATA_PATH}\n\n"
        f"Expected location:\n"
        f"datasets/demo/oil_gas_demo_22features.csv"
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading demo dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset loaded | Rows:",
    len(df),
    "| Columns:",
    len(df.columns),
)


# ============================================================
# 3. CHECK FEATURES
# ============================================================

print_separator()
print("FEATURE VALIDATION")
print_separator()

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing required features:\n"
        + "\n".join(
            missing_features
        )
    )


print(
    "All 22 ML features are present."
)

print(
    FEATURE_COLUMNS
)


# ============================================================
# 4. CHECK CLASS COLUMN
# ============================================================

if "class" not in df.columns:

    raise ValueError(
        "Demo dataset must contain "
        "a 'class' column."
    )


print_separator()
print("CLASS DISTRIBUTION")
print_separator()

print(
    df["class"]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# ============================================================
# 5. TIMESTAMP
# ============================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values "
            "found in demo dataset."
        )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    print(
        "\nTimeline:",
        df["timestamp"].min(),
        "→",
        df["timestamp"].max(),
    )

else:

    print(
        "\nWARNING:"
        " No timestamp column found."
    )

    print(
        "Using CSV row order."
    )


# ============================================================
# 6. CONVERT FEATURES TO NUMERIC
# ============================================================

print_separator()
print("CLEANING SENSOR DATA")
print_separator()

for feature in FEATURE_COLUMNS:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )


# Replace inf

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


print(
    "Missing values before cleaning:",
    int(
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    ),
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

# Interpolate using the time/row order.

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .interpolate(
        method="linear",
        limit_direction="both",
    )
)


# Remaining NaNs -> median

for feature in FEATURE_COLUMNS:

    if df[feature].isna().any():

        median = df[feature].median()

        if pd.isna(median):

            raise ValueError(
                f"Feature '{feature}' "
                "contains only NaN values."
            )

        df[feature] = (
            df[feature]
            .fillna(median)
        )


missing_after = int(
    df[FEATURE_COLUMNS]
    .isna()
    .sum()
    .sum()
)

print(
    "Missing values after cleaning:",
    missing_after,
)


if missing_after > 0:

    raise ValueError(
        "Dataset still contains "
        "missing feature values."
    )


# ============================================================
# 8. SELECT NORMAL DATA
# ============================================================

print_separator()
print("SELECTING NORMAL TRAINING DATA")
print_separator()

normal_mask = (
    pd.to_numeric(
        df["class"],
        errors="coerce",
    )
    == 0
)

normal_df = df[
    normal_mask
].copy()


print(
    "Total rows:",
    len(df)
)

print(
    "Normal rows:",
    len(normal_df)
)

print(
    "Non-normal rows:",
    len(df) - len(normal_df)
)


if len(normal_df) < WINDOW_SIZE:

    raise ValueError(
        "Not enough normal rows "
        "to create a 120-row window."
    )


# ============================================================
# 9. FIT PREPROCESSOR
# ============================================================

print_separator()
print("FITTING PREPROCESSOR")
print_separator()

preprocessor = Preprocessor()

# IMPORTANT:
# Follow the same pattern as the working
# ml/train.py pipeline.

preprocessor.fit(
    normal_df
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


if (
    len(preprocessor.feature_columns)
    != 22
):

    raise ValueError(
        "Expected 22 features, got "
        f"{len(preprocessor.feature_columns)}"
    )


# ============================================================
# 10. CREATE WINDOWS
# ============================================================

print_separator()
print("CREATING TRAINING WINDOWS")
print_separator()

window_generator = (
    TimeSeriesWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
)


# ------------------------------------------------------------
# IMPORTANT
#
# Do NOT simply concatenate all normal rows and then
# window them.
#
# If an anomaly exists between two normal sections,
# filtering it out would make two unrelated timestamps
# appear adjacent.
#
# Therefore we create windows from contiguous normal
# sections.
# ------------------------------------------------------------

normal_values = (
    normal_mask
    .astype(int)
    .to_numpy()
)


segments = []

segment_start = None

for index, value in enumerate(
    normal_values
):

    if value == 1:

        if segment_start is None:
            segment_start = index

    else:

        if segment_start is not None:

            segments.append(
                (
                    segment_start,
                    index,
                )
            )

            segment_start = None


# Handle final normal segment

if segment_start is not None:

    segments.append(
        (
            segment_start,
            len(df),
        )
    )


print(
    "Contiguous normal segments:",
    len(segments)
)


# ============================================================
# 11. BUILD WINDOWS
# ============================================================

all_windows = []

total_windows = 0


for segment_index, (
    start,
    end,
) in enumerate(
    segments,
    start=1,
):

    segment = df.iloc[
        start:end
    ].copy()


    if len(segment) < WINDOW_SIZE:
        continue


    processed = (
        preprocessor.transform(
            segment
        )
    )


    windows = (
        window_generator.create_windows(
            processed
        )
    )


    if len(windows) == 0:
        continue


    remaining = (
        MAX_TRAIN_WINDOWS
        - total_windows
    )


    if remaining <= 0:
        break


    if len(windows) > remaining:

        windows = windows[
            :remaining
        ]


    all_windows.append(
        windows
    )


    total_windows += len(
        windows
    )


    print(
        f"Segment {segment_index}: "
        f"rows={len(segment):,} | "
        f"windows={len(windows)} | "
        f"total={total_windows}"
    )


    if total_windows >= MAX_TRAIN_WINDOWS:

        break


# ============================================================
# 12. FINAL WINDOWS
# ============================================================

if not all_windows:

    raise RuntimeError(
        "No training windows were created."
    )


windows = np.concatenate(
    all_windows,
    axis=0,
).astype(
    np.float32
)


print_separator()
print("TRAINING WINDOW DATASET")
print_separator()

print(
    "Training windows:",
    windows.shape
)

print(
    "Dataset samples:",
    len(windows)
)

print(
    "Window size:",
    WINDOW_SIZE
)

print(
    "Features:",
    windows.shape[-1]
)


if windows.shape[1] != WINDOW_SIZE:

    raise ValueError(
        "Unexpected window size: "
        f"{windows.shape}"
    )

if windows.shape[2] != 22:

    raise ValueError(
        "Unexpected feature count: "
        f"{windows.shape}"
    )


# ============================================================
# 13. PYTORCH DATASET
# ============================================================

train_dataset = WindowDataset(
    windows
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# 14. CREATE MODEL
# ============================================================

print_separator()
print("CREATING TRANAD MODEL")
print_separator()

model = TranADNetwork(
    input_size=22,
    hidden_size=64,
    num_heads=4,
)


# ============================================================
# 15. TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
)


# ============================================================
# 16. TRAIN
# ============================================================

print_separator()
print("STARTING TRAINING")
print_separator()


for epoch in range(
    1,
    EPOCHS + 1,
):

    loss = trainer.train_epoch(
        train_loader
    )

    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"- Loss: {loss:.6f}"
    )


# ============================================================
# 17. SAVE MODEL
# ============================================================

print_separator()
print("SAVING MODEL")
print_separator()

ARTIFACT_DIR.mkdir(
    exist_ok=True
)


torch.save(
    model.state_dict(),
    MODEL_PATH,
)


print(
    "Model saved to:",
    MODEL_PATH
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print_separator()
print("TRANAD DEMO TRAINING COMPLETE")
print_separator()

print(
    "Dataset rows:",
    len(df)
)

print(
    "Normal rows:",
    len(normal_df)
)

print(
    "Training windows:",
    len(windows)
)

print(
    "Input features:",
    len(
        preprocessor.feature_columns
    )
)

print(
    "Window size:",
    WINDOW_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Model:",
    MODEL_PATH
)

print_separator()