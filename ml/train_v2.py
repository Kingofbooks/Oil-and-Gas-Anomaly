from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances
from ml.preprocessing import Preprocessor
from ml.windowing import TimeSeriesWindowGenerator
from ml.torch_dataset import WindowDataset
from ml.model import TranADNetwork
from ml.trainer import Trainer


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

ARTIFACT_DIR = Path("artifacts")

ARTIFACT_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "tranad_v2.pt"
)

WINDOW_SIZE = 120
STRIDE = 60

# Same restriction used by your working train.py
MAX_ROWS_PER_INSTANCE = 2000

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-3


# ============================================================
# SMALL DATASET
# ============================================================

SELECTED_WELLS = {
    "WELL-00011",
    "WELL-00014",
    "WELL-00016",
    "WELL-00038",
}


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRAINING TRANAD V2 - SMALL DATASET")
print("=" * 70)

print("Device:", device)

print("\nSelected wells:")

for well in sorted(SELECTED_WELLS):
    print(" ", well)


# ============================================================
# 1. BUILD DATASET INDEX
# ============================================================

print("\nBuilding dataset index...")

dataset = ThreeWDataset(
    DATASET_ROOT
)

metadata = dataset.build_index()


# ============================================================
# 2. CREATE METADATA DATAFRAME
# ============================================================

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
# 3. SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_metadata, validation_metadata, test_metadata = (
    split_real_instances(
        metadata_df
    )
)


print("\nOriginal dataset split:")

print(
    "Train:",
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


# ============================================================
# 4. KEEP ONLY SELECTED TRAINING WELLS
# ============================================================

train_metadata = train_metadata[
    train_metadata["well_id"].isin(
        SELECTED_WELLS
    )
].copy()


print("\nSmall training split:")

print(
    "Train instances:",
    len(train_metadata)
)

print(
    "Train wells:",
    train_metadata["well_id"].nunique()
)

print(
    sorted(
        train_metadata["well_id"].unique()
    )
)


if train_metadata.empty:

    raise RuntimeError(
        "No training instances found "
        "for the selected wells."
    )


# ============================================================
# 5. COLLECT NORMAL TRAINING DATA
# ============================================================

print("\nCollecting normal training data...")


training_frames = []

total_rows = 0
normal_rows = 0


for row in train_metadata.itertuples(
    index=False
):

    df = pd.read_parquet(
        row.path
    )

    total_rows += len(df)

    # --------------------------------------------------------
    # TranAD learns NORMAL behavior only.
    # --------------------------------------------------------

    normal = df[
        df["class"] == 0
    ].copy()


    if normal.empty:
        continue


    # --------------------------------------------------------
    # Keep the exact same limitation as v1.
    # --------------------------------------------------------

    normal = normal.iloc[
        :MAX_ROWS_PER_INSTANCE
    ]


    if len(normal) < WINDOW_SIZE:
        continue


    normal_rows += len(normal)


    training_frames.append(
        normal
    )


print(
    "\nRaw rows encountered:",
    f"{total_rows:,}"
)

print(
    "Normal rows used:",
    f"{normal_rows:,}"
)

print(
    "Training files used:",
    len(training_frames)
)


if not training_frames:

    raise RuntimeError(
        "No normal training data found."
    )


# ============================================================
# 6. FIT PREPROCESSOR
# ============================================================

print("\nFitting preprocessor...")


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

print(
    preprocessor.feature_columns
)


# ============================================================
# 7. CREATE WINDOWS
# ============================================================

print("\nCreating training windows...")


window_generator = (
    TimeSeriesWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )
)


all_windows = []


for index, row in enumerate(
    train_metadata.itertuples(
        index=False
    ),
    start=1
):

    df = pd.read_parquet(
        row.path
    )


    # --------------------------------------------------------
    # ONLY NORMAL DATA
    # --------------------------------------------------------

    normal = df[
        df["class"] == 0
    ].copy()


    if len(normal) < WINDOW_SIZE:

        continue


    # --------------------------------------------------------
    # SAME LIMIT AS TRAINING DATA
    # --------------------------------------------------------

    normal = normal.iloc[
        :MAX_ROWS_PER_INSTANCE
    ]


    if len(normal) < WINDOW_SIZE:

        continue


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    processed = (
        preprocessor.transform(
            normal
        )
    )


    # --------------------------------------------------------
    # CREATE WINDOWS
    # --------------------------------------------------------

    windows = (
        window_generator.create_windows(
            processed
        )
    )


    if len(windows) > 0:

        all_windows.append(
            windows
        )


    if index % 25 == 0:

        print(
            f"Processed "
            f"{index}/"
            f"{len(train_metadata)}"
        )


# ============================================================
# 8. COMBINE WINDOWS
# ============================================================

if not all_windows:

    raise RuntimeError(
        "No training windows created."
    )


windows = np.concatenate(
    all_windows,
    axis=0
).astype(
    np.float32
)


print(
    "\nTraining windows:",
    windows.shape
)


# ============================================================
# 9. PYTORCH DATASET
# ============================================================

train_dataset = WindowDataset(
    windows
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print(
    "Dataset samples:",
    len(train_dataset)
)


print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# 10. CREATE TRANAD MODEL
# ============================================================

print("\nCreating TranAD model...")


model = TranADNetwork(
    input_size=len(
        preprocessor.feature_columns
    ),
    hidden_size=64,
    num_heads=4
)


# ============================================================
# 11. TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    learning_rate=LEARNING_RATE,
    device=device
)


# ============================================================
# 12. TRAIN
# ============================================================

print("\nStarting training...")


for epoch in range(
    1,
    EPOCHS + 1
):

    loss = (
        trainer.train_epoch(
            train_loader
        )
    )

    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"- Loss: {loss:.6f}"
    )


# ============================================================
# 13. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# 14. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRANAD V2 TRAINING COMPLETE")
print("=" * 70)


print(
    "Model saved to:",
    MODEL_PATH
)


print(
    "Selected wells:",
    sorted(SELECTED_WELLS)
)


print(
    "Training files:",
    len(training_frames)
)


print(
    "Normal rows used:",
    f"{normal_rows:,}"
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
    "Training windows:",
    len(windows)
)