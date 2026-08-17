from pathlib import Path

import numpy as np
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
    / "tranad_v1.pt"
)

WINDOW_SIZE = 120
STRIDE = 60

MAX_ROWS_PER_INSTANCE = 2000

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-3


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("TRAINING ANOMALY MODEL")
print("=" * 70)

print("Device:", device)


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


import pandas as pd

metadata_df = pd.DataFrame(
    records
)


# ============================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_metadata, validation_metadata, test_metadata = (
    split_real_instances(
        metadata_df
    )
)


print("\nDataset split:")

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
# 3. COLLECT NORMAL TRAINING DATA
# ============================================================

print("\nCollecting normal training data...")


training_frames = []


for row in train_metadata.itertuples(
    index=False
):

    df = pd.read_parquet(
        row.path
    )


    # IMPORTANT:
    # The anomaly detector learns NORMAL behavior.
    normal = df[
        df["class"] == 0
    ].copy()


    if normal.empty:
        continue


    normal = normal.iloc[
        :MAX_ROWS_PER_INSTANCE
    ]


    training_frames.append(
        normal
    )


if not training_frames:

    raise RuntimeError(
        "No normal training data found."
    )


# ============================================================
# 4. FIT PREPROCESSOR
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
# 5. CREATE WINDOWS
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


    normal = df[
        df["class"] == 0
    ].copy()


    if len(normal) < WINDOW_SIZE:
        continue


    normal = normal.iloc[
        :MAX_ROWS_PER_INSTANCE
    ]


    processed = (
        preprocessor.transform(
            normal
        )
    )


    windows = (
        window_generator.create_windows(
            processed
        )
    )


    if len(windows) > 0:

        all_windows.append(
            windows
        )


    if index % 50 == 0:

        print(
            f"Processed {index}/"
            f"{len(train_metadata)}"
        )


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
# 6. PYTORCH DATASET
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


# ============================================================
# 7. CREATE MODEL
# ============================================================

model = TranADNetwork(
    input_size=len(
        preprocessor.feature_columns
    ),
    hidden_size=64,
    num_heads=4
)


# ============================================================
# 8. TRAIN
# ============================================================

trainer = Trainer(
    model=model,
    learning_rate=LEARNING_RATE,
    device=device
)


print("\nStarting training...")


for epoch in range(1, EPOCHS + 1):
    loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss:.6f}")


# ============================================================
# 9. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Model saved to:",
    MODEL_PATH
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