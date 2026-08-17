from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances
from ml.preprocessing import Preprocessor
from ml.model import TranADNetwork
from ml.scoring import AnomalyScorer


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas"
    r"\datasets\3w_dataset"
)

MODEL_PATH = Path(
    "artifacts/tranad_v1.pt"
)

TARGET_FILE = (
    DATASET_ROOT
    / "0"
    / "WELL-00001_20170201010207.parquet"
)

WINDOW_SIZE = 120

THRESHOLD = 0.029617823120545266

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FAITHFUL TRANAD WINDOW TEST")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# 1. BUILD SAME DATASET SPLIT AS TRAINING
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


metadata_df = pd.DataFrame(records)


train_metadata, _, _ = (
    split_real_instances(
        metadata_df
    )
)


print(
    "Training instances:",
    len(train_metadata)
)

print(
    "Training wells:",
    train_metadata["well_id"].nunique()
)


# ============================================================
# 2. COLLECT NORMAL TRAINING DATA
# ============================================================

print("\nCollecting normal training data...")

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

    training_frames.append(
        normal
    )


training_data = pd.concat(
    training_frames,
    axis=0
)


print(
    "Training rows:",
    len(training_data)
)


# ============================================================
# 3. FIT EXACT SAME PREPROCESSOR
# ============================================================

print("\nFitting preprocessor...")

preprocessor = Preprocessor()

preprocessor.fit(
    training_data
)


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

print("\nLoading TranAD...")

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
    "Model loaded successfully."
)


# ============================================================
# 5. LOAD TARGET FILE
# ============================================================

print("\n" + "=" * 70)
print("TARGET FILE")
print("=" * 70)

print(
    TARGET_FILE
)


df = pd.read_parquet(
    TARGET_FILE
)


print(
    "Rows:",
    len(df)
)


# ============================================================
# 6. PROCESS MULTIPLE WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("WINDOW SCORES")
print("=" * 70)


starts = [
    0,
    60,
    120,
    300,
    600,
    1200,
    3000,
]


for start in starts:

    end = (
        start
        + WINDOW_SIZE
    )

    if end > len(df):

        continue


    window = df.iloc[
        start:end
    ]


    # --------------------------------------------------------
    # SAME TRANSFORM AS PRODUCTION
    # --------------------------------------------------------

    processed = (
        preprocessor.transform(
            window
        )
    )


    values = processed.to_numpy(
        dtype=np.float32
    )


    values = np.expand_dims(
        values,
        axis=0
    )


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = float(
        scorer.score(
            values
        )[0]
    )


    prediction = (
        "ANOMALY"
        if score >= THRESHOLD
        else "NORMAL"
    )


    print()
    print(
        f"Rows {start:5d}-{end:5d}"
    )

    print(
        "Timestamp:",
        window.index[0],
        "→",
        window.index[-1],
    )

    print(
        "Score:",
        score
    )

    print(
        "Threshold:",
        THRESHOLD
    )

    print(
        "Prediction:",
        prediction
    )


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)