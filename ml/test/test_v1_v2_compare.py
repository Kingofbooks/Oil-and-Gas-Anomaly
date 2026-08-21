from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.preprocessing import Preprocessor
from ml.model import TranADNetwork
from ml.scoring import AnomalyScorer


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset"
)

FILE_PATH = (
    DATASET_ROOT
    / "1"
    / "DRAWN_00001.parquet"
)

V1_MODEL = Path(
    "artifacts/tranad_v1.pt"
)

V2_MODEL = Path(
    "artifacts/tranad_v2.pt"
)

WINDOW_SIZE = 120

HIDDEN_SIZE = 64
NUM_HEADS = 4

THRESHOLD_V1 = 0.029617823120545266

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TARGET WINDOW
# ============================================================

TARGET_TIME = pd.Timestamp(
    "2018-09-06 01:04:00"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("TRANAD V1 vs V2 - EXACT SAME ANOMALY WINDOW")
print("=" * 80)

print("Device:", DEVICE)


# ============================================================
# 1. LOAD FILE
# ============================================================

print("\nLoading target file...")

df = pd.read_parquet(
    FILE_PATH
)


print(
    "File:",
    FILE_PATH
)

print(
    "Rows:",
    len(df)
)

print(
    "Timeline:",
    df.index.min(),
    "→",
    df.index.max()
)


# ============================================================
# 2. FIND TARGET TIMESTAMP
# ============================================================

if TARGET_TIME not in df.index:

    nearest_index = (
        np.abs(
            df.index
            .astype("int64")
            - TARGET_TIME.value
        )
    ).argmin()

    target_index = int(
        nearest_index
    )

else:

    target_index = df.index.get_loc(
        TARGET_TIME
    )


print("\nTarget time:")
print(
    TARGET_TIME
)

print(
    "Target row:",
    target_index
)


# ============================================================
# 3. EXTRACT EXACT 120 ROW WINDOW
# ============================================================

start = (
    target_index
    - WINDOW_SIZE
    + 1
)

end = (
    target_index
    + 1
)


if start < 0:

    raise RuntimeError(
        "Not enough rows before target timestamp."
    )


window = df.iloc[
    start:end
].copy()


if len(window) != WINDOW_SIZE:

    raise RuntimeError(
        f"Expected {WINDOW_SIZE} rows, "
        f"got {len(window)}"
    )


print("\n" + "=" * 80)
print("EXACT WINDOW")
print("=" * 80)

print(
    "Rows:",
    start,
    "→",
    target_index
)

print(
    "Timestamp:",
    window.index[0],
    "→",
    window.index[-1]
)


# ============================================================
# 4. GROUND TRUTH
# ============================================================

class_values = (
    window["class"]
    .fillna(0)
    .to_numpy()
)


unique_classes = set(
    class_values.tolist()
)


contains_anomaly = np.any(
    class_values != 0
)


print(
    "\nGround truth classes:",
    unique_classes
)

print(
    "Contains anomaly:",
    contains_anomaly
)


# ============================================================
# 5. FIT PREPROCESSOR
# ============================================================

print("\n" + "=" * 80)
print("FITTING PREPROCESSOR")
print("=" * 80)


# ------------------------------------------------------------
# IMPORTANT:
#
# The preprocessor must be fitted on NORMAL TRAINING DATA.
# We reproduce the same training data used by the project.
# ------------------------------------------------------------

from ml.dataset import ThreeWDataset
from ml.split import split_real_instances


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


metadata_df = pd.DataFrame(
    records
)


train_metadata, _, _ = (
    split_real_instances(
        metadata_df
    )
)


training_frames = []


for row in train_metadata.itertuples(
    index=False
):

    train_df = pd.read_parquet(
        row.path
    )

    normal = train_df[
        train_df["class"] == 0
    ].copy()


    if normal.empty:
        continue


    normal = normal.iloc[
        :2000
    ]


    training_frames.append(
        normal
    )


if not training_frames:

    raise RuntimeError(
        "Could not collect training data."
    )


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


# ============================================================
# 6. PREPROCESS TARGET WINDOW
# ============================================================

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


print(
    "Input shape:",
    values.shape
)


# ============================================================
# 7. FUNCTION TO TEST MODEL
# ============================================================

def test_model(
    model_path: Path,
    name: str,
):

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    model = TranADNetwork(
        input_size=len(
            preprocessor.feature_columns
        ),
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
    )


    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )


    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()


    scorer = AnomalyScorer(
        model=model,
        device=DEVICE,
    )


    scores = scorer.score(
        values
    )


    score = float(
        scores[0]
    )


    print(
        "Model:",
        name
    )

    print(
        "Score:",
        score
    )


    return score


# ============================================================
# 8. RUN V1
# ============================================================

v1_score = test_model(
    V1_MODEL,
    "TranAD V1"
)


# ============================================================
# 9. RUN V2
# ============================================================

v2_score = test_model(
    V2_MODEL,
    "TranAD V2"
)


# ============================================================
# 10. COMPARE
# ============================================================

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)


print(
    "\nGround truth:",
    "ANOMALY"
    if contains_anomaly
    else "NORMAL"
)


print(
    "\nV1 score:",
    v1_score
)

print(
    "V1 threshold:",
    THRESHOLD_V1
)

print(
    "V1 prediction:",
    "ANOMALY"
    if v1_score >= THRESHOLD_V1
    else "NORMAL"
)


print(
    "\nV2 score:",
    v2_score
)


print(
    "V2 prediction using V1 threshold:",
    "ANOMALY"
    if v2_score >= THRESHOLD_V1
    else "NORMAL"
)


# ============================================================
# 11. SCORE CHANGE
# ============================================================

difference = (
    v2_score
    - v1_score
)


ratio = (
    v2_score / v1_score
    if v1_score != 0
    else float("inf")
)


print(
    "\nScore difference:",
    difference
)

print(
    "V2 / V1 ratio:",
    ratio
)


# ============================================================
# 12. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)


if v2_score > v1_score:

    print(
        "V2 produces a HIGHER anomaly score."
    )

elif v2_score < v1_score:

    print(
        "V2 produces a LOWER anomaly score."
    )

else:

    print(
        "V1 and V2 produce the same score."
    )


if (
    v1_score < THRESHOLD_V1
    and v2_score >= THRESHOLD_V1
):

    print(
        "\nSUCCESS:"
    )

    print(
        "V2 detects the anomaly that V1 missed."
    )


elif (
    v1_score < THRESHOLD_V1
    and v2_score < THRESHOLD_V1
):

    print(
        "\nV2 ALSO MISSES THE ANOMALY."
    )

    print(
        "Changing the training dataset alone "
        "did not solve the problem."
    )


elif (
    v1_score >= THRESHOLD_V1
    and v2_score >= THRESHOLD_V1
):

    print(
        "\nBoth models detect the anomaly."
    )


print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)