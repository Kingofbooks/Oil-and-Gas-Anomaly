from pathlib import Path

import pandas as pd
import torch

from ml.preprocessing import Preprocessor
from ml.tranad_detector import TranADDetector


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\demo\oil_gas_demo_22features.csv"
)

MODEL_PATH = Path(
    "artifacts/tranad_demo_v1.pt"
)

THRESHOLD = 0.005179412059486259

WINDOW_SIZE = 120

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

def prepare_data(df):

    df = df.copy()

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(
            drop=True
        )

    for feature in FEATURE_COLUMNS:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    for feature in FEATURE_COLUMNS:

        if df[feature].isna().any():

            df[feature] = (
                df[feature]
                .fillna(
                    df[feature].median()
                )
            )

    return df


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TRANAD DEMO DETECTOR TEST")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# 1. LOAD DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


df = pd.read_csv(
    DATA_PATH
)

df = prepare_data(
    df
)


print(
    "\nDataset rows:",
    len(df)
)


# ============================================================
# 2. FIT PREPROCESSOR
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
    len(
        preprocessor.feature_columns
    )
)


# ============================================================
# 3. LOAD DETECTOR
# ============================================================

print("\nLoading detector...")

detector = TranADDetector(
    model_path=MODEL_PATH,
    preprocessor=preprocessor,
    threshold=THRESHOLD,
    device=DEVICE,
)


# ============================================================
# 4. FIND NORMAL WINDOW
# ============================================================

normal_indices = (
    df.index[
        df["class"] == 0
    ].tolist()
)


normal_start = None

for start in normal_indices:

    end = start + WINDOW_SIZE

    if end > len(df):
        break

    window_classes = (
        df["class"]
        .iloc[start:end]
    )

    if (
        window_classes == 0
    ).all():

        normal_start = start
        break


if normal_start is None:

    raise RuntimeError(
        "Could not find a completely "
        "normal 120-row window."
    )


normal_window = df.iloc[
    normal_start:
    normal_start + WINDOW_SIZE
].copy()


# ============================================================
# 5. TEST NORMAL WINDOW
# ============================================================

print("\n" + "=" * 70)
print("NORMAL WINDOW TEST")
print("=" * 70)

print(
    "Rows:",
    normal_start,
    "→",
    normal_start + WINDOW_SIZE - 1,
)

if "timestamp" in normal_window.columns:

    print(
        "Timestamp:",
        normal_window["timestamp"].iloc[0],
        "→",
        normal_window["timestamp"].iloc[-1],
    )

print(
    "Ground truth: NORMAL"
)


normal_result = detector.detect(
    normal_window
)


print(
    "\nModel:",
    normal_result.model_name
)

print(
    "Version:",
    normal_result.model_version
)

print(
    "Score:",
    normal_result.anomaly_score
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Prediction:",
    (
        "ANOMALY"
        if normal_result.is_anomaly
        else "NORMAL"
    )
)


# ============================================================
# 6. FIND STRONG ANOMALY WINDOW
# ============================================================

print(
    "\nSearching for the strongest anomaly window..."
)

best_anomaly_start = None
best_anomaly_count = -1


for start in range(
    0,
    len(df) - WINDOW_SIZE + 1,
    30,
):

    end = start + WINDOW_SIZE

    window_classes = (
        df["class"]
        .iloc[start:end]
    )

    anomaly_count = int(
        (window_classes != 0).sum()
    )

    if anomaly_count > best_anomaly_count:

        best_anomaly_count = (
            anomaly_count
        )

        best_anomaly_start = start


if best_anomaly_start is None:

    raise RuntimeError(
        "Could not find an anomaly window."
    )


anomaly_window = df.iloc[
    best_anomaly_start:
    best_anomaly_start + WINDOW_SIZE
].copy()


print(
    "Strongest anomaly window found:"
)

print(
    "Rows:",
    best_anomaly_start,
    "→",
    best_anomaly_start + WINDOW_SIZE - 1,
)

print(
    "Anomalous rows:",
    best_anomaly_count,
    "/",
    WINDOW_SIZE,
)


if "timestamp" in anomaly_window.columns:

    print(
        "Timestamp:",
        anomaly_window["timestamp"].iloc[0],
        "→",
        anomaly_window["timestamp"].iloc[-1],
    )

print(
    "Classes:",
    anomaly_window["class"]
    .value_counts()
    .to_dict()
)

# ============================================================
# 7. TEST DIFFERENT ANOMALY WINDOW STRENGTHS
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY WINDOW ROBUSTNESS TEST")
print("=" * 70)


candidate_windows = []


for start in range(
    0,
    len(df) - WINDOW_SIZE + 1,
    30,
):

    end = start + WINDOW_SIZE

    window = df.iloc[
        start:end
    ]

    anomaly_count = int(
        (window["class"] != 0).sum()
    )

    if anomaly_count > 0:

        candidate_windows.append(
            (
                anomaly_count,
                start,
                end,
            )
        )


candidate_windows.sort(
    reverse=True
)


# Test up to 5 strongest windows.

tested = set()


for (
    anomaly_count,
    start,
    end,
) in candidate_windows[:5]:

    if start in tested:
        continue

    tested.add(start)

    window = df.iloc[
        start:end
    ].copy()

    result = detector.detect(
        window
    )

    print(
        f"\nRows {start} → {end - 1}"
    )

    print(
        "Anomalous rows:",
        anomaly_count,
        "/",
        WINDOW_SIZE,
    )

    print(
        "Score:",
        result.anomaly_score,
    )

    print(
        "Threshold:",
        THRESHOLD,
    )

    print(
        "Prediction:",
        (
            "ANOMALY"
            if result.is_anomaly
            else "NORMAL"
        ),
    )


# ============================================================
# 8. CONTRACT CHECK
# ============================================================

# Execute detection on the strongest anomaly window defined in section 6
anomaly_result = detector.detect(anomaly_window)

print("\n" + "=" * 70)
print("CONTRACT CHECK")
print("=" * 70)

assert normal_result.model_name == "TranAD"
assert anomaly_result.model_name == "TranAD"

assert normal_result.model_version == "demo-v1"
assert anomaly_result.model_version == "demo-v1"

assert isinstance(
    normal_result.anomaly_score,
    float,
)

assert isinstance(
    anomaly_result.anomaly_score,
    float,
)

print(
    "DetectionResult contract: PASSED"
)

# ============================================================
# 9. EXPECTED RESULT CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED RESULT CHECK")
print("=" * 70)


normal_correct = (
    not normal_result.is_anomaly
)

anomaly_correct = (
    anomaly_result.is_anomaly
)


print(
    "Normal classified correctly:",
    normal_correct
)

print(
    "Anomaly classified correctly:",
    anomaly_correct
)


if not normal_correct:

    print(
        "\nWARNING:"
        " Normal window was classified "
        "as ANOMALY."
    )


if not anomaly_correct:

    print(
        "\nWARNING:"
        " Anomaly window was classified "
        "as NORMAL."
    )


if (
    normal_correct
    and anomaly_correct
):

    print(
        "\nSUCCESS:"
        " Detector correctly classified "
        "both test windows."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DEMO DETECTOR TEST COMPLETE")
print("=" * 70)