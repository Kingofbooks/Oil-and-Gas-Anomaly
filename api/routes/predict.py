from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.preprocessing import Preprocessor
from ml.tranad_detector import TranADDetector
from datetime import datetime 

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
)


WINDOW_SIZE = 120
FEATURE_COUNT = 22

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tranad_demo_v1.pt"
)

DATASET_ROOT = PROJECT_ROOT.parent

DEMO_DATASET_PATH = (
    DATASET_ROOT
    / "datasets"
    / "demo"
    / "oil_gas_demo_22features.csv"
)

THRESHOLD = 0.005

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


class SensorWindow(BaseModel):
    readings: list[list[float]] = Field(
        ...,
        description="Exactly 120 rows containing 22 sensor features.",
    )


class PredictionResponse(BaseModel):
    model: str
    version: str
    window_size: int
    feature_count: int
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    predicted_at: datetime


_detector: TranADDetector | None = None


def get_detector() -> TranADDetector:

    global _detector

    if _detector is not None:
        return _detector

    try:
        preprocessor = Preprocessor(
            feature_columns=FEATURE_COLUMNS
        )

        demo_df = pd.read_csv(DEMO_DATASET_PATH)

        preprocessor.fit(
            demo_df,
            feature_columns=FEATURE_COLUMNS,
        )

        _detector = TranADDetector(
            model_path=MODEL_PATH,
            preprocessor=preprocessor,
            threshold=THRESHOLD,
            device="cpu",
        )

        return _detector

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load TranAD detector: {exc}",
        ) from exc


@router.post(
    "",
    response_model=PredictionResponse,
)
def predict(
    window: SensorWindow,
) -> PredictionResponse:

    if len(window.readings) != WINDOW_SIZE:

        raise HTTPException(
            status_code=400,
            detail=(
                f"TranAD requires exactly {WINDOW_SIZE} rows. "
                f"Received {len(window.readings)}."
            ),
        )

    for index, row in enumerate(window.readings):

        if len(row) != FEATURE_COUNT:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Row {index} must contain exactly "
                    f"{FEATURE_COUNT} features. "
                    f"Received {len(row)}."
                ),
            )

    try:

        window_df = pd.DataFrame(
            window.readings,
            columns=FEATURE_COLUMNS,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid sensor window: {exc}",
        ) from exc

    detector = get_detector()

    try:

        result = detector.detect(
            window_df
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"TranAD detection failed: {exc}",
        ) from exc

    # --------------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------------

    return PredictionResponse(
        model=result.model_name,
        version=result.model_version,
        window_size=WINDOW_SIZE,
        feature_count=FEATURE_COUNT,
        anomaly_score=float(result.anomaly_score),
        threshold=THRESHOLD,
        is_anomaly=bool(result.is_anomaly),
        predicted_at=datetime.now(),
    )