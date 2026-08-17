from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ml.detector import AnomalyDetector, DetectionResult
from ml.model import TranADNetwork
from ml.preprocessing import Preprocessor
from ml.scoring import AnomalyScorer


class TranADDetector(AnomalyDetector):
    def __init__(
        self,
        model_path: str | Path,
        preprocessor: Preprocessor,
        threshold: float,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.device = device

        self.model = TranADNetwork(
            input_size=len(preprocessor.feature_columns),
            hidden_size=64,
            num_heads=4,
        )

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.scorer = AnomalyScorer(model=self.model, device=self.device)

        print(
            f"TranAD model loaded | Path: {self.model_path} | "
            f"Features: {len(preprocessor.feature_columns)} | Threshold: {self.threshold}"
        )

    def detect(self, window: pd.DataFrame) -> DetectionResult:
        expected_features = self.preprocessor.feature_columns
        missing = [feature for feature in expected_features if feature not in window.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        data = window[expected_features].copy()
        processed = self.preprocessor.transform(data)
        values = processed.to_numpy(dtype=np.float32)

        if values.shape[0] != 120:
            raise ValueError(f"TranAD requires exactly 120 rows. Got {values.shape[0]}")

        values = np.expand_dims(values, axis=0)
        scores = self.scorer.score(values)
        score = float(scores[0])
        is_anomaly = score >= self.threshold

        return DetectionResult(
            model_name="TranAD",
            model_version="v1",
            anomaly_score=score,
            is_anomaly=is_anomaly,
        )