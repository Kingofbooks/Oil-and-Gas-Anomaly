from abc import ABC,abstractmethod
from dataclasses import dataclass

import pandas as pd

@dataclass
class DetectionResult:
    model_name: str
    model_version: str
    anomaly_score: float
    is_anomaly: bool

class AnomalyDetector(ABC):
    @abstractmethod
    def detect(self, window: pd.DataFrame) -> DetectionResult:
        pass