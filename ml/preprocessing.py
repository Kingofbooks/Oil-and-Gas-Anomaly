from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Preprocessor:

    def __init__(self, feature_columns: list[str] | None = None):
        self.DROP_COLUMNS = {
            "QBS",
            "P-MON-SDV-P",
            "P-JUS-BS",
            "PT-P",
            "P-MON-CKGL",
        }
        self.LABEL_COLUMNS = {"class", "state"}
        self.METADATA_COLUMNS = {"timestamp", "well_id", "time", "date"}

        self.scaler = StandardScaler()
        self.feature_columns: list[str] = feature_columns if feature_columns is not None else []
        self.imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
            fill_value=0,
        )

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Auto-detect numeric feature columns excluding metadata and specified drop targets."""
        excluded = (
            self.DROP_COLUMNS
            | self.LABEL_COLUMNS
            | self.METADATA_COLUMNS
        )

        return [
            column
            for column in df.columns
            if column not in excluded
            and pd.api.types.is_numeric_dtype(df[column])
        ]

    def handle_missing_values(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """Coerce feature columns to numeric and filter extreme values."""
        features = df[feature_columns].copy()

        # Ensure numeric types so .abs() won't throw datetime ufunc errors
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

        # Mask non-physical extreme values
        features = features.mask(features.abs() > 1e10)

        # Forward/Backward fill missing entries
        features = features.ffill().bfill()

        return features

    def fit(
        self, df: pd.DataFrame, feature_columns: list[str] | None = None
    ) -> "Preprocessor":
        """Fit imputer and scaler on the dataset features."""
        if feature_columns is not None:
            self.feature_columns = feature_columns
        elif not self.feature_columns:
            self.feature_columns = self.get_feature_columns(df)

        features = self.handle_missing_values(df, self.feature_columns)

        self.imputer.fit(features)
        imputed = self.imputer.transform(features)

        self.scaler.fit(imputed)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input features using fitted imputer and scaler."""
        if not self.feature_columns:
            raise RuntimeError(
                "Preprocessor has not been fitted yet. Call fit() first."
            )

        features = self.handle_missing_values(df, self.feature_columns)
        imputed = self.imputer.transform(features)
        scaled = self.scaler.transform(imputed)

        return pd.DataFrame(
            scaled,
            index=df.index,
            columns=self.feature_columns,
        )