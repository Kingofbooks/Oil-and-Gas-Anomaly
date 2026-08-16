from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

class Preprocessor:
    def __init__(self):
        self.DROP_COLUMNS = {"QBS","P-MON-SDV-P","P-JUS-BS","PT-P","P-MON-CKGL"}
        self.LABEL_COLUMNS = {"class","state"}
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        # Keep the feature dimension fixed even when a sensor is completely missing in a particular instance.
        # Zero is only a fallback for completely empty features.
        self.imputer = SimpleImputer(strategy="median",keep_empty_features=True, fill_value=0)
        
    def get_feature_columns(self,df: pd.DataFrame) -> list[str]:
        excluded = self.DROP_COLUMNS | self.LABEL_COLUMNS

        return [
            column
            for column in df.columns
            if column not in excluded
        ] 
    
    def handle_missing_values(self,df: pd.DataFrame,feature_columns: list[str]) -> pd.DataFrame:
        features = df[feature_columns].copy()
        features = features.ffill()
        features = features.bfill() 

        return features
    
    def fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = self.get_feature_columns(df)
        features = self.handle_missing_values(df,self.feature_columns)
        
        self.imputer.fit(features)
        imputed = self.imputer.transform(features)
        
        self.scaler.fit(imputed)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.handle_missing_values(df,self.feature_columns)
        imputed = self.imputer.transform(features)
        scaled = self.scaler.transform(imputed)
        return pd.DataFrame(
            scaled,
            index=df.index,
            columns=self.feature_columns,
        )
        
