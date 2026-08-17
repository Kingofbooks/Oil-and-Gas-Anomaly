from pathlib import Path

import pandas as pd


DATASET_ROOT = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas"
    r"\datasets\3w_dataset"
)

FILE = (
    DATASET_ROOT
    / "0"
    / "WELL-00001_20170201010207.parquet"
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


df = pd.read_parquet(FILE)

row = df.iloc[0]

print("=" * 70)
print("ORIGINAL PARQUET ROW")
print("=" * 70)

print(
    "Timestamp:",
    df.index[0],
)

for feature in FEATURE_COLUMNS:

    print(
        f"{feature:20s}: {row[feature]}"
    )