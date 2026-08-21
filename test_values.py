import pandas as pd
import json
from pathlib import Path

# CSV = "datasets/demo/oil_gas_demo_22features.csv"
CSV = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\demo\oil_gas_demo_22features.csv"
)
FEATURES = [
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

df = pd.read_csv(CSV)

window = df.iloc[6200:6320][FEATURES]

payload = {
    "readings": window.astype(float).values.tolist()
}

with open("anomaly_predict.json", "w") as f:
    json.dump(payload, f)

print("Created anomaly_predict.json")
print("Shape:", window.shape)