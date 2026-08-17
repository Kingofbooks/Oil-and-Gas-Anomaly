import json
import time
from pathlib import Path
import pandas as pd
import paho.mqtt.client as mqtt
from config import Settings

FEATURE_COLUMNS = [
    "ABER-CKGL", "ABER-CKP", "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2",
    "ESTADO-PXO", "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2",
    "ESTADO-XO", "P-ANULAR", "P-JUS-CKGL", "P-JUS-CKP", "P-MON-CKP",
    "P-PDG", "P-TPT", "QGL", "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT",
]


class MqttPublisher:
    def __init__(self):
        self.settings = Settings()
        # self.parquet_path = Path(
        #     r"F:\Ecrio_Company!\sample_project_1- Oil & Gas"
        #     r"\datasets\3w_dataset\0"
        #     r"\WELL-00001_20170201010207.parquet"
        # )
        self.parquet_path = Path(
            r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset\1\DRAWN_00001.parquet"
        )
        self.topic = "oilgas/WELL-001/sensors"
        self.interval = 0.05

    def connect(self):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(
            self.settings.mqtt_broker_host,
            self.settings.mqtt_broker_port,
        )
        client.loop_start()
        return client

    def load_data(self):
        print(f"\n{'='*70}\nLOADING 3W DATA | File: {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        print(f"Dataset Loaded | Rows: {len(df)} | Columns: {len(df.columns)}\n{'='*70}")
        return df

    def create_message(self, row):
        data = {
            "timestamp": row.name.isoformat(),
            "operating_phase": "production",
            "ground_truth_anomaly": (
                0 if pd.isna(row["class"]) else int(row["class"] != 0)
            ),
        }
        for feature in FEATURE_COLUMNS:
            value = row[feature]
            if pd.isna(value):
                data[feature] = None
            else:
                data[feature] = float(value)
        return data

    def stream(self, df, client):
        print(f"\n{'='*70}\nSTARTING REAL 3W MQTT STREAM\n{'='*70}")
        for index, row in df.iterrows():
            data = self.create_message(row)
            payload = json.dumps(data)
            result = client.publish(self.topic, payload, qos=0)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"Publish failed with code: {result.rc}")
                continue
            print(f"Published Index: {index} | Anomaly: {data['ground_truth_anomaly']}")
            time.sleep(self.interval)

    def disconnect(self, client):
        client.loop_stop()
        client.disconnect()


def main():
    publisher = MqttPublisher()
    client = None
    try:
        client = publisher.connect()
        df = publisher.load_data()
        publisher.stream(df, client)
    except KeyboardInterrupt:
        print("\nPublisher stopped by user.")
    finally:
        if client:
            publisher.disconnect(client)


if __name__ == "__main__":
    main()