import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import torch

from config import Settings
from database.connection import ConnectionDB
from database.repository import SensorRepository
from ml.dataset import ThreeWDataset
from ml.split import split_real_instances
from ml.preprocessing import Preprocessor
from ml.tranad_detector import TranADDetector

DATASET_ROOT = Path(r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset")
MODEL_PATH = Path("artifacts/tranad_v1.pt")
WINDOW_SIZE = 120
THRESHOLD = 0.029617823120545266
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_MAPPING = {
    "ABER-CKGL": "ABER_CKGL", "ABER-CKP": "ABER_CKP", "ESTADO-DHSV": "ESTADO_DHSV",
    "ESTADO-M1": "ESTADO_M1", "ESTADO-M2": "ESTADO_M2", "ESTADO-PXO": "ESTADO_PXO",
    "ESTADO-SDV-GL": "ESTADO_SDV_GL", "ESTADO-SDV-P": "ESTADO_SDV_P", "ESTADO-W1": "ESTADO_W1",
    "ESTADO-W2": "ESTADO_W2", "ESTADO-XO": "ESTADO_XO", "P-ANULAR": "P_ANULAR",
    "P-JUS-CKGL": "P_JUS_CKGL", "P-JUS-CKP": "P_JUS_CKP", "P-MON-CKP": "P_MON_CKP",
    "P-PDG": "P_PDG", "P-TPT": "P_TPT", "QGL": "QGL",
    "T-JUS-CKP": "T_JUS_CKP", "T-MON-CKP": "T_MON_CKP", "T-PDG": "T_PDG", "T-TPT": "T_TPT",
}

FEATURE_COLUMNS = list(FEATURE_MAPPING.keys())
REQUIRED_FIELDS = {"timestamp", *FEATURE_COLUMNS}


class MqttSubscriber:
    def __init__(self):
        print(f"\n{'='*70}\nOIL & GAS ANOMALY DETECTOR | Device: {DEVICE}\n{'='*70}")
        self.settings = Settings()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.subscribe_topic = self.settings.mqtt_topic
        self.db = ConnectionDB()
        self.repository = SensorRepository(self.db)
        self.window = deque(maxlen=WINDOW_SIZE)

        print("\nPreparing ML preprocessor...")
        self.preprocessor = self.build_preprocessor()

        print("\nLoading TranAD...")
        self.detector = TranADDetector(
            model_path=MODEL_PATH,
            preprocessor=self.preprocessor,
            threshold=THRESHOLD,
            device=DEVICE,
        )
        print("\nSystem ready.")

    def build_preprocessor(self):
        dataset = ThreeWDataset(DATASET_ROOT)
        metadata = dataset.build_index()
        records = [
            {
                "path": item.path, "well_id": item.well_id, "event_class": item.folder_type,
                "source": item.source, "num_rows": item.num_rows, "start_time": item.start_time, "end_time": item.end_time,
            }
            for item in metadata
        ]
        metadata_df = pd.DataFrame(records)
        train_metadata, _, _ = split_real_instances(metadata_df)

        training_frames = []
        print("Collecting training data...")
        for row in train_metadata.itertuples(index=False):
            df = pd.read_parquet(row.path)
            normal = df[df["class"] == 0].copy()
            if not normal.empty:
                training_frames.append(normal)

        training_data = pd.concat(training_frames, axis=0)
        preprocessor = Preprocessor()
        preprocessor.fit(training_data)
        print(f"Preprocessor Fitted | Features: {len(preprocessor.feature_columns)}")
        return preprocessor

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print(f"\nConnected to MQTT Broker. Subscribing to: {self.subscribe_topic}")
            client.subscribe(self.subscribe_topic)
        else:
            print(f"MQTT connection failed with code: {reason_code}")

    def decode_message(self, payload):
        return json.loads(payload.decode("utf-8"))

    def validate_reading(self, data):
        missing_fields = REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            print(f"Missing fields: {sorted(missing_fields)}")
            return False

        try:
            datetime.fromisoformat(data["timestamp"])
        except (TypeError, ValueError):
            print("Invalid timestamp format")
            return False

        for feature in FEATURE_COLUMNS:
            value = data[feature]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                print(f"{feature} must be numeric or null")
                return False

        return True

    def prepare_reading(self, data, topic):
        topic_parts = topic.split("/")
        if len(topic_parts) != 3:
            raise ValueError(f"Unexpected MQTT topic structure: {topic}")

        well_id = topic_parts[1]
        timestamp = datetime.fromisoformat(data["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        reading = {
            "well_id": well_id,
            "timestamp": timestamp,
            "received_at": datetime.now(timezone.utc),
            "source": "mqtt",
        }
        for mqtt_name, db_name in FEATURE_MAPPING.items():
            value = data[mqtt_name]
            reading[db_name] = (0.0 if value is None else float(value))

        return reading

    def create_ml_row(self, data):
        return {
            feature: np.nan if data[feature] is None else float(data[feature])
            for feature in FEATURE_COLUMNS
        }

    def process_ml(self, data):
        row = self.create_ml_row(data)
        self.window.append(row)
        current_size = len(self.window)
        print(f"ML buffer status: {current_size}/{WINDOW_SIZE}")

        if current_size < WINDOW_SIZE:
            return

        window_df = pd.DataFrame(list(self.window))
        result = self.detector.detect(window_df)
        pred_label = "ANOMALY" if result.is_anomaly else "NORMAL"
        print(
            f"\n{'='*70}\nTRANAD RESULT | Model: {result.model_name} ({result.model_version}) | "
            f"Score: {result.anomaly_score:.6f} | Threshold: {THRESHOLD:.6f} | Prediction: {pred_label}\n{'='*70}"
        )

    def process_reading(self, data, topic):
        reading_data = self.prepare_reading(data, topic)
        reading = self.repository.save_sensor_reading(reading_data)
        print(f"Sensor reading saved | ID: {reading.id} | Well: {reading.well_id}")
        self.process_ml(data)

    def on_message(self, client, userdata, message):
        print(f"\n{'-'*70}\nReceived MQTT message on topic: {message.topic}")
        try:
            data = self.decode_message(message.payload)
            if not self.validate_reading(data):
                print("Rejected reading")
                return
            self.process_reading(data, message.topic)
        except Exception as e:
            print(f"Failed to process message: {e}")

    def run(self):
        print("\nWaiting for MQTT messages...")
        self.client.connect(
            self.settings.mqtt_broker_host,
            self.settings.mqtt_broker_port,
        )
        self.client.loop_forever()


def main():
    subscriber = MqttSubscriber()
    subscriber.run()


if __name__ == "__main__":
    main()