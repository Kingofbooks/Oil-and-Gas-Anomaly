# import json
# import time
# from pathlib import Path
# import pandas as pd
# import paho.mqtt.client as mqtt
# from config import Settings

# FEATURE_COLUMNS = [
#     "ABER-CKGL", "ABER-CKP", "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2",
#     "ESTADO-PXO", "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2",
#     "ESTADO-XO", "P-ANULAR", "P-JUS-CKGL", "P-JUS-CKP", "P-MON-CKP",
#     "P-PDG", "P-TPT", "QGL", "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT",
# ]


# class MqttPublisher:
#     def __init__(self):
#         self.settings = Settings()
#         # self.parquet_path = Path(
#         #     r"F:\Ecrio_Company!\sample_project_1- Oil & Gas"
#         #     r"\datasets\3w_dataset\0"
#         #     r"\WELL-00001_20170201010207.parquet"
#         # )
#         self.parquet_path = Path(
#             r"F:\Ecrio_Company!\sample_project_1- Oil & Gas\datasets\3w_dataset\1\DRAWN_00001.parquet"
#         )
#         self.topic = "oilgas/WELL-001/sensors"
#         self.interval = 0.05

#     def connect(self):
#         client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
#         client.connect(
#             self.settings.mqtt_broker_host,
#             self.settings.mqtt_broker_port,
#         )
#         client.loop_start()
#         return client

#     def load_data(self):
#         print(f"\n{'='*70}\nLOADING 3W DATA | File: {self.parquet_path}")
#         df = pd.read_parquet(self.parquet_path)
#         print(f"Dataset Loaded | Rows: {len(df)} | Columns: {len(df.columns)}\n{'='*70}")
#         return df

#     def create_message(self, row):
#         data = {
#             "timestamp": row.name.isoformat(),
#             "operating_phase": "production",
#             "ground_truth_anomaly": (
#                 0 if pd.isna(row["class"]) else int(row["class"] != 0)
#             ),
#         }
#         for feature in FEATURE_COLUMNS:
#             value = row[feature]
#             if pd.isna(value):
#                 data[feature] = None
#             else:
#                 data[feature] = float(value)
#         return data

#     def stream(self, df, client):
#         print(f"\n{'='*70}\nSTARTING REAL 3W MQTT STREAM\n{'='*70}")
#         for index, row in df.iterrows():
#             data = self.create_message(row)
#             payload = json.dumps(data)
#             result = client.publish(self.topic, payload, qos=0)
#             if result.rc != mqtt.MQTT_ERR_SUCCESS:
#                 print(f"Publish failed with code: {result.rc}")
#                 continue
#             print(f"Published Index: {index} | Anomaly: {data['ground_truth_anomaly']}")
#             time.sleep(self.interval)

#     def disconnect(self, client):
#         client.loop_stop()
#         client.disconnect()


# def main():
#     publisher = MqttPublisher()
#     client = None
#     try:
#         client = publisher.connect()
#         df = publisher.load_data()
#         publisher.stream(df, client)
#     except KeyboardInterrupt:
#         print("\nPublisher stopped by user.")
#     finally:
#         if client:
#             publisher.disconnect(client)


# if __name__ == "__main__":
#     main()

import json
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt

from config import Settings


# ============================================================
# CONFIG   
# ============================================================

DEMO_DATA_PATH = Path(
    r"F:\Ecrio_Company!\sample_project_1- Oil & Gas"
    r"\datasets\demo\oil_gas_demo_22features.csv"
)

TOPIC = "oilgas/WELL-001/sensors"
1
PUBLISH_INTERVAL = 0.05

START_ROW = 0
END_ROW = 30000

# ============================================================
# 22 ML FEATURES
# ============================================================

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
# MQTT PUBLISHER
# ============================================================

class MqttPublisher:

    def __init__(self):

        self.settings = Settings()

        self.data_path = DEMO_DATA_PATH

        self.topic = TOPIC

        self.interval = PUBLISH_INTERVAL

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    def connect(self):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CONNECTING TO MQTT BROKER"
        )

        print(
            "=" * 70
        )

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2
        )

        client.connect(
            self.settings.mqtt_broker_host,
            self.settings.mqtt_broker_port,
        )

        client.loop_start()

        print(
            "Connected to:",
            self.settings.mqtt_broker_host,
            self.settings.mqtt_broker_port,
        )

        print(
            "Topic:",
            self.topic,
        )

        return client

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    def load_data(self):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "LOADING DEMO SENSOR DATA"
        )

        print(
            "=" * 70
        )

        print(
            "File:",
            self.data_path,
        )

        if not self.data_path.exists():

            raise FileNotFoundError(
                f"Demo dataset not found:\n"
                f"{self.data_path}"
            )

        df = pd.read_csv(
            self.data_path
        )

        print(
            "Rows:",
            len(df),
        )

        print(
            "Columns:",
            len(df.columns),
        )

        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing ML features:\n"
                f"{missing_features}"
            )

        print(
            "All 22 ML features present."
        )

        return df

    # --------------------------------------------------------
    # CREATE MQTT MESSAGE
    # --------------------------------------------------------

    def create_message(
        self,
        row,
    ):

        # Timestamp

        timestamp = row["timestamp"]

        if pd.isna(timestamp):

            raise ValueError(
                "Row contains invalid timestamp."
            )

        timestamp = pd.to_datetime(
            timestamp
        )

        # Ground truth is useful for
        # validating the demo.
        #
        # It is NOT sent to the ML model.

        if "class" in row.index:

            if pd.isna(
                row["class"]
            ):

                ground_truth = 0

            else:

                ground_truth = int(
                    int(row["class"]) != 0
                )

        else:

            ground_truth = 0

        data = {

            "timestamp":
                timestamp.isoformat(),

            "operating_phase":
                "production",

            "ground_truth_anomaly":
                ground_truth,
        }

        # Add the 22 ML features

        for feature in FEATURE_COLUMNS:

            value = row[feature]

            if pd.isna(value):

                data[feature] = None

            else:

                data[feature] = float(
                    value
                )

        return data

    # --------------------------------------------------------
    # STREAM DATA
    # --------------------------------------------------------

    def stream(
        self,
        df,
        client,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STARTING DEMO OIL & GAS MQTT STREAM"
        )

        print(
            "=" * 70
        )

        print(
            "Rows:",
            len(df),
        )

        print(
            "Interval:",
            self.interval,
            "seconds",
        )

        print(
            "\nPress CTRL+C to stop.\n"
        )
        
        df = df.iloc[START_ROW:END_ROW]

        for index, row in df.iterrows():

            try:

                data = self.create_message(
                    row
                )

                payload = json.dumps(
                    data
                )

                result = client.publish(
                    self.topic,
                    payload,
                    qos=1,
                )

                if (
                    result.rc
                    != mqtt.MQTT_ERR_SUCCESS
                ):

                    print(
                        "Publish failed:",
                        result.rc,
                    )

                    continue

                print(
                    f"Published row {index:>6} | "
                    f"timestamp={data['timestamp']} | "
                    f"ground_truth={data['ground_truth_anomaly']}"
                )

                time.sleep(
                    self.interval
                )

            except Exception as e:

                print(
                    f"Failed to publish row "
                    f"{index}: {e}"
                )

    # --------------------------------------------------------
    # DISCONNECT
    # --------------------------------------------------------

    def disconnect(
        self,
        client,
    ):

        if client is None:

            return

        client.loop_stop()

        client.disconnect()

        print(
            "\nPublisher disconnected."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    publisher = MqttPublisher()

    client = None

    try:

        client = publisher.connect()

        df = publisher.load_data()

        publisher.stream(
            df,
            client,
        )

    except KeyboardInterrupt:

        print(
            "\nPublisher stopped by user."
        )

    finally:

        publisher.disconnect(
            client
        )


if __name__ == "__main__":

    main()