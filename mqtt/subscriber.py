import json
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from database.connection import ConnectionDB
from database.repository import SensorRepository
from config import Settings
REQUIRED_FIELDS = {
    "timestamp",
    "operating_phase",
    "ground_truth_anomaly",
    "pressure",
    "temperature",
    "flow_rate",
    "production_choke",
    "gas_lift_choke",
    "production_valve",
    "dhsv",
}


NUMERIC_FIELDS = {
    "pressure",
    "temperature",
    "flow_rate",
    "production_choke",
    "gas_lift_choke",
    "production_valve",
    "dhsv",
}


class MqttSubscriber:

    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = self.on_message
        self.settings = Settings()
        self.client.connect(
            self.settings.mqtt_broker_host,
            self.settings.mqtt_broker_port
        )
        self.subscribe_data = self.settings.mqtt_topic
        self.client.subscribe(self.subscribe_data)
        self.db = ConnectionDB()
        self.repository = SensorRepository(self.db)

    def decode_message(self, payload):
        text = payload.decode()
        data = json.loads(text)

        return data

    def validate_reading(self, data):
        missing_fields = REQUIRED_FIELDS - data.keys()

        if missing_fields:
            print("Missing fields:", missing_fields)
            return False

        for field in NUMERIC_FIELDS:
            if not isinstance(data[field], (int, float)):
                print(f"{field} must be numeric")
                return False

        if data["pressure"] < 0:
            print("Pressure cannot be negative")
            return False

        if not 0 <= data["production_choke"] <= 100:
            print("production_choke must be between 0 and 100")
            return False

        if not 0 <= data["gas_lift_choke"] <= 100:
            print("gas_lift_choke must be between 0 and 100")
            return False

        if data["production_valve"] not in (0, 1):
            print("production_valve must be 0 or 1")
            return False

        if data["dhsv"] not in (0, 1):
            print("dhsv must be 0 or 1")
            return False

        if not isinstance(data["operating_phase"], str):
            print("operating_phase must be a string")
            return False
        
        if data["ground_truth_anomaly"] not in (0, 1):
            print("ground_truth_anomaly must be 0 or 1")
            return False
        
        try:
            datetime.fromisoformat(data["timestamp"])
        except ValueError:
            print("timestamp must be a valid ISO datetime")
            return False

        return True
    def prepare_reading(self, data):
        well_id = self.subscribe_data.split("/")[1]
        timestamp = datetime.fromisoformat(data["timestamp"]).replace(tzinfo=timezone.utc)
        reading = {
            "well_id": well_id,

            "timestamp": timestamp,

            "received_at": datetime.now(timezone.utc),

            "pressure": data["pressure"],
            "temperature": data["temperature"],
            "flow_rate": data["flow_rate"],

            "production_choke": data["production_choke"],
            "gas_lift_choke": data["gas_lift_choke"],

            "production_valve": data["production_valve"],
            "dhsv": data["dhsv"],

            "source": "mqtt"
        }

        return reading
    
    def process_reading(self, data):
        reading_data = self.prepare_reading(data)

        reading = self.repository.save_sensor_reading(reading_data)

        print("Sensor reading saved")
        print("Reading ID:", reading.id)

    def on_message(self, client, userdata, message):
        try:
            data = self.decode_message(message.payload)

            if not self.validate_reading(data):
                print("Rejected sensor reading")
                return

            self.process_reading(data)

        except Exception as e:
            print("Failed to process message:", e)

    def run(self):
        self.client.loop_forever()


def main():
    sub = MqttSubscriber()
    sub.run()


if __name__ == "__main__":
    main()