import json
import paho.mqtt.client as mqtt


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
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2
        )
        self.client.on_message = self.on_message
        self.client.connect("localhost",1883)
        self.client.subscribe("oilgas/WELL-001/sensors")

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

        return True

    def process_reading(self, data):
        print("Valid sensor reading")
        print("Pressure:", data["pressure"])
        print("Temperature:", data["temperature"])
        print("Flow rate:", data["flow_rate"])

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