from datetime import datetime, timezone

from connection import ConnectionDB
from repository import SensorRepository


def main():

    db = ConnectionDB()

    repository = SensorRepository(db)

    data = {
        "well_id": "WELL-001",

        "timestamp": datetime.now(timezone.utc),

        "received_at": datetime.now(timezone.utc),

        "pressure": 100.25,

        "temperature": 65.05,

        "flow_rate": 249.94,

        "production_choke": 79.25,

        "gas_lift_choke": 45.28,

        "production_valve": 1,

        "dhsv": 1,

        "source": "mqtt"
    }

    reading = repository.save_sensor_reading(data)

    print("Reading saved!")
    print("ID:", reading.id)
    print("Pressure:", reading.pressure)


if __name__ == "__main__":
    main()