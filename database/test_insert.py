from datetime import datetime, timezone
from connection import ConnectionDB
from models import SensorReading

def main():

    db = ConnectionDB()

    session = db.SessionLocal()

    reading = SensorReading(
        well_id="WELL-001",

        timestamp=datetime.now(timezone.utc),

        received_at=datetime.now(timezone.utc),

        pressure=99.98,

        temperature=64.91,

        flow_rate=249.20,

        production_choke=80.68,

        gas_lift_choke=45.09,

        production_valve=1,

        dhsv=1,

        source="mqtt"
    )

    session.add(reading)

    session.commit()

    print("Reading saved!")

    session.close()


if __name__ == "__main__":
    main()