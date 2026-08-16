from datetime import datetime, timezone
from database.connection import ConnectionDB
from database.models import SensorReading

class SensorRepository:

    def __init__(self, db: ConnectionDB):
        self.db = db
    
    def save_sensor_reading(self, data):
        session=self.db.SessionLocal()
        
        try:
            reading = SensorReading(**data)
            session.add(reading)
            session.commit()
            session.refresh(reading)
            return reading
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        