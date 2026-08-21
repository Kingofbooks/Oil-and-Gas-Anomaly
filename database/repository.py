from datetime import datetime, timezone

from database.connection import ConnectionDB
from database.models import SensorReading, AnomalyResult, Alert
from datetime import datetime, timezone
class SensorRepository:

    def __init__(self, db: ConnectionDB):
        self.db = db

    def save_sensor_reading(self, data):
        session = self.db.SessionLocal()
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

    def save_anomaly_result(self,reading_id: int,timestamp,model_name: str,model_version: str,anomaly_score: float,is_anomaly: bool):
        session = self.db.SessionLocal()
        try:
            result = AnomalyResult(
                reading_id=reading_id,
                timestamp=timestamp,
                model_name=model_name,
                model_version=model_version,
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                processed_at=datetime.now(timezone.utc),
            )

            session.add(result)
            session.commit()
            session.refresh(result)

            return result

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()
    
    def save_alert(self,anomaly_result_id: int,well_id: str,severity: str,message: str):
        session = self.db.SessionLocal()
        try:
            alert = Alert(
                anomaly_result_id=anomaly_result_id,
                well_id=well_id,
                created_at=datetime.now(timezone.utc),
                severity=severity,
                message=message,
                status="OPEN",
                resolved_at=None,
            )

            session.add(alert)
            session.commit()
            session.refresh(alert)

            return alert

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()
    
    def get_active_alert(self, well_id: str):
        session = self.db.SessionLocal()

        try:
            alert = (
                session.query(Alert)
                .filter(
                    Alert.well_id == well_id,
                    Alert.status == "OPEN",
                )
                .order_by(Alert.created_at.desc())
                .first()
            )

            return alert

        finally:
            session.close()