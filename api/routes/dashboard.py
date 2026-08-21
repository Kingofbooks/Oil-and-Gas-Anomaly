from fastapi import APIRouter

from sqlalchemy import func

from database.connection import ConnectionDB
from database.models import (
    SensorReading,
    AnomalyResult,
    Alert,
)


router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
)


@router.get("/summary")
def dashboard_summary() -> dict:

    db = ConnectionDB()
    session = db.SessionLocal()

    try:
        total_readings = (
            session.query(
                func.count(SensorReading.id)
            ).scalar()
        )

        total_anomalies = (
            session.query(
                func.count(AnomalyResult.id)
            )
            .filter(
                AnomalyResult.is_anomaly == True
            )
            .scalar()
        )

        active_alerts = (
            session.query(
                func.count(Alert.id)
            )
            .filter(
                Alert.status == "OPEN"
            )
            .scalar()
        )

        total_wells = (
            session.query(
                func.count(
                    func.distinct(
                        SensorReading.well_id
                    )
                )
            ).scalar()
        )

        return {
            "total_readings": total_readings or 0,
            "total_anomalies": total_anomalies or 0,
            "active_alerts": active_alerts or 0,
            "total_wells": total_wells or 0,
        }

    finally:
        session.close()