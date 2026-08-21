from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from database.connection import ConnectionDB
from database.models import AnomalyResult, SensorReading


router = APIRouter(
    prefix="/anomalies",
    tags=["anomalies"],
)


class AnomalyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reading_id: int
    timestamp: datetime
    model_name: str
    model_version: str
    anomaly_score: float
    is_anomaly: bool
    processed_at: datetime


@router.get("", response_model=list[AnomalyResponse])
def list_anomalies(
    well_id: str | None = None,
    is_anomaly: bool | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[AnomalyResult]:

    db = ConnectionDB()
    session = db.SessionLocal()

    try:
        query = (
            session.query(AnomalyResult)
            .join(
                SensorReading,
                AnomalyResult.reading_id == SensorReading.id,
            )
            .order_by(AnomalyResult.timestamp.desc())
        )

        if well_id:
            query = query.filter(
                SensorReading.well_id == well_id
            )

        if is_anomaly is not None:
            query = query.filter(
                AnomalyResult.is_anomaly == is_anomaly
            )

        return query.limit(limit).all()

    finally:
        session.close()


@router.get("/latest", response_model=list[AnomalyResponse])
def latest_anomalies(
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[AnomalyResult]:

    db = ConnectionDB()
    session = db.SessionLocal()

    try:
        return (
            session.query(AnomalyResult)
            .order_by(AnomalyResult.timestamp.desc())
            .limit(limit)
            .all()
        )

    finally:
        session.close()


@router.get("/{anomaly_id}", response_model=AnomalyResponse)
def get_anomaly(anomaly_id: int) -> AnomalyResult:

    db = ConnectionDB()
    session = db.SessionLocal()

    try:
        anomaly = session.get(
            AnomalyResult,
            anomaly_id,
        )

        if anomaly is None:
            raise HTTPException(
                status_code=404,
                detail="Anomaly result not found",
            )

        return anomaly

    finally:
        session.close()