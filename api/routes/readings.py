from fastapi import APIRouter, Query, status

from api.schemas import SensorReadingCreate, SensorReadingResponse
from database.connection import ConnectionDB
from database.models import SensorReading
from database.repository import SensorRepository

router = APIRouter(prefix="/readings", tags=["readings"])


def _session():
    return ConnectionDB().SessionLocal()


@router.post("", response_model=SensorReadingResponse, status_code=status.HTTP_201_CREATED)
def create_reading(reading: SensorReadingCreate) -> SensorReading:
    return SensorRepository(ConnectionDB()).save_sensor_reading(reading.model_dump())


@router.get("", response_model=list[SensorReadingResponse])
def list_readings(
    well_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[SensorReading]:
    session = _session()
    try:
        query = session.query(SensorReading).order_by(SensorReading.timestamp.desc())
        if well_id:
            query = query.filter(SensorReading.well_id == well_id)
        return query.limit(limit).all()
    finally:
        session.close()
