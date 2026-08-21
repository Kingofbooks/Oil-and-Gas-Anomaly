from fastapi import APIRouter

from database.connection import ConnectionDB
from database.models import SensorReading


router = APIRouter(
    prefix="/wells",
    tags=["wells"],
)


@router.get("", response_model=list[str])
def list_wells() -> list[str]:

    db = ConnectionDB()
    session = db.SessionLocal()

    try:
        wells = (
            session.query(SensorReading.well_id)
            .distinct()
            .order_by(SensorReading.well_id)
            .all()
        )

        return [well[0] for well in wells]

    finally:
        session.close()