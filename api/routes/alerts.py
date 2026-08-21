from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.schemas import AlertResponse, AlertResolveRequest
from database.connection import ConnectionDB
from database.models import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertResponse])
def list_alerts(
    well_id: str | None = None,
    status: str | None = Query(default=None, pattern="^[A-Za-z_]+$"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[Alert]:
    db = ConnectionDB()
    session = db.SessionLocal()
    try:
        query = session.query(Alert).order_by(Alert.created_at.desc())
        if well_id:
            query = query.filter(Alert.well_id == well_id)
        if status:
            query = query.filter(Alert.status == status.upper())
        return query.limit(limit).all()
    finally:
        session.close()


@router.patch("/{alert_id}", response_model=AlertResponse)
def resolve_alert(alert_id: int, update: AlertResolveRequest) -> Alert:
    db = ConnectionDB()
    session = db.SessionLocal()
    try:
        alert = session.get(Alert, alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        alert.status = update.status
        alert.resolved_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(alert)
        return alert
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
