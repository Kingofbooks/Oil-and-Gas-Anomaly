from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SensorReadingCreate(BaseModel):
    well_id: str = Field(min_length=1, max_length=50)
    timestamp: datetime
    received_at: datetime
    source: str = Field(min_length=1, max_length=20)
    ABER_CKGL: float
    ABER_CKP: float
    ESTADO_DHSV: float
    ESTADO_M1: float
    ESTADO_M2: float
    ESTADO_PXO: float
    ESTADO_SDV_GL: float
    ESTADO_SDV_P: float
    ESTADO_W1: float
    ESTADO_W2: float
    ESTADO_XO: float
    P_ANULAR: float
    P_JUS_CKGL: float
    P_JUS_CKP: float
    P_MON_CKP: float
    P_PDG: float
    P_TPT: float
    QGL: float
    T_JUS_CKP: float
    T_MON_CKP: float
    T_PDG: float
    T_TPT: float


class SensorReadingResponse(SensorReadingCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    anomaly_result_id: int
    well_id: str
    created_at: datetime
    severity: str
    message: str
    status: str
    resolved_at: datetime | None = None


class AlertResolveRequest(BaseModel):
    status: str = Field(default="RESOLVED", pattern="^(RESOLVED|CLOSED)$")
