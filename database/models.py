from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import BigInteger, String, Float, SmallInteger, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

class Base(DeclarativeBase):
    pass


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(BigInteger,primary_key=True)

    well_id: Mapped[str] = mapped_column(String(50),nullable=False)
    
    pressure: Mapped[float] = mapped_column(Float,nullable=False)

    temperature: Mapped[float] = mapped_column(Float,nullable=False)

    flow_rate: Mapped[float] = mapped_column(Float,nullable=False)

    production_choke: Mapped[float] = mapped_column(Float,nullable=False)

    gas_lift_choke: Mapped[float] = mapped_column(Float,nullable=False)
    
    production_valve: Mapped[int] = mapped_column(SmallInteger,nullable=False)
    
    dhsv: Mapped[int] = mapped_column(SmallInteger,nullable=False)
    
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True),nullable=False)
    
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),nullable=False)
    
    source: Mapped[str] = mapped_column(String(20),nullable=False)


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"

    id: Mapped[int] = mapped_column(BigInteger,primary_key=True)

    reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_readings.id"),nullable=False)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True),nullable=False)

    model_name: Mapped[str] = mapped_column(String(100),nullable=False)

    model_version: Mapped[str] = mapped_column(String(50),nullable=False)

    anomaly_score: Mapped[float] = mapped_column(Float,nullable=False)

    is_anomaly: Mapped[bool] = mapped_column(nullable=False)

    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),nullable=False)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigInteger,primary_key=True)

    anomaly_result_id: Mapped[int] = mapped_column(ForeignKey("anomaly_results.id"),nullable=False)

    well_id: Mapped[str] = mapped_column(String(50),nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),nullable=False)

    severity: Mapped[str] = mapped_column(String(20),nullable=False)

    message: Mapped[str] = mapped_column(String,nullable=False)

    status: Mapped[str] = mapped_column(String(20),nullable=False)

    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),nullable=True)