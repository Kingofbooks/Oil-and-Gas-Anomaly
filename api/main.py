from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.alerts import router as alerts_router
from api.routes.readings import router as readings_router
from api.routes.wells import router as wells_router
from api.routes.anomalies import router as anomalies_router
from api.routes.dashboard import router as dashboard_router
from api.routes.predict import router as predict_router


app = FastAPI(
    title="Oil and Gas Anomaly API",
    version="0.1.0",
    description="API for sensor readings, wells, anomalies, alerts and ML prediction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wells_router)
app.include_router(readings_router)
app.include_router(anomalies_router)
app.include_router(alerts_router)
app.include_router(dashboard_router)
app.include_router(predict_router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}