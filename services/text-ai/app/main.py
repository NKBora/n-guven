from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.schemas.health import HealthResponse


app = FastAPI(
    title="N-Güven Text AI Service",
    summary="Turkish text AI-generation signal service foundation",
    description=(
        "Stable HTTP contracts for Turkish text synthetic-signal analysis. "
        "No ML model is loaded in the current service foundation."
    ),
    version="0.1.0",
)
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name)
