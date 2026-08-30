from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.schemas.health import HealthResponse


app = FastAPI(
    title="N-Güven Text AI Service",
    summary="Turkish text AI-generation signal service foundation",
    description=(
        "Calibrated BERTurk prototype signal for Turkish synthetic-text analysis. "
        "The service returns an advisory signal, not proof of authorship."
    ),
    version="0.2.0",
)
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name)
