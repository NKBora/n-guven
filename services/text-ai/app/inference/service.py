from functools import lru_cache
from typing import Protocol

from app.core.config import settings
from app.inference.berturk import (
    BerturkTextInferenceService,
    LocalBerturkPredictor,
    load_verified_berturk_release,
)
from app.schemas.analysis import (
    ConfidenceLevel,
    TextAnalysisRequest,
    TextAnalysisResponse,
)


class TextInferenceService(Protocol):
    async def analyze(self, request: TextAnalysisRequest) -> TextAnalysisResponse:
        """Return a model-independent analysis contract."""
        ...


class NotConfiguredTextInferenceService:
    """Deterministic service used until a validated model adapter is configured."""

    async def analyze(self, request: TextAnalysisRequest) -> TextAnalysisResponse:
        return TextAnalysisResponse(
            analysis_id=request.analysis_id,
            score=None,
            confidence_level=ConfidenceLevel.UNAVAILABLE,
            model_version="not-configured",
            threshold_version="not-configured",
            inference_ms=0,
            explanation="Model inference is not configured yet.",
        )


@lru_cache(maxsize=1)
def get_text_inference_service() -> TextInferenceService:
    if settings.model_root is None:
        return NotConfiguredTextInferenceService()
    release = load_verified_berturk_release(settings.model_root)
    return BerturkTextInferenceService(release, LocalBerturkPredictor(release))
