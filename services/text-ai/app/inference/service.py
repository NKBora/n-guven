from typing import Protocol

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


_text_inference_service: TextInferenceService = NotConfiguredTextInferenceService()


def get_text_inference_service() -> TextInferenceService:
    return _text_inference_service
