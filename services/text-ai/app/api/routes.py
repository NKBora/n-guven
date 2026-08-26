from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.inference.service import (
    TextInferenceService,
    get_text_inference_service,
)
from app.schemas.analysis import TextAnalysisRequest, TextAnalysisResponse


router = APIRouter(prefix="/v1", tags=["text-analysis"])


@router.post(
    "/analyze/text",
    response_model=TextAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a Turkish text AI-generation signal",
)
async def analyze_text(
    request: TextAnalysisRequest,
    inference_service: Annotated[
        TextInferenceService,
        Depends(get_text_inference_service),
    ],
) -> TextAnalysisResponse:
    return await inference_service.analyze(request)
