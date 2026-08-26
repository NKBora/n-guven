from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.core.config import settings


AnalysisId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=settings.max_analysis_id_length,
    ),
]
AnalysisText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=settings.max_text_length,
    ),
]


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"
    HIGH = "HIGH"
    UNAVAILABLE = "UNAVAILABLE"


class ApiContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )


class TextAnalysisRequest(ApiContract):
    analysis_id: AnalysisId = Field(alias="analysisId")
    text: AnalysisText


class TextAnalysisResponse(ApiContract):
    analysis_id: AnalysisId = Field(alias="analysisId")
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = Field(alias="confidenceLevel")
    model_version: str = Field(alias="modelVersion", min_length=1)
    threshold_version: str = Field(alias="thresholdVersion", min_length=1)
    inference_ms: int = Field(alias="inferenceMs", ge=0)
    explanation: str = Field(min_length=1)
