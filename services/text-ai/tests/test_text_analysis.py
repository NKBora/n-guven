import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


client = TestClient(app)


def test_analyze_text_returns_unavailable_contract() -> None:
    response = client.post(
        "/v1/analyze/text",
        json={
            "analysisId": "analysis-001",
            "text": "Bu analiz edilecek Türkçe metindir.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "analysisId": "analysis-001",
        "score": None,
        "confidenceLevel": "UNAVAILABLE",
        "modelVersion": "not-configured",
        "thresholdVersion": "not-configured",
        "inferenceMs": 0,
        "explanation": "Model inference is not configured yet.",
    }


@pytest.mark.parametrize("text", ["", "   "])
def test_analyze_text_rejects_blank_text(text: str) -> None:
    response = client.post(
        "/v1/analyze/text",
        json={"analysisId": "analysis-002", "text": text},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Geçerli metin"},
        {"analysisId": "", "text": "Geçerli metin"},
        {"analysisId": "   ", "text": "Geçerli metin"},
    ],
)
def test_analyze_text_rejects_missing_or_blank_analysis_id(
    payload: dict[str, str],
) -> None:
    response = client.post("/v1/analyze/text", json=payload)

    assert response.status_code == 422


def test_analyze_text_rejects_text_over_configured_limit() -> None:
    response = client.post(
        "/v1/analyze/text",
        json={
            "analysisId": "analysis-003",
            "text": "a" * (settings.max_text_length + 1),
        },
    )

    assert response.status_code == 422
