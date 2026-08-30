from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.inference.berturk import (
    BerturkTextInferenceService,
    ReleaseVerificationError,
    load_verified_berturk_release,
    preprocess_turkish_text,
    temperature_scale,
)
from app.inference.service import get_text_inference_service
from app.main import app
from app.schemas.analysis import TextAnalysisRequest


UPSTREAM_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"


class RecordingPredictor:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.inputs: list[str] = []

    def predict_synthetic_probability(self, text: str) -> float:
        self.inputs.append(text)
        return self.probability


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _release_bundle(tmp_path: Path) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    weights = artifact_root / "model.safetensors"
    weights.write_bytes(b"safe-test-weights")

    manifest = {
        "modelId": "berturk-text-origin-v1-seed-17",
        "adapterId": "berturk",
        "preprocessingVersion": "tr-text-v1",
        "fineTuning": {"seed": 17},
        "runtime": {
            "artifactFormat": "safetensors",
            "maxSequenceLength": 128,
        },
        "upstream": {
            "repository": "dbmdz/bert-base-turkish-cased",
            "revision": UPSTREAM_REVISION,
        },
        "labels": {"0": "human", "1": "synthetic"},
        "artifacts": [
            {
                "path": "model.safetensors",
                "role": "weights",
                "sha256": _sha256(weights),
                "sizeBytes": weights.stat().st_size,
            }
        ],
    }
    manifest_path = artifact_root / "model-manifest.json"
    _write_json(manifest_path, manifest)

    calibration = {
        "method": "temperature-scaling",
        "fittedSplit": "validation",
        "model": {"name": "berturk", "version": "text-origin-tr-v1"},
        "temperature": 1.0,
        "artifacts": {
            "manifestSha256": "a" * 64,
            "predictionsSha256": "b" * 64,
        },
    }
    calibration_path = artifact_root / "calibration.json"
    _write_json(calibration_path, calibration)

    release = {
        "releaseId": "berturk-text-origin-v1",
        "stage": "prototype",
        "model": {
            "modelId": manifest["modelId"],
            "seed": 17,
            "manifestSha256": _sha256(manifest_path),
            "weightsSha256": _sha256(weights),
            "weightsSizeBytes": weights.stat().st_size,
        },
        "calibration": {
            "temperature": 1.0,
            "artifactSha256": _sha256(calibration_path),
            "validationManifestSha256": "a" * 64,
            "validationPredictionsSha256": "b" * 64,
        },
        "thresholds": {
            "version": "text-origin-thresholds-v1",
            "lowMaximum": 0.2,
            "highMinimum": 0.8,
        },
        "artifactLayout": {
            "modelManifest": "model-manifest.json",
            "calibration": "calibration.json",
        },
    }
    release_path = tmp_path / "release.json"
    _write_json(release_path, release)
    return artifact_root, release_path


def test_loads_hash_verified_berturk_release(tmp_path: Path) -> None:
    artifact_root, release_path = _release_bundle(tmp_path)

    release = load_verified_berturk_release(
        artifact_root,
        release_path=release_path,
    )

    assert release.model_id == "berturk-text-origin-v1-seed-17"
    assert release.threshold_version == "text-origin-thresholds-v1"
    assert release.labels == {"0": "human", "1": "synthetic"}
    assert release.max_sequence_length == 128


def test_rejects_tampered_weights(tmp_path: Path) -> None:
    artifact_root, release_path = _release_bundle(tmp_path)
    (artifact_root / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(ReleaseVerificationError, match="hash mismatch"):
        load_verified_berturk_release(artifact_root, release_path=release_path)


def test_rejects_tampered_calibration(tmp_path: Path) -> None:
    artifact_root, release_path = _release_bundle(tmp_path)
    (artifact_root / "calibration.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ReleaseVerificationError, match="Calibration artifact hash"):
        load_verified_berturk_release(artifact_root, release_path=release_path)


def test_rejects_symbolic_link_artifact(tmp_path: Path) -> None:
    artifact_root, release_path = _release_bundle(tmp_path)
    weights = artifact_root / "model.safetensors"
    target = tmp_path / "outside.safetensors"
    target.write_bytes(weights.read_bytes())
    weights.unlink()
    try:
        weights.symlink_to(target)
    except OSError:
        pytest.skip("Symbolic links are unavailable on this platform")

    with pytest.raises(ReleaseVerificationError, match="symbolic link"):
        load_verified_berturk_release(artifact_root, release_path=release_path)


@pytest.mark.parametrize(
    ("probability", "expected_level"),
    [(0.1, "LOW"), (0.5, "UNCERTAIN"), (0.9, "HIGH")],
)
def test_service_applies_calibration_and_thresholds(
    tmp_path: Path,
    probability: float,
    expected_level: str,
) -> None:
    artifact_root, release_path = _release_bundle(tmp_path)
    release = load_verified_berturk_release(
        artifact_root,
        release_path=release_path,
    )
    predictor = RecordingPredictor(probability)
    service = BerturkTextInferenceService(release, predictor)
    request = TextAnalysisRequest(
        analysisId="analysis-001",
        text="Türkçe örnek metin",
    )

    response = asyncio.run(service.analyze(request))

    assert math.isclose(response.score or 0.0, probability)
    assert response.confidence_level.value == expected_level
    assert response.model_version == "berturk-text-origin-v1-seed-17"
    assert response.threshold_version == "text-origin-thresholds-v1"
    assert response.inference_ms >= 0


def test_api_preserves_and_preprocesses_text_before_inference(tmp_path: Path) -> None:
    artifact_root, release_path = _release_bundle(tmp_path)
    release = load_verified_berturk_release(
        artifact_root,
        release_path=release_path,
    )
    predictor = RecordingPredictor(0.9)
    service = BerturkTextInferenceService(release, predictor)
    app.dependency_overrides[get_text_inference_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/analyze/text",
                json={
                    "analysisId": "analysis-002",
                    "text": "  \ufeffTu\u0308rkc\u0327e\r\nmetin  ",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["confidenceLevel"] == "HIGH"
    assert predictor.inputs == ["  \ufeffTürkçe\nmetin  "]


def test_preprocessing_matches_tr_text_v1_contract() -> None:
    assert preprocess_turkish_text("\ufeffTu\u0308rkc\u0327e\rmetin") == "Türkçe\nmetin"


def test_api_rejects_nul_before_inference() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze/text",
            json={"analysisId": "analysis-003", "text": "güvensiz\u0000metin"},
        )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("probability", "temperature"),
    [(float("nan"), 1.0), (-0.1, 1.0), (1.1, 1.0), (0.5, 0.01)],
)
def test_temperature_scaling_rejects_invalid_inputs(
    probability: float,
    temperature: float,
) -> None:
    with pytest.raises(ValueError):
        temperature_scale(probability, temperature)
