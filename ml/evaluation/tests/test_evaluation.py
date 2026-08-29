from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nguven_evaluation.evaluation import (
    EvaluationInputError,
    EvaluationMetadata,
    evaluate_predictions,
    load_predictions,
)
from nguven_evaluation.calibration import fit_temperature_calibration


def manifest_record(record_id: str, label: str) -> dict[str, object]:
    return {
        "id": record_id,
        "label": label,
    }


def prediction(record_id: str, label: str, inference_ms: float = 5.0) -> dict[str, object]:
    return {
        "id": record_id,
        "predictedLabel": label,
        "score": 0.8,
        "inferenceMs": inference_ms,
    }


def metadata() -> EvaluationMetadata:
    return EvaluationMetadata(
        run_id="synthetic-test-run",
        git_commit="abcdef1234567",
        seed=42,
        dataset_version="synthetic-fixture-v1",
        model_name="fake-adapter",
        model_version="test-only",
    )


def evaluate(
    manifest: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> dict[str, object]:
    return evaluate_predictions(
        manifest,
        predictions,
        metadata=metadata(),
        manifest_sha256="a" * 64,
        predictions_sha256="b" * 64,
        now=lambda: datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc),
    )


def test_load_predictions_validates_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    records = [prediction("a", "human"), prediction("b", "synthetic")]
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")

    assert load_predictions(path) == records


def test_load_predictions_rejects_score_outside_unit_interval(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    invalid = prediction("a", "human")
    invalid["score"] = 1.1
    path.write_text(json.dumps([invalid]), encoding="utf-8")

    with pytest.raises(EvaluationInputError, match="score"):
        load_predictions(path)


def test_evaluate_predictions_computes_macro_metrics() -> None:
    manifest = [
        manifest_record("a", "human"),
        manifest_record("b", "human"),
        manifest_record("c", "synthetic"),
        manifest_record("d", "synthetic"),
    ]
    predictions = [
        prediction("a", "human", 2),
        prediction("b", "synthetic", 4),
        prediction("c", "synthetic", 6),
        prediction("d", "synthetic", 8),
    ]

    result = evaluate(manifest, predictions)

    assert result["metrics"]["accuracy"] == pytest.approx(0.75)
    assert result["metrics"]["macroPrecision"] == pytest.approx(5 / 6)
    assert result["metrics"]["macroRecall"] == pytest.approx(0.75)
    assert result["metrics"]["macroF1"] == pytest.approx(11 / 15)
    assert result["metrics"]["meanInferenceMs"] == pytest.approx(5.0)
    assert result["metrics"]["prAuc"] == pytest.approx(2 / 3)
    assert result["metrics"]["p95InferenceMs"] == pytest.approx(8.0)
    assert result["metrics"]["highConfidenceFalsePositiveRate"] == pytest.approx(0.5)
    assert {item["dimension"] for item in result["slices"]} == {
        "source",
        "generator",
        "transformation",
    }
    assert result["run"]["createdAt"] == "2026-08-27T12:00:00Z"


def test_evaluate_applies_validation_calibration_to_matching_model() -> None:
    validation_manifest = [
        {"id": f"v-{index}", "label": "human" if index < 10 else "synthetic", "split": "validation"}
        for index in range(20)
    ]
    validation_predictions = [
        prediction(
            f"v-{index}",
            ("human" if index < 10 else "synthetic") if index not in {1, 12} else ("synthetic" if index < 10 else "human"),
        )
        for index in range(20)
    ]
    calibration = fit_temperature_calibration(
        validation_manifest,
        validation_predictions,
        model_name="berturk",
        model_version="test-only",
        manifest_sha256="c" * 64,
        predictions_sha256="d" * 64,
    )
    manifest = [manifest_record("a", "human"), manifest_record("b", "synthetic")]
    predictions = [prediction("a", "human"), prediction("b", "synthetic")]

    result = evaluate_predictions(
        manifest,
        predictions,
        metadata=EvaluationMetadata(
            run_id="berturk-test-run",
            git_commit="abcdef1234567",
            seed=42,
            dataset_version="synthetic-fixture-v1",
            model_name="berturk",
            model_version="test-only",
        ),
        manifest_sha256="a" * 64,
        predictions_sha256="b" * 64,
        calibration_artifact=calibration,
    )

    assert result["calibration"]["temperature"] == calibration["temperature"]
    assert result["calibration"]["validationManifestSha256"] == "c" * 64


def test_evaluate_predictions_is_order_independent() -> None:
    manifest = [manifest_record("a", "human"), manifest_record("b", "synthetic")]
    predictions = [prediction("a", "human"), prediction("b", "synthetic")]

    assert evaluate(manifest, predictions) == evaluate(list(reversed(manifest)), list(reversed(predictions)))


def test_evaluate_predictions_requires_complete_coverage() -> None:
    manifest = [manifest_record("a", "human"), manifest_record("b", "synthetic")]

    with pytest.raises(EvaluationInputError, match="missing prediction ids: b"):
        evaluate(manifest, [prediction("a", "human")])


def test_evaluate_predictions_rejects_unknown_ids() -> None:
    manifest = [manifest_record("a", "human")]

    with pytest.raises(EvaluationInputError, match="unknown prediction ids: b"):
        evaluate(manifest, [prediction("a", "human"), prediction("b", "human")])
