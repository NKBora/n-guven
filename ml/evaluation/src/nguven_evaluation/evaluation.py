"""Model-independent evaluation of offline prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from nguven_evaluation.manifests import ManifestValidationError
from nguven_evaluation.calibration import (
    CalibrationError,
    brier_score,
    expected_calibration_error,
    predicted_synthetic_probability,
    temperature_scale,
)

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTION_SCHEMA_PATH = EVALUATION_ROOT / "predictions" / "schema.json"
DEFAULT_RESULT_SCHEMA_PATH = EVALUATION_ROOT / "results" / "schema.json"


class EvaluationInputError(ValueError):
    """Raised when predictions cannot be evaluated against a manifest."""


@dataclass(frozen=True)
class EvaluationMetadata:
    run_id: str
    git_commit: str
    seed: int
    dataset_version: str
    model_name: str
    model_version: str


def load_predictions(
    predictions_path: Path,
    *,
    schema_path: Path = DEFAULT_PREDICTION_SCHEMA_PATH,
) -> list[dict[str, Any]]:
    """Load and validate JSON or JSON Lines prediction records."""
    schema = _load_json_object(schema_path, "prediction schema")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    document = _load_json_or_jsonl(predictions_path)
    records = document if isinstance(document, list) else [document]
    if not records:
        raise EvaluationInputError(f"Prediction artifact contains no records: {predictions_path}")

    errors: list[str] = []
    for number, record in enumerate(records, start=1):
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
            field = ".".join(str(part) for part in error.absolute_path) or "<record>"
            errors.append(f"record {number}, {field}: {error.message}")
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise EvaluationInputError(f"Prediction validation failed:\n{details}")
    return records


def evaluate_predictions(
    manifest_records: Sequence[dict[str, Any]],
    prediction_records: Sequence[dict[str, Any]],
    *,
    metadata: EvaluationMetadata,
    manifest_sha256: str,
    predictions_sha256: str,
    calibration_artifact: Mapping[str, Any] | None = None,
    high_confidence_threshold: float = 0.8,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Compute classification metrics and construct a traceable result artifact."""
    if not manifest_records:
        raise EvaluationInputError("Manifest contains no records")

    predictions_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for prediction in prediction_records:
        record_id = str(prediction["id"])
        if record_id in predictions_by_id:
            duplicate_ids.append(record_id)
        predictions_by_id[record_id] = prediction
    if duplicate_ids:
        raise EvaluationInputError(
            f"Duplicate prediction id(s): {', '.join(sorted(set(duplicate_ids)))}"
        )

    manifest_by_id = {str(record["id"]): record for record in manifest_records}
    if len(manifest_by_id) != len(manifest_records):
        raise EvaluationInputError("Manifest contains duplicate record ids")

    missing = sorted(set(manifest_by_id) - set(predictions_by_id))
    extra = sorted(set(predictions_by_id) - set(manifest_by_id))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing prediction ids: {', '.join(missing)}")
        if extra:
            details.append(f"unknown prediction ids: {', '.join(extra)}")
        raise EvaluationInputError("Prediction coverage mismatch; " + "; ".join(details))

    actual = [manifest_by_id[record_id]["label"] for record_id in sorted(manifest_by_id)]
    predicted = [predictions_by_id[record_id]["predictedLabel"] for record_id in sorted(manifest_by_id)]
    inference_times = [
        float(predictions_by_id[record_id]["inferenceMs"])
        for record_id in sorted(manifest_by_id)
    ]
    metrics = _classification_metrics(actual, predicted)
    metrics["meanInferenceMs"] = sum(inference_times) / len(inference_times)

    advanced = _advanced_binary_metrics(
        manifest_by_id,
        predictions_by_id,
        inference_times=inference_times,
        calibration_artifact=calibration_artifact,
        metadata=metadata,
        high_confidence_threshold=high_confidence_threshold,
    )
    if advanced is not None:
        metrics.update(advanced["metrics"])

    result = {
        "schemaVersion": "1.0",
        "run": {
            "id": metadata.run_id,
            "createdAt": now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "gitCommit": metadata.git_commit,
            "seed": metadata.seed,
        },
        "dataset": {
            "version": metadata.dataset_version,
            "recordCount": len(manifest_records),
        },
        "model": {
            "name": metadata.model_name,
            "version": metadata.model_version,
        },
        "metrics": metrics,
        "artifacts": {
            "manifestSha256": manifest_sha256,
            "predictionsSha256": predictions_sha256,
        },
    }
    if advanced is not None:
        result["slices"] = advanced["slices"]
        if advanced["calibration"] is not None:
            result["calibration"] = advanced["calibration"]
    validate_result(result)
    return result


def validate_result(
    result: dict[str, Any],
    *,
    schema_path: Path = DEFAULT_RESULT_SCHEMA_PATH,
) -> None:
    schema = _load_json_object(schema_path, "result schema")
    Draft202012Validator.check_schema(schema)
    errors = list(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(result)
    )
    if errors:
        details = "\n".join(f"- {error.message}" for error in errors)
        raise EvaluationInputError(f"Evaluation result validation failed:\n{details}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_metrics(actual: Sequence[Any], predicted: Sequence[Any]) -> dict[str, float]:
    labels = sorted({_label_key(value) for value in [*actual, *predicted]})
    actual_keys = [_label_key(value) for value in actual]
    predicted_keys = [_label_key(value) for value in predicted]
    correct = sum(left == right for left, right in zip(actual_keys, predicted_keys))

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in labels:
        true_positive = sum(a == label and p == label for a, p in zip(actual_keys, predicted_keys))
        false_positive = sum(a != label and p == label for a, p in zip(actual_keys, predicted_keys))
        false_negative = sum(a == label and p != label for a, p in zip(actual_keys, predicted_keys))
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": correct / len(actual),
        "macroPrecision": sum(precisions) / len(precisions),
        "macroRecall": sum(recalls) / len(recalls),
        "macroF1": sum(f1_scores) / len(f1_scores),
    }


def _advanced_binary_metrics(
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
    *,
    inference_times: Sequence[float],
    calibration_artifact: Mapping[str, Any] | None,
    metadata: EvaluationMetadata,
    high_confidence_threshold: float,
) -> dict[str, Any] | None:
    ordered_ids = sorted(manifest_by_id)
    labels = [manifest_by_id[record_id]["label"] for record_id in ordered_ids]
    if set(labels) - {"human", "synthetic"}:
        return None
    if any(predictions_by_id[record_id].get("score") is None for record_id in ordered_ids):
        return None
    if not 0.5 < high_confidence_threshold <= 1:
        raise EvaluationInputError("High-confidence threshold must be within (0.5, 1]")

    temperature = 1.0
    calibration_document: dict[str, Any] | None = None
    if calibration_artifact is not None:
        model = calibration_artifact["model"]
        if model["name"] != metadata.model_name or model["version"] != metadata.model_version:
            raise EvaluationInputError("Calibration artifact model identity mismatch")
        temperature = float(calibration_artifact["temperature"])
        calibration_document = {
            "method": "temperature-scaling",
            "temperature": temperature,
            "validationManifestSha256": calibration_artifact["artifacts"]["manifestSha256"],
            "validationPredictionsSha256": calibration_artifact["artifacts"]["predictionsSha256"],
        }
    try:
        probabilities = [
            temperature_scale(
                predicted_synthetic_probability(predictions_by_id[record_id]),
                temperature,
            )
            for record_id in ordered_ids
        ]
    except CalibrationError as error:
        raise EvaluationInputError("Unable to derive calibrated binary probabilities") from error
    binary_actual = [1 if label == "synthetic" else 0 for label in labels]
    false_positives = sum(
        actual == 0 and probability >= high_confidence_threshold
        for actual, probability in zip(binary_actual, probabilities)
    )
    human_count = sum(actual == 0 for actual in binary_actual)
    if human_count == 0 or len(set(binary_actual)) != 2:
        raise EvaluationInputError("Binary evaluation requires both human and synthetic labels")
    metrics = {
        "prAuc": _average_precision(binary_actual, probabilities),
        "brierScore": brier_score(binary_actual, probabilities),
        "ece": expected_calibration_error(binary_actual, probabilities, bins=10),
        "highConfidenceFalsePositiveRate": false_positives / human_count,
        "p95InferenceMs": _percentile_nearest_rank(inference_times, 0.95),
    }
    return {
        "metrics": metrics,
        "calibration": calibration_document,
        "slices": _build_slices(
            ordered_ids,
            manifest_by_id,
            predictions_by_id,
        ),
    }


def _average_precision(actual: Sequence[int], probabilities: Sequence[float]) -> float:
    positives = sum(actual)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(probabilities, actual), key=lambda item: item[0], reverse=True)
    true_positives = 0
    seen = 0
    average_precision = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group: list[int] = []
        while index < len(ordered) and ordered[index][0] == score:
            group.append(ordered[index][1])
            index += 1
        new_positives = sum(group)
        true_positives += new_positives
        seen += len(group)
        if new_positives:
            average_precision += (new_positives / positives) * (true_positives / seen)
    return average_precision


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise EvaluationInputError("Latency metric requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _build_slices(
    ordered_ids: Sequence[str],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    dimensions = {
        "source": lambda record: record.get("sourceGroup") or record.get("source") or "unknown",
        "generator": lambda record: record.get("generatorFamily") or record.get("generatorModel") or "human",
        "transformation": lambda record: record.get("transformation") or "none",
    }
    slices: list[dict[str, Any]] = []
    for dimension, key_function in dimensions.items():
        groups: dict[str, list[str]] = {}
        for record_id in ordered_ids:
            key = str(key_function(manifest_by_id[record_id]))
            groups.setdefault(key, []).append(record_id)
        for key in sorted(groups):
            record_ids = groups[key]
            actual = [manifest_by_id[record_id]["label"] for record_id in record_ids]
            predicted = [
                predictions_by_id[record_id]["predictedLabel"] for record_id in record_ids
            ]
            human_count = sum(label == "human" for label in actual)
            false_positives = sum(
                left == "human" and right == "synthetic"
                for left, right in zip(actual, predicted)
            )
            slices.append(
                {
                    "dimension": dimension,
                    "key": key,
                    "recordCount": len(record_ids),
                    "macroF1": _classification_metrics(actual, predicted)["macroF1"],
                    "falsePositiveRate": false_positives / human_count if human_count else 0.0,
                }
            )
    return slices


def _label_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _load_json_or_jsonl(path: Path) -> Any:
    if not path.is_file():
        raise EvaluationInputError(f"Prediction artifact does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise EvaluationInputError(
                    f"Invalid JSON on line {line_number} of {path}: {error.msg}"
                ) from error
        return records
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationInputError(f"Invalid JSON in {path}: {error.msg}") from error


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ManifestValidationError(f"Unable to load {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ManifestValidationError(f"{name.title()} must be a JSON object: {path}")
    return document
