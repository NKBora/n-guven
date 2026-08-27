"""Model-independent evaluation of offline prediction artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from nguven_evaluation.manifests import ManifestValidationError

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
