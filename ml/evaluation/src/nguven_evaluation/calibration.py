"""Validation-only temperature calibration for binary text-origin predictions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_SCHEMA_PATH = EVALUATION_ROOT / "calibration" / "schema.json"
DEFAULT_CALIBRATION_BINS = 10
PROBABILITY_EPSILON = 1e-7


class CalibrationError(ValueError):
    """Raised when calibration data or provenance is unsafe or incomplete."""


def predicted_synthetic_probability(prediction: Mapping[str, Any]) -> float:
    """Convert predicted-label confidence into the binary synthetic probability."""
    label = prediction.get("predictedLabel")
    score = prediction.get("score")
    if label not in {"human", "synthetic"} or score is None:
        raise CalibrationError(
            "Calibration requires human/synthetic labels and non-null confidence scores"
        )
    probability = float(score)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise CalibrationError("Prediction confidence must be finite and within [0, 1]")
    return probability if label == "synthetic" else 1 - probability


def temperature_scale(probability: float, temperature: float) -> float:
    """Apply scalar temperature scaling to one binary probability."""
    if not math.isfinite(temperature) or not 0.05 <= temperature <= 10:
        raise CalibrationError("Temperature must be finite and within [0.05, 10]")
    clipped = min(max(probability, PROBABILITY_EPSILON), 1 - PROBABILITY_EPSILON)
    logit = math.log(clipped / (1 - clipped))
    return 1 / (1 + math.exp(-logit / temperature))


def fit_temperature_calibration(
    manifest_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    model_version: str,
    manifest_sha256: str,
    predictions_sha256: str,
    bins: int = DEFAULT_CALIBRATION_BINS,
    schema_path: Path = DEFAULT_CALIBRATION_SCHEMA_PATH,
) -> dict[str, Any]:
    """Fit deterministic temperature scaling using validation records only."""
    actual, probabilities = _aligned_binary_inputs(
        manifest_records, prediction_records, required_split="validation"
    )
    temperatures = [math.exp(math.log(0.05) + index * (math.log(10 / 0.05) / 400)) for index in range(401)]
    temperature = min(
        temperatures,
        key=lambda candidate: _negative_log_likelihood(
            actual,
            [temperature_scale(value, candidate) for value in probabilities],
        ),
    )
    calibrated = [temperature_scale(value, temperature) for value in probabilities]
    artifact = {
        "schemaVersion": "1.0",
        "method": "temperature-scaling",
        "fittedSplit": "validation",
        "model": {"name": model_name, "version": model_version},
        "temperature": temperature,
        "recordCount": len(actual),
        "metrics": {
            "negativeLogLikelihoodBefore": _negative_log_likelihood(actual, probabilities),
            "negativeLogLikelihoodAfter": _negative_log_likelihood(actual, calibrated),
            "brierBefore": brier_score(actual, probabilities),
            "brierAfter": brier_score(actual, calibrated),
            "eceBefore": expected_calibration_error(actual, probabilities, bins=bins),
            "eceAfter": expected_calibration_error(actual, calibrated, bins=bins),
        },
        "artifacts": {
            "manifestSha256": manifest_sha256.lower(),
            "predictionsSha256": predictions_sha256.lower(),
        },
    }
    validate_calibration_artifact(artifact, schema_path=schema_path)
    return artifact


def load_calibration_artifact(
    path: Path, *, schema_path: Path = DEFAULT_CALIBRATION_SCHEMA_PATH
) -> dict[str, Any]:
    """Load one bounded non-symbolic calibration artifact."""
    if path.is_symlink() or not path.is_file():
        raise CalibrationError("Calibration artifact must be a regular file")
    if path.stat().st_size > 1024 * 1024:
        raise CalibrationError("Calibration artifact exceeds the size limit")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationError("Calibration artifact must be valid UTF-8 JSON") from error
    if not isinstance(artifact, dict):
        raise CalibrationError("Calibration artifact must be a JSON object")
    validate_calibration_artifact(artifact, schema_path=schema_path)
    return artifact


def validate_calibration_artifact(
    artifact: Mapping[str, Any], *, schema_path: Path = DEFAULT_CALIBRATION_SCHEMA_PATH
) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationError("Unable to load calibration schema") from error
    Draft202012Validator.check_schema(schema)
    if list(Draft202012Validator(schema).iter_errors(artifact)):
        raise CalibrationError("Calibration artifact violates its schema")
    metrics = artifact["metrics"]
    if metrics["negativeLogLikelihoodAfter"] > metrics["negativeLogLikelihoodBefore"] + 1e-12:
        raise CalibrationError("Temperature calibration cannot worsen validation NLL")


def brier_score(actual: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate_metric_inputs(actual, probabilities)
    return sum((probability - label) ** 2 for label, probability in zip(actual, probabilities)) / len(actual)


def expected_calibration_error(
    actual: Sequence[int], probabilities: Sequence[float], *, bins: int
) -> float:
    _validate_metric_inputs(actual, probabilities)
    if bins < 2 or bins > 100:
        raise CalibrationError("ECE bin count must be between 2 and 100")
    total = len(actual)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1)
        ]
        if not members:
            continue
        confidence = sum(probabilities[item] for item in members) / len(members)
        frequency = sum(actual[item] for item in members) / len(members)
        error += len(members) / total * abs(confidence - frequency)
    return error


def _aligned_binary_inputs(
    manifest_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    required_split: str | None,
) -> tuple[list[int], list[float]]:
    manifest_by_id = {str(record["id"]): record for record in manifest_records}
    predictions_by_id = {str(record["id"]): record for record in prediction_records}
    if len(manifest_by_id) != len(manifest_records) or len(predictions_by_id) != len(prediction_records):
        raise CalibrationError("Calibration inputs contain duplicate record ids")
    if set(manifest_by_id) != set(predictions_by_id):
        raise CalibrationError("Calibration prediction coverage mismatch")
    actual: list[int] = []
    probabilities: list[float] = []
    for record_id in sorted(manifest_by_id):
        record = manifest_by_id[record_id]
        if required_split is not None and record.get("split") != required_split:
            raise CalibrationError(
                f"Calibration requires only the {required_split} split"
            )
        label = record.get("label")
        if label not in {"human", "synthetic"}:
            raise CalibrationError("Calibration requires the text-origin-v1 ontology")
        actual.append(1 if label == "synthetic" else 0)
        probabilities.append(predicted_synthetic_probability(predictions_by_id[record_id]))
    if len(set(actual)) != 2:
        raise CalibrationError("Calibration requires both human and synthetic labels")
    return actual, probabilities


def _negative_log_likelihood(actual: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate_metric_inputs(actual, probabilities)
    losses = []
    for label, probability in zip(actual, probabilities):
        clipped = min(max(probability, PROBABILITY_EPSILON), 1 - PROBABILITY_EPSILON)
        losses.append(-(label * math.log(clipped) + (1 - label) * math.log(1 - clipped)))
    return sum(losses) / len(losses)


def _validate_metric_inputs(actual: Sequence[int], probabilities: Sequence[float]) -> None:
    if not actual or len(actual) != len(probabilities):
        raise CalibrationError("Metric inputs must be non-empty and aligned")
    if any(label not in {0, 1} for label in actual):
        raise CalibrationError("Metric labels must be binary")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise CalibrationError("Metric probabilities must be within [0, 1]")
