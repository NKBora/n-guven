"""Secure generation of offline text-classification prediction artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from jsonschema import Draft202012Validator

from nguven_evaluation.evaluation import DEFAULT_PREDICTION_SCHEMA_PATH
from nguven_evaluation.integrity import compute_text_content_hash
from nguven_evaluation.model_adapters import ModelAdapterError, TextModelAdapter
from nguven_evaluation.offline_preprocessing import DEFAULT_PREPROCESSED_SCHEMA_PATH

DEFAULT_MAX_PREPROCESSED_BYTES = 128 * 1024 * 1024


class OfflinePredictionError(ValueError):
    """Raised when an offline prediction artifact cannot be produced safely."""


def load_preprocessed_records(
    input_path: Path,
    *,
    schema_path: Path = DEFAULT_PREPROCESSED_SCHEMA_PATH,
    max_file_bytes: int = DEFAULT_MAX_PREPROCESSED_BYTES,
) -> list[dict[str, Any]]:
    """Load bounded JSONL preprocessing output and recheck its content hashes."""
    if max_file_bytes <= 0:
        raise OfflinePredictionError("Maximum preprocessed file size must be positive")
    if input_path.is_symlink() or not input_path.is_file():
        raise OfflinePredictionError(
            "Preprocessed input must be an existing non-symbolic-link file"
        )
    if input_path.suffix.lower() != ".jsonl":
        raise OfflinePredictionError("Preprocessed input must use the .jsonl extension")
    if input_path.stat().st_size > max_file_bytes:
        raise OfflinePredictionError("Preprocessed input exceeds the configured size limit")

    schema = _load_schema(schema_path, "preprocessed input")
    validator = Draft202012Validator(schema)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise OfflinePredictionError("Preprocessed input must be valid UTF-8 JSON Lines") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise OfflinePredictionError(
                f"Invalid JSON in preprocessed input at line {line_number}"
            ) from error
        errors = list(validator.iter_errors(record))
        if errors:
            raise OfflinePredictionError(
                f"Preprocessed input violates its schema at line {line_number}"
            )
        record_id = str(record["id"])
        if record_id in seen_ids:
            raise OfflinePredictionError(f"Duplicate preprocessed record id: {record_id}")
        seen_ids.add(record_id)
        actual_hash = compute_text_content_hash(str(record["text"]))
        if not hmac.compare_digest(actual_hash, str(record["outputContentHash"])):
            raise OfflinePredictionError(
                f"Preprocessed content hash mismatch for record id: {record_id}"
            )
        records.append(record)
    if not records:
        raise OfflinePredictionError("Preprocessed input contains no records")
    return records


def build_prediction_records(
    preprocessed_records: Sequence[dict[str, Any]],
    *,
    adapter: TextModelAdapter,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    schema_path: Path = DEFAULT_PREDICTION_SCHEMA_PATH,
) -> list[dict[str, Any]]:
    """Run one local adapter in stable record order and validate every prediction."""
    if not preprocessed_records:
        raise OfflinePredictionError("Preprocessed input contains no records")
    indexed: dict[str, dict[str, Any]] = {}
    for record in preprocessed_records:
        record_id = str(record["id"])
        if record_id in indexed:
            raise OfflinePredictionError(f"Duplicate preprocessed record id: {record_id}")
        if record.get("preprocessingVersion") != adapter.preprocessing_version:
            raise OfflinePredictionError(
                f"Preprocessing version mismatch for record id: {record_id}"
            )
        indexed[record_id] = record

    output: list[dict[str, Any]] = []
    for record_id in sorted(indexed):
        started_ns = clock_ns()
        try:
            prediction = adapter.predict(str(indexed[record_id]["text"]))
        except ModelAdapterError as error:
            raise OfflinePredictionError(f"Inference failed for record id: {record_id}") from error
        finished_ns = clock_ns()
        if finished_ns < started_ns:
            raise OfflinePredictionError("Monotonic inference clock moved backwards")
        output.append(
            {
                "id": record_id,
                "predictedLabel": prediction.label,
                "score": prediction.score,
                "inferenceMs": (finished_ns - started_ns) / 1_000_000,
            }
        )

    _validate_records(output, schema_path=schema_path)
    return output


def write_private_predictions(
    records: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    force: bool = False,
) -> str:
    """Atomically write owner-only prediction JSONL and return its SHA-256."""
    if output_path.suffix.lower() != ".jsonl":
        raise OfflinePredictionError("Prediction output must use the .jsonl extension")
    if output_path.is_symlink():
        raise OfflinePredictionError("Prediction output must not be a symbolic link")
    if output_path.exists() and not force:
        raise OfflinePredictionError(
            f"Output already exists; use --force to replace it: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            os.chmod(temporary_path, 0o600)
            for record in records:
                temporary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as error:
        raise OfflinePredictionError("Unable to write private prediction output") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return _sha256_file(output_path)


def ensure_prediction_path_is_distinct(output_path: Path, protected_paths: Sequence[Path]) -> None:
    """Prevent the prediction output from replacing an input or model artifact."""
    output_resolved = output_path.resolve(strict=False)
    for protected_path in protected_paths:
        if output_resolved == protected_path.resolve(strict=False):
            raise OfflinePredictionError("Prediction output must differ from all input artifacts")


def _load_schema(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OfflinePredictionError(f"{label.title()} schema is unavailable")
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OfflinePredictionError(f"{label.title()} schema must be valid UTF-8 JSON") from error
    if not isinstance(schema, dict):
        raise OfflinePredictionError(f"{label.title()} schema must be a JSON object")
    Draft202012Validator.check_schema(schema)
    return schema


def _validate_records(records: Sequence[dict[str, Any]], *, schema_path: Path) -> None:
    validator = Draft202012Validator(_load_schema(schema_path, "prediction output"))
    invalid = [
        number
        for number, record in enumerate(records, start=1)
        if any(validator.iter_errors(record))
    ]
    if invalid:
        raise OfflinePredictionError(
            "Generated predictions violate the output schema at record(s): "
            + ", ".join(str(number) for number in invalid)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
