"""Secure loading and validation for local-only text dataset inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_SCHEMA_PATH = EVALUATION_ROOT / "inputs" / "schema.json"
DEFAULT_MAX_INPUT_BYTES = 128 * 1024 * 1024


class DatasetInputError(ValueError):
    """Raised when a local text input artifact cannot be safely loaded."""


def load_dataset_input(
    input_path: Path,
    *,
    schema_path: Path = DEFAULT_INPUT_SCHEMA_PATH,
    max_file_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> list[dict[str, Any]]:
    """Load a strict JSON/JSONL text artifact without exposing text in errors."""
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be greater than zero")
    if input_path.is_symlink():
        raise DatasetInputError(f"Dataset input must not be a symbolic link: {input_path}")
    if not input_path.is_file():
        raise DatasetInputError(f"Dataset input file does not exist: {input_path}")

    raw_bytes = _read_bounded(input_path, max_file_bytes=max_file_bytes)
    try:
        document = _parse_document(input_path, raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise DatasetInputError(f"Dataset input must be valid UTF-8: {input_path}") from error

    records = document if isinstance(document, list) else [document]
    if not records:
        raise DatasetInputError(f"Dataset input contains no records: {input_path}")

    schema = _load_schema(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validation_errors: list[str] = []
    for record_number, record in enumerate(records, start=1):
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
            validation_errors.append(_safe_validation_message(record_number, error))

    if validation_errors:
        details = "\n".join(f"- {message}" for message in validation_errors)
        raise DatasetInputError(f"Dataset input validation failed:\n{details}")

    duplicate_ids = _duplicate_ids(records)
    if duplicate_ids:
        raise DatasetInputError(
            "Dataset input contains duplicate id(s): " + ", ".join(duplicate_ids)
        )

    return records


def _read_bounded(path: Path, *, max_file_bytes: int) -> bytes:
    with path.open("rb") as input_file:
        raw_bytes = input_file.read(max_file_bytes + 1)
    if len(raw_bytes) > max_file_bytes:
        raise DatasetInputError(
            f"Dataset input exceeds the {max_file_bytes}-byte safety limit: {path}"
        )
    return raw_bytes


def _parse_document(path: Path, text: str) -> Any:
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise DatasetInputError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error
        return records

    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise DatasetInputError(f"Invalid JSON in dataset input: {path}") from error


def _load_schema(schema_path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetInputError(f"Dataset input schema does not exist: {schema_path}") from error
    except json.JSONDecodeError as error:
        raise DatasetInputError(f"Dataset input schema is invalid JSON: {schema_path}") from error
    if not isinstance(schema, dict):
        raise DatasetInputError(f"Dataset input schema must be a JSON object: {schema_path}")
    return schema


def _safe_validation_message(record_number: int, error: ValidationError) -> str:
    field = ".".join(str(part) for part in error.absolute_path) or "<record>"
    messages = {
        "required": "is missing a required field",
        "additionalProperties": "contains an unsupported field",
        "type": "has an invalid type",
        "minLength": "must not be empty",
        "maxLength": "exceeds the allowed length",
    }
    reason = messages.get(str(error.validator), "violates the input contract")
    return f"record {record_number}, {field}: {reason}"


def _duplicate_ids(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        record_id = str(record["id"])
        if record_id in seen:
            duplicates.add(record_id)
        seen.add(record_id)
    return sorted(duplicates)
