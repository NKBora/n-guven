"""Dataset manifest loading and schema validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "manifests" / "schema.json"


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot be parsed or violates its schema."""


def load_manifest(
    manifest_path: Path,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> list[dict[str, Any]]:
    """Load and validate all records from a JSON or JSON Lines manifest."""
    schema = _load_json_object(schema_path, document_name="schema")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    records = _load_records(manifest_path)

    errors: list[str] = []
    for record_number, record in enumerate(records, start=1):
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
            field_path = ".".join(str(part) for part in error.absolute_path) or "<record>"
            errors.append(f"record {record_number}, {field_path}: {error.message}")

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ManifestValidationError(
            f"Manifest {manifest_path} failed schema validation:\n{details}"
        )

    return records


def _load_records(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise ManifestValidationError(f"Manifest file does not exist: {manifest_path}")

    if manifest_path.suffix.lower() == ".jsonl":
        records = _load_json_lines(manifest_path)
    else:
        document = _load_json(manifest_path, document_name="manifest")
        records = document if isinstance(document, list) else [document]

    if not records:
        raise ManifestValidationError(f"Manifest contains no records: {manifest_path}")

    if not all(isinstance(record, dict) for record in records):
        raise ManifestValidationError("Every manifest record must be a JSON object")

    return records


def _load_json_lines(manifest_path: Path) -> list[Any]:
    records: list[Any] = []
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            records.append(json.loads(raw_line))
        except json.JSONDecodeError as error:
            raise ManifestValidationError(
                f"Invalid JSON on line {line_number} of {manifest_path}: {error.msg}"
            ) from error
    return records


def _load_json(path: Path, *, document_name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestValidationError(f"{document_name.title()} file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            f"Invalid JSON in {document_name} {path}: {error.msg}"
        ) from error


def _load_json_object(path: Path, *, document_name: str) -> dict[str, Any]:
    document = _load_json(path, document_name=document_name)
    if not isinstance(document, dict):
        raise ManifestValidationError(f"{document_name.title()} must be a JSON object: {path}")
    return document
