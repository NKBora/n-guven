"""Fail-closed loading and integrity verification for private image inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_INPUT_SCHEMA_PATH = EVALUATION_ROOT / "image" / "inputs" / "schema.json"
DEFAULT_MAX_IMAGE_MANIFEST_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_TOTAL_IMAGE_BYTES = 5 * 1024 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024


class ImageDatasetInputError(ValueError):
    """Raised when private image inputs fail their security or integrity contract."""


@dataclass(frozen=True, slots=True)
class VerifiedImageInput:
    """Non-sensitive identity for one verified local image file."""

    record_id: str
    path: Path
    sha256: str
    size_bytes: int


def load_image_dataset_inputs(
    manifest_path: Path,
    *,
    image_root: Path,
    schema_path: Path = DEFAULT_IMAGE_INPUT_SCHEMA_PATH,
    max_manifest_bytes: int = DEFAULT_MAX_IMAGE_MANIFEST_BYTES,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_IMAGE_BYTES,
) -> list[VerifiedImageInput]:
    """Validate a JSON/JSONL manifest and verify every referenced image byte-for-byte."""
    _require_positive_limit(max_manifest_bytes, "Maximum manifest size")
    _require_positive_limit(max_image_bytes, "Maximum image size")
    _require_positive_limit(max_total_bytes, "Maximum total image size")
    _require_regular_file(manifest_path, "Image input manifest")
    if manifest_path.stat().st_size > max_manifest_bytes:
        raise ImageDatasetInputError("Image input manifest exceeds the safety limit")
    root = _require_image_root(image_root)
    schema = _load_schema(schema_path)
    records = _load_records(manifest_path)
    if not records:
        raise ImageDatasetInputError("Image input manifest contains no records")

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validation_errors: list[str] = []
    for number, record in enumerate(records, start=1):
        errors = sorted(
            validator.iter_errors(record),
            key=lambda item: list(item.path),
        )
        for error in errors:
            validation_errors.append(_safe_validation_message(number, error))
    if validation_errors:
        details = "\n".join(f"- {message}" for message in validation_errors)
        raise ImageDatasetInputError(
            "Image input manifest validation failed:\n" + details
        )

    duplicate_ids = _duplicates(str(record["id"]) for record in records)
    if duplicate_ids:
        raise ImageDatasetInputError(
            "Image input manifest contains duplicate id(s): "
            + ", ".join(duplicate_ids)
        )
    duplicate_paths = _duplicates(str(record["path"]) for record in records)
    if duplicate_paths:
        raise ImageDatasetInputError("Image input manifest contains duplicate path(s)")

    verified: list[VerifiedImageInput] = []
    total_bytes = 0
    for record in records:
        record_id = str(record["id"])
        relative = _safe_relative_path(str(record["path"]), record_id=record_id)
        _reject_symlink_components(root, relative, record_id=record_id)
        candidate = root / relative
        _require_regular_file(candidate, f"Image input for record {record_id}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ImageDatasetInputError(
                f"Image input escapes the configured root for record {record_id}"
            )

        declared_size = int(record["sizeBytes"])
        actual_size = resolved.stat().st_size
        if declared_size > max_image_bytes or actual_size > max_image_bytes:
            raise ImageDatasetInputError(
                f"Image input exceeds the per-file safety limit for record {record_id}"
            )
        if actual_size != declared_size:
            raise ImageDatasetInputError(
                f"Image input size mismatch for record {record_id}"
            )
        total_bytes += actual_size
        if total_bytes > max_total_bytes:
            raise ImageDatasetInputError("Image inputs exceed the total safety limit")

        expected_hash = str(record["sha256"])
        if _sha256_file(resolved) != expected_hash:
            raise ImageDatasetInputError(
                f"Image input SHA-256 mismatch for record {record_id}"
            )
        verified.append(
            VerifiedImageInput(
                record_id=record_id,
                path=resolved,
                sha256=expected_hash,
                size_bytes=actual_size,
            )
        )
    return verified


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ImageDatasetInputError(
            "Image input manifest must be valid UTF-8"
        ) from error
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ImageDatasetInputError(
                    f"Invalid JSON at image manifest line {number}"
                ) from error
        document: Any = records
    else:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ImageDatasetInputError(
                "Image input manifest must be valid JSON or JSON Lines"
            ) from error
    if not isinstance(document, list) or any(
        not isinstance(record, dict) for record in document
    ):
        raise ImageDatasetInputError("Image input manifest must contain JSON objects")
    return document


def _load_schema(path: Path) -> Mapping[str, Any]:
    _require_regular_file(path, "Image input schema")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageDatasetInputError(
            "Image input schema must be valid UTF-8 JSON"
        ) from error
    if not isinstance(document, dict):
        raise ImageDatasetInputError("Image input schema must be a JSON object")
    return document


def _safe_relative_path(value: str, *, record_id: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ImageDatasetInputError(
            f"Image input path is unsafe for record {record_id}"
        )
    return path


def _reject_symlink_components(root: Path, relative: Path, *, record_id: str) -> None:
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ImageDatasetInputError(
                f"Image input path contains a symbolic link for record {record_id}"
            )


def _require_image_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ImageDatasetInputError(
            "Image root must be an existing non-symbolic-link directory"
        )
    return path.resolve(strict=True)


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ImageDatasetInputError(
            f"{label} must be an existing non-symbolic-link file"
        )


def _require_positive_limit(value: int, label: str) -> None:
    if value < 1:
        raise ValueError(f"{label} must be positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _safe_validation_message(number: int, error: ValidationError) -> str:
    field = ".".join(str(part) for part in error.absolute_path) or "<record>"
    reasons = {
        "required": "is missing a required field",
        "additionalProperties": "contains an unsupported field",
        "type": "has an invalid type",
        "pattern": "has an invalid format",
        "minimum": "is below the allowed minimum",
        "maximum": "exceeds the allowed maximum",
    }
    reason = reasons.get(str(error.validator), "violates the input contract")
    return f"record {number}, {field}: {reason}"
