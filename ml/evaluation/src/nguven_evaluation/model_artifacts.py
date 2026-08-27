"""Strict local model-artifact manifest loading and integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_SCHEMA_PATH = EVALUATION_ROOT / "models" / "schema.json"
DEFAULT_MAX_MANIFEST_BYTES = 1024 * 1024


class ModelArtifactError(ValueError):
    """Raised when model provenance or local artifacts fail closed."""


@dataclass(frozen=True)
class VerifiedModelArtifacts:
    """Summary safe to log after all local artifact checks succeed."""

    model_id: str
    artifact_count: int
    total_bytes: int


def load_model_artifact_manifest(
    manifest_path: Path,
    *,
    schema_path: Path = DEFAULT_MODEL_SCHEMA_PATH,
    max_file_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
) -> dict[str, Any]:
    """Load one bounded UTF-8 JSON manifest and validate it strictly."""
    if max_file_bytes <= 0:
        raise ModelArtifactError("Maximum manifest size must be positive")
    _require_regular_file(manifest_path, "Model manifest")
    if manifest_path.stat().st_size > max_file_bytes:
        raise ModelArtifactError("Model manifest exceeds the configured size limit")

    schema = _load_json_object(schema_path, "model schema")
    Draft202012Validator.check_schema(schema)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelArtifactError("Model manifest must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ModelArtifactError("Model manifest must be a JSON object")

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    if errors:
        details = []
        for error in errors:
            field = ".".join(str(part) for part in error.absolute_path) or "<manifest>"
            details.append(f"{field}: {error.message}")
        raise ModelArtifactError(
            "Model manifest validation failed:\n" + "\n".join(f"- {item}" for item in details)
        )
    return document


def verify_model_artifacts(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> VerifiedModelArtifacts:
    """Verify declared artifact sizes and SHA-256 hashes within a local root."""
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ModelArtifactError("Artifact root must be an existing non-symbolic-link directory")
    resolved_root = artifact_root.resolve(strict=True)
    artifacts: Sequence[Mapping[str, Any]] = manifest["artifacts"]
    seen_paths: set[str] = set()
    total_bytes = 0

    for artifact in artifacts:
        relative_path = str(artifact["path"])
        if relative_path in seen_paths:
            raise ModelArtifactError(f"Duplicate model artifact path: {relative_path}")
        seen_paths.add(relative_path)
        candidate = artifact_root / relative_path
        _require_regular_file(candidate, f"Model artifact {relative_path}")
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_relative_to(resolved_root):
            raise ModelArtifactError(f"Model artifact escapes the configured root: {relative_path}")

        actual_size = resolved_candidate.stat().st_size
        if actual_size != int(artifact["sizeBytes"]):
            raise ModelArtifactError(f"Model artifact size mismatch: {relative_path}")
        if _sha256_file(resolved_candidate) != str(artifact["sha256"]):
            raise ModelArtifactError(f"Model artifact hash mismatch: {relative_path}")
        total_bytes += actual_size

    return VerifiedModelArtifacts(
        model_id=str(manifest["modelId"]),
        artifact_count=len(artifacts),
        total_bytes=total_bytes,
    )


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ModelArtifactError(f"{label} must be an existing non-symbolic-link file")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label.title())
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelArtifactError(f"{label.title()} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ModelArtifactError(f"{label.title()} must be a JSON object")
    return document
