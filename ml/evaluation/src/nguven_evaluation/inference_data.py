"""Controlled materialization of validation and frozen-test inference inputs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nguven_evaluation.experiments import validate_comparable_experiments
from nguven_evaluation.splitting import audit_manifest

INFERENCE_SPLITS = ("validation", "test")


class InferenceDataError(ValueError):
    """Raised when an inference split would violate evidence boundaries."""


@dataclass(frozen=True)
class PreparedInferenceSplit:
    """One aligned manifest/preprocessing slice ready for offline inference."""

    split: str
    manifest_records: list[dict[str, Any]]
    preprocessed_records: list[dict[str, Any]]


def prepare_inference_split(
    manifest_records: Sequence[dict[str, Any]],
    preprocessed_records: Sequence[dict[str, Any]],
    *,
    split: str,
    experiments: Sequence[Mapping[str, Any]] = (),
) -> PreparedInferenceSplit:
    """Select an aligned split and gate frozen-test access on completed candidates."""
    if split not in INFERENCE_SPLITS:
        raise InferenceDataError("Inference split must be validation or test")
    if split == "test":
        _require_completed_candidates(experiments)
    elif experiments:
        raise InferenceDataError("Validation extraction does not accept experiment gates")

    audit_manifest(manifest_records)
    manifest_by_id = _index_unique(manifest_records, "manifest")
    preprocessed_by_id = _index_unique(preprocessed_records, "preprocessed input")
    if set(manifest_by_id) != set(preprocessed_by_id):
        raise InferenceDataError("Manifest and preprocessed input must have identical coverage")

    selected_manifest: list[dict[str, Any]] = []
    selected_preprocessed: list[dict[str, Any]] = []
    for record_id in sorted(manifest_by_id):
        manifest = manifest_by_id[record_id]
        if manifest["split"] != split:
            continue
        preprocessed = preprocessed_by_id[record_id]
        if not hmac.compare_digest(
            str(manifest["contentHash"]).lower(),
            str(preprocessed["inputContentHash"]).lower(),
        ):
            raise InferenceDataError(
                f"Manifest/preprocessing content hash mismatch for record id: {record_id}"
            )
        selected_manifest.append(dict(manifest))
        selected_preprocessed.append(dict(preprocessed))

    if not selected_manifest:
        raise InferenceDataError(f"Inference split contains no records: {split}")
    labels = {str(record["label"]) for record in selected_manifest}
    if labels != {"human", "synthetic"}:
        raise InferenceDataError("Inference split must contain both reviewed labels")
    return PreparedInferenceSplit(
        split=split,
        manifest_records=selected_manifest,
        preprocessed_records=selected_preprocessed,
    )


def write_private_inference_split(
    package: PreparedInferenceSplit,
    output_root: Path,
    *,
    source_manifest_sha256: str,
    source_preprocessed_sha256: str,
) -> dict[str, Any]:
    """Atomically write an owner-only inference package and provenance record."""
    if output_root.exists() or output_root.is_symlink():
        raise InferenceDataError("Inference output directory must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.parent.is_symlink():
        raise InferenceDataError("Inference output parent must not be a symbolic link")

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        os.chmod(temporary_root, 0o700)
        manifest_path = temporary_root / "manifest.jsonl"
        preprocessed_path = temporary_root / "preprocessed.jsonl"
        _write_jsonl(manifest_path, package.manifest_records)
        _write_jsonl(preprocessed_path, package.preprocessed_records)
        provenance = {
            "schemaVersion": "1.0",
            "split": package.split,
            "recordCount": len(package.manifest_records),
            "source": {
                "manifestSha256": source_manifest_sha256,
                "preprocessedSha256": source_preprocessed_sha256,
            },
            "artifacts": {
                "manifestSha256": sha256_file(manifest_path),
                "preprocessedSha256": sha256_file(preprocessed_path),
            },
        }
        _write_json(temporary_root / "provenance.json", provenance)
        os.replace(temporary_root, output_root)
        return provenance
    except OSError as error:
        raise InferenceDataError("Unable to write private inference split") from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def sha256_file(path: Path) -> str:
    """Hash one regular input without following symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise InferenceDataError("Inference source must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_completed_candidates(experiments: Sequence[Mapping[str, Any]]) -> None:
    try:
        validate_comparable_experiments(list(experiments))
    except ValueError as error:
        raise InferenceDataError(
            "Frozen test access requires both comparable candidate experiments"
        ) from error
    for experiment in experiments:
        execution = experiment["execution"]
        if (
            execution["status"] != "completed"
            or execution["allowed"] is not False
            or execution["resultArtifact"] is None
        ):
            raise InferenceDataError(
                "Frozen test access requires both candidate experiments to be completed"
            )


def _index_unique(
    records: Sequence[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record["id"])
        if record_id in indexed:
            raise InferenceDataError(f"Duplicate {label} record id: {record_id}")
        indexed[record_id] = record
    return indexed


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as artifact:
        os.chmod(path, 0o600)
        for record in records:
            artifact.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as artifact:
        os.chmod(path, 0o600)
        json.dump(document, artifact, ensure_ascii=False, indent=2)
        artifact.write("\n")
