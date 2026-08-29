"""Fail-closed packaging for locally fine-tuned text classifiers."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from nguven_evaluation.finetuning import (
    DEFAULT_LABELS,
    load_finetuning_plan,
    load_label_ontology,
    sha256_regular_file,
    write_private_json,
)
from nguven_evaluation.model_adapters import CANDIDATES
from nguven_evaluation.model_artifacts import DEFAULT_MODEL_SCHEMA_PATH
from nguven_evaluation.preprocessing import DEFAULT_PREPROCESSING_VERSION


class ModelPackagingError(ValueError):
    """Raised when a fine-tuned artifact bundle cannot be trusted or packaged."""


_ALLOWED_FILES: Mapping[str, tuple[str, str]] = {
    "model.safetensors": ("weights", "application/vnd.safetensors"),
    "config.json": ("configuration", "application/json"),
    "tokenizer.json": ("tokenizer", "application/json"),
    "tokenizer_config.json": ("tokenizer", "application/json"),
    "special_tokens_map.json": ("tokenizer", "application/json"),
    "added_tokens.json": ("tokenizer", "application/json"),
    "vocab.txt": ("tokenizer", "text/plain"),
}
_REQUIRED_FILES = {"model.safetensors", "config.json", "tokenizer_config.json"}


def package_finetuned_model(
    artifact_root: Path,
    *,
    model_id: str,
    adapter_id: str,
    framework_version: str,
    max_sequence_length: int,
    plan_path: Path,
    seed: int,
    git_commit: str,
    intended_use: str,
    limitations: str,
    ontology_path: Path | None = None,
    schema_path: Path = DEFAULT_MODEL_SCHEMA_PATH,
) -> dict[str, Any]:
    """Inspect a clean safetensors export and produce its validated manifest."""
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ModelPackagingError("Model artifact root must be a non-symbolic-link directory")
    try:
        candidate = CANDIDATES[adapter_id]
    except KeyError as error:
        raise ModelPackagingError(f"Unsupported model candidate: {adapter_id}") from error

    plan = load_finetuning_plan(plan_path)
    planned_candidate_ids = {str(item["adapterId"]) for item in plan["candidates"]}
    if adapter_id not in planned_candidate_ids:
        raise ModelPackagingError("Model candidate is not present in the fine-tuning plan")
    if seed not in {int(value) for value in plan["protocol"]["seeds"]}:
        raise ModelPackagingError("Model seed is not present in the fine-tuning plan")
    if max_sequence_length != int(plan["protocol"]["maxSequenceLength"]):
        raise ModelPackagingError("Model sequence length differs from the fine-tuning plan")
    ontology = load_label_ontology() if ontology_path is None else load_label_ontology(ontology_path)
    labels = ontology["labels"]
    entries = sorted(artifact_root.iterdir(), key=lambda item: item.name)
    if not entries:
        raise ModelPackagingError("Fine-tuned model artifact directory is empty")
    names = {entry.name for entry in entries}
    missing = sorted(_REQUIRED_FILES - names)
    if "tokenizer.json" not in names and "vocab.txt" not in names:
        missing.append("tokenizer.json or vocab.txt")
    if missing:
        raise ModelPackagingError("Fine-tuned artifact bundle is missing: " + ", ".join(missing))

    artifacts: list[dict[str, Any]] = []
    for entry in entries:
        if entry.is_symlink():
            raise ModelPackagingError(f"Model artifact must not be a symbolic link: {entry.name}")
        if not entry.is_file():
            raise ModelPackagingError(f"Nested model artifact paths are not allowed: {entry.name}")
        if entry.name not in _ALLOWED_FILES:
            raise ModelPackagingError(f"Unsupported file in model artifact bundle: {entry.name}")
        size = entry.stat().st_size
        if size < 1:
            raise ModelPackagingError(f"Model artifact must not be empty: {entry.name}")
        role, media_type = _ALLOWED_FILES[entry.name]
        artifacts.append(
            {
                "path": entry.name,
                "role": role,
                "mediaType": media_type,
                "sha256": _sha256_file(entry),
                "sizeBytes": size,
            }
        )

    _validate_classifier_config(artifact_root / "config.json", labels)
    manifest = {
        "schemaVersion": "1.0",
        "modelId": model_id,
        "adapterId": adapter_id,
        "task": "text-classification",
        "languages": ["tr"],
        "preprocessingVersion": DEFAULT_PREPROCESSING_VERSION,
        "upstream": {
            "provider": candidate.provider,
            "repository": candidate.repository,
            "revision": candidate.revision,
            "tokenizerRepository": candidate.repository,
            "tokenizerRevision": candidate.revision,
            "weightsLicense": {
                "spdxId": candidate.weights_license,
                "url": _license_url(candidate.weights_license),
            },
            "codeLicense": {
                "spdxId": "Apache-2.0",
                "url": "https://spdx.org/licenses/Apache-2.0.html",
            },
        },
        "fineTuning": {
            "planId": plan["planId"],
            "planSha256": sha256_regular_file(plan_path),
            "datasetManifestSha256": plan["dataset"]["manifestSha256"],
            "preprocessedSha256": plan["dataset"]["preprocessedSha256"],
            "seed": seed,
            "gitCommit": git_commit,
        },
        "runtime": {
            "framework": "transformers",
            "frameworkVersion": framework_version,
            "pythonVersion": platform.python_version(),
            "artifactFormat": "safetensors",
            "maxSequenceLength": max_sequence_length,
        },
        "labels": labels,
        "artifacts": artifacts,
        "intendedUse": intended_use,
        "limitations": limitations,
    }
    _validate_manifest(manifest, schema_path=schema_path)
    return manifest


def write_model_manifest(manifest: Mapping[str, Any], output_path: Path) -> None:
    """Write a generated model manifest using the private atomic writer."""
    write_private_json(manifest, output_path)


def _validate_classifier_config(path: Path, labels: Mapping[str, str]) -> None:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelPackagingError("Fine-tuned model config must be valid UTF-8 JSON") from error
    if not isinstance(config, dict):
        raise ModelPackagingError("Fine-tuned model config must be a JSON object")
    id_to_label = config.get("id2label")
    normalized = {str(index): str(label) for index, label in id_to_label.items()} if isinstance(id_to_label, dict) else {}
    if normalized != dict(labels) or normalized != dict(DEFAULT_LABELS):
        raise ModelPackagingError("Fine-tuned model config label mapping is incompatible")


def _validate_manifest(manifest: Mapping[str, Any], *, schema_path: Path) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelPackagingError("Unable to load the model artifact schema") from error
    Draft202012Validator.check_schema(schema)
    errors = list(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(manifest)
    )
    if errors:
        fields = sorted(
            ".".join(str(part) for part in error.absolute_path) or "<manifest>"
            for error in errors
        )
        raise ModelPackagingError(
            "Generated model manifest violates its schema at: " + ", ".join(fields)
        )


def _license_url(spdx_id: str) -> str:
    urls = {
        "MIT": "https://spdx.org/licenses/MIT.html",
        "Apache-2.0": "https://spdx.org/licenses/Apache-2.0.html",
    }
    try:
        return urls[spdx_id]
    except KeyError as error:
        raise ModelPackagingError(f"Unsupported candidate weight license: {spdx_id}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
