"""Frozen external benchmark contract and evidence gate for image detectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from nguven_evaluation.image_model_adapters import DEFAULT_IMAGE_CANDIDATES_PATH
from nguven_evaluation.image_preprocessing import ROBUSTNESS_TRANSFORMATIONS


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_BENCHMARK_PATH = (
    EVALUATION_ROOT / "image" / "benchmarks" / "image-origin-robustness-v1.json"
)
DEFAULT_IMAGE_BENCHMARK_SCHEMA_PATH = (
    EVALUATION_ROOT / "image" / "benchmarks" / "schema.json"
)
MAX_IMAGE_BENCHMARK_BYTES = 256 * 1024


class ImageBenchmarkContractError(ValueError):
    """Raised when an image benchmark could permit invalid comparison evidence."""


def load_image_benchmark_lock(
    path: Path = DEFAULT_IMAGE_BENCHMARK_PATH,
    *,
    schema_path: Path = DEFAULT_IMAGE_BENCHMARK_SCHEMA_PATH,
    candidate_registry_path: Path = DEFAULT_IMAGE_CANDIDATES_PATH,
) -> dict[str, Any]:
    """Validate a source lock and bind it to the exact reviewed candidate registry."""
    lock = _load_json_object(path, "image benchmark lock")
    schema = _load_json_object(schema_path, "image benchmark schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(lock),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise ImageBenchmarkContractError(
            f"Image benchmark lock violates schema at {location}: {errors[0].message}"
        )
    _validate_semantics(lock, candidate_registry_path)
    return lock


def image_benchmark_evidence_allowed(lock: Mapping[str, Any]) -> bool:
    """Return true only after a reviewed, hash-bound materialization exists."""
    release = lock["release"]
    return bool(
        release["status"] == "materialized"
        and release["evidenceStatus"] == "reviewed"
        and release["resultsAllowed"] is True
        and release["materializedArtifact"] is not None
    )


def _validate_semantics(lock: Mapping[str, Any], registry_path: Path) -> None:
    registry_hash = _sha256_regular_file(registry_path, "candidate registry")
    if lock["candidateRegistrySha256"] != registry_hash:
        raise ImageBenchmarkContractError(
            "Image benchmark candidate registry hash does not match the reviewed lock"
        )

    subsets = lock["source"]["subsets"]
    identities = {(item["configuration"], item["label"]) for item in subsets}
    if identities != {("fp_450", "human"), ("syncred_600", "synthetic")}:
        raise ImageBenchmarkContractError("Image benchmark requires the reviewed human/synthetic subsets")
    if sum(int(item["recordCount"]) for item in subsets) != int(lock["protocol"]["recordCount"]):
        raise ImageBenchmarkContractError("Image benchmark subset counts do not match the protocol")

    transformations = tuple(lock["protocol"]["transformations"])
    if set(transformations) != set(ROBUSTNESS_TRANSFORMATIONS):
        raise ImageBenchmarkContractError("Image benchmark transformations differ from preprocessing")

    artifacts = lock["source"]["artifacts"]
    paths = [str(item["path"]) for item in artifacts]
    if len(paths) != len(set(paths)):
        raise ImageBenchmarkContractError("Image benchmark artifact paths must be unique")

    release = lock["release"]
    if release["resultsAllowed"] is True and not image_benchmark_evidence_allowed(lock):
        raise ImageBenchmarkContractError(
            "Image benchmark results cannot be enabled before reviewed materialization"
        )
    if release["status"] == "source-locked" and release["materializedArtifact"] is not None:
        raise ImageBenchmarkContractError(
            "Source-locked image benchmark cannot claim a materialized artifact"
        )


def _sha256_regular_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ImageBenchmarkContractError(f"{label.title()} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ImageBenchmarkContractError(f"{label.title()} must be a regular file")
    if path.stat().st_size > MAX_IMAGE_BENCHMARK_BYTES:
        raise ImageBenchmarkContractError(f"{label.title()} exceeds the size limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageBenchmarkContractError(f"Unable to parse {label}") from error
    if not isinstance(document, dict):
        raise ImageBenchmarkContractError(f"{label.title()} must be a JSON object")
    return document
