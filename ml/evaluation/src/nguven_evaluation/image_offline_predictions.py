"""Integrity-checked offline predictions for the frozen image robustness benchmark."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
from PIL import Image, UnidentifiedImageError

from nguven_evaluation.image_benchmark_materialization import DEFAULT_IMAGE_LABEL_SCHEMA_PATH
from nguven_evaluation.image_model_adapters import ImageModelAdapterError
from nguven_evaluation.image_preprocessing import (
    DEFAULT_IMAGE_PREPROCESSED_SCHEMA_PATH,
    DEFAULT_IMAGE_PREPROCESSING_VERSION,
    ROBUSTNESS_TRANSFORMATIONS,
)


MAX_IMAGE_MANIFEST_BYTES = 16 * 1024 * 1024


class ImageOfflinePredictionError(ValueError):
    """Raised when image benchmark inputs or predictions violate the protocol."""


@dataclass(frozen=True, slots=True)
class VerifiedImageVariant:
    source_id: str
    variant_id: str
    transformation: str
    path: Path


def load_preprocessed_image_variants(
    root: Path,
    *,
    schema_path: Path = DEFAULT_IMAGE_PREPROCESSED_SCHEMA_PATH,
) -> tuple[VerifiedImageVariant, ...]:
    """Verify the generated manifest and every local image variant byte-for-byte."""
    if root.is_symlink() or not root.is_dir():
        raise ImageOfflinePredictionError("Preprocessed image root must be a regular directory")
    manifest_path = root / "manifest.jsonl"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ImageOfflinePredictionError("Preprocessed image manifest is unavailable")
    if manifest_path.stat().st_size > MAX_IMAGE_MANIFEST_BYTES:
        raise ImageOfflinePredictionError("Preprocessed image manifest exceeds the size limit")
    validator = Draft202012Validator(_load_schema(schema_path, "preprocessed image"))
    variants: list[VerifiedImageVariant] = []
    seen: set[tuple[str, str]] = set()
    transformations: Counter[str] = Counter()
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ImageOfflinePredictionError(f"Invalid image manifest JSON at line {number}") from error
        if list(validator.iter_errors(record)):
            raise ImageOfflinePredictionError(f"Image manifest violates schema at line {number}")
        source_id = str(record["id"])
        transformation = str(record["transformation"])
        identity = (source_id, transformation)
        if identity in seen:
            raise ImageOfflinePredictionError("Duplicate image variant identity")
        seen.add(identity)
        relative = Path(str(record["path"]))
        path = root / relative
        _verify_variant_file(path, root, record)
        variant_id = f"{source_id}--{transformation}"
        if len(variant_id) > 256:
            raise ImageOfflinePredictionError("Image variant id exceeds prediction limits")
        variants.append(VerifiedImageVariant(source_id, variant_id, transformation, path))
        transformations[source_id] += 1
    if not variants:
        raise ImageOfflinePredictionError("Preprocessed image manifest contains no variants")
    expected_count = len(ROBUSTNESS_TRANSFORMATIONS)
    if any(count != expected_count for count in transformations.values()):
        raise ImageOfflinePredictionError("Every image must contain the complete robustness suite")
    return tuple(sorted(variants, key=lambda item: item.variant_id))


def load_image_benchmark_labels(
    path: Path,
    *,
    schema_path: Path = DEFAULT_IMAGE_LABEL_SCHEMA_PATH,
) -> dict[str, dict[str, Any]]:
    """Load private labels without exposing image content."""
    if path.is_symlink() or not path.is_file():
        raise ImageOfflinePredictionError("Image benchmark labels must be a regular file")
    validator = Draft202012Validator(_load_schema(schema_path, "image label"))
    labels: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ImageOfflinePredictionError(f"Invalid image label JSON at line {number}") from error
        if list(validator.iter_errors(record)):
            raise ImageOfflinePredictionError(f"Image label violates schema at line {number}")
        record_id = str(record["id"])
        if record_id in labels:
            raise ImageOfflinePredictionError(f"Duplicate image label id: {record_id}")
        labels[record_id] = record
    if not labels:
        raise ImageOfflinePredictionError("Image benchmark labels contain no records")
    return labels


def build_image_benchmark_predictions(
    variants: Sequence[VerifiedImageVariant],
    labels: Mapping[str, Mapping[str, Any]],
    *,
    adapter: Any,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce aligned evaluation-manifest and prediction records in stable order."""
    source_ids = {item.source_id for item in variants}
    if source_ids != set(labels):
        raise ImageOfflinePredictionError("Image variants and private labels have different coverage")
    manifest: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for variant in sorted(variants, key=lambda item: item.variant_id):
        label = labels[variant.source_id]
        try:
            with Image.open(variant.path) as source:
                source.load()
                image = source.convert("RGB")
            started = clock_ns()
            prediction = adapter.predict(image)
            finished = clock_ns()
        except (OSError, UnidentifiedImageError, ImageModelAdapterError) as error:
            raise ImageOfflinePredictionError(
                f"Image inference failed for variant id: {variant.variant_id}"
            ) from error
        if finished < started:
            raise ImageOfflinePredictionError("Monotonic image inference clock moved backwards")
        manifest.append(
            {
                "id": variant.variant_id,
                "label": label["label"],
                "sourceGroup": label["sourceGroup"],
                "generatorFamily": label["generatorFamily"],
                "transformation": variant.transformation,
            }
        )
        predictions.append(
            {
                "id": variant.variant_id,
                "predictedLabel": prediction.label,
                "score": prediction.score,
                "inferenceMs": (finished - started) / 1_000_000,
            }
        )
    return manifest, predictions


def _verify_variant_file(path: Path, root: Path, record: Mapping[str, Any]) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise ImageOfflinePredictionError("Image variant path escapes its private root") from error
    if path.is_symlink() or not path.is_file():
        raise ImageOfflinePredictionError("Image variant must be a regular file")
    if path.stat().st_size != int(record["sizeBytes"]):
        raise ImageOfflinePredictionError("Image variant size mismatch")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not hmac.compare_digest(digest, str(record["sha256"])):
        raise ImageOfflinePredictionError("Image variant hash mismatch")
    if record["preprocessingVersion"] != DEFAULT_IMAGE_PREPROCESSING_VERSION:
        raise ImageOfflinePredictionError("Image preprocessing version mismatch")


def _load_schema(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ImageOfflinePredictionError(f"{label.title()} schema is unavailable")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageOfflinePredictionError(f"Unable to parse {label} schema") from error
    if not isinstance(document, dict):
        raise ImageOfflinePredictionError(f"{label.title()} schema must be an object")
    Draft202012Validator.check_schema(document)
    return document
