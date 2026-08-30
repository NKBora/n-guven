"""Reviewed image candidate registry and model-independent adapter boundary."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from jsonschema import Draft202012Validator, FormatChecker
from PIL import Image

from nguven_evaluation.image_preprocessing import DEFAULT_IMAGE_PREPROCESSING_VERSION


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_CANDIDATES_PATH = EVALUATION_ROOT / "image" / "models" / "candidates.json"
DEFAULT_IMAGE_CANDIDATES_SCHEMA_PATH = (
    EVALUATION_ROOT / "image" / "models" / "candidates.schema.json"
)
MAX_CANDIDATE_REGISTRY_BYTES = 128 * 1024


class ImageModelAdapterError(ValueError):
    """Raised when reviewed candidate identity or inference output is invalid."""


@dataclass(frozen=True, slots=True)
class ImageCandidateDescriptor:
    adapter_id: str
    display_name: str
    provider: str
    repository: str
    revision: str
    weights_license: str
    architecture: str
    upstream_labels: Mapping[int, str]
    normalized_labels: Mapping[int, str]
    artifacts: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True, slots=True)
class ImagePrediction:
    label: str
    score: float


@runtime_checkable
class ImagePredictionBackend(Protocol):
    """Injected local backend; the adapter itself never downloads model artifacts."""

    def predict(self, image: Image.Image) -> ImagePrediction:
        """Classify one normalized RGB image."""


REVIEWED_CANDIDATES: Mapping[str, tuple[str, str, str, str]] = {
    "siglip2-aiornot": (
        "prithivMLmods/AIorNot-SigLIP2",
        "f4e6a281725e8dfb11a1d8c959b69737bba1e91d",
        "Apache-2.0",
        "SiglipForImageClassification",
    ),
    "vit-cifake": (
        "capcheck/ai-image-detection",
        "a6661e07d38f1a097bba07ca9415538819278f09",
        "Apache-2.0",
        "ViTForImageClassification",
    ),
}


def load_image_candidate_registry(
    path: Path = DEFAULT_IMAGE_CANDIDATES_PATH,
    *,
    schema_path: Path = DEFAULT_IMAGE_CANDIDATES_SCHEMA_PATH,
) -> tuple[ImageCandidateDescriptor, ...]:
    """Load the complete reviewed pair and reject identity or safety-policy drift."""
    document = _load_json_object(path, "candidate registry")
    schema = _load_json_object(schema_path, "candidate schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise ImageModelAdapterError(
            f"Image candidate registry violates schema at {location}: {errors[0].message}"
        )

    descriptors: list[ImageCandidateDescriptor] = []
    seen: set[str] = set()
    for candidate in document["candidates"]:
        adapter_id = str(candidate["adapterId"])
        if adapter_id in seen:
            raise ImageModelAdapterError(f"Duplicate image candidate: {adapter_id}")
        seen.add(adapter_id)
        expected = REVIEWED_CANDIDATES.get(adapter_id)
        if expected is None:
            raise ImageModelAdapterError(f"Unreviewed image candidate: {adapter_id}")
        upstream = candidate["upstream"]
        actual = (
            str(upstream["repository"]),
            str(upstream["revision"]),
            str(candidate["weightsLicense"]["spdxId"]),
            str(candidate["architecture"]),
        )
        if actual != expected:
            raise ImageModelAdapterError(
                f"Image candidate identity differs from reviewed pin: {adapter_id}"
            )
        descriptors.append(
            ImageCandidateDescriptor(
                adapter_id=adapter_id,
                display_name=str(candidate["displayName"]),
                provider=str(candidate["provider"]),
                repository=actual[0],
                revision=actual[1],
                weights_license=actual[2],
                architecture=actual[3],
                upstream_labels={int(key): str(value) for key, value in candidate["upstreamLabels"].items()},
                normalized_labels={int(key): str(value) for key, value in candidate["normalizedLabels"].items()},
                artifacts=tuple(
                    (str(item["path"]), int(item["sizeBytes"]), str(item["sha256"]))
                    for item in candidate["artifacts"]
                ),
            )
        )

    if seen != set(REVIEWED_CANDIDATES):
        raise ImageModelAdapterError("Image candidate registry must contain the reviewed pair")
    return tuple(sorted(descriptors, key=lambda item: item.adapter_id))


class CandidateImageModelAdapter:
    """Fail-closed normalization around one reviewed, locally loaded classifier."""

    def __init__(
        self,
        candidate: ImageCandidateDescriptor,
        backend: ImagePredictionBackend,
        *,
        preprocessing_version: str = DEFAULT_IMAGE_PREPROCESSING_VERSION,
    ) -> None:
        if candidate.adapter_id not in REVIEWED_CANDIDATES:
            raise ImageModelAdapterError("Image adapter requires a reviewed candidate")
        if preprocessing_version != DEFAULT_IMAGE_PREPROCESSING_VERSION:
            raise ImageModelAdapterError(
                f"Image adapter requires {DEFAULT_IMAGE_PREPROCESSING_VERSION}"
            )
        if not isinstance(backend, ImagePredictionBackend):
            raise ImageModelAdapterError("Image backend does not implement the prediction contract")
        self._candidate = candidate
        self._backend = backend
        self._preprocessing_version = preprocessing_version

    @property
    def adapter_id(self) -> str:
        return self._candidate.adapter_id

    @property
    def preprocessing_version(self) -> str:
        return self._preprocessing_version

    def predict(self, image: Image.Image) -> ImagePrediction:
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ImageModelAdapterError("Image adapter input must be a normalized RGB image")
        prediction = self._backend.predict(image)
        if not isinstance(prediction, ImagePrediction):
            raise ImageModelAdapterError("Image backend returned an invalid prediction type")
        if prediction.label not in set(self._candidate.normalized_labels.values()):
            raise ImageModelAdapterError("Image backend returned a label outside the registry")
        if not math.isfinite(prediction.score) or not 0 <= prediction.score <= 1:
            raise ImageModelAdapterError("Image backend score must be finite and within [0, 1]")
        return prediction


def _load_json_object(path: Path, document_name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ImageModelAdapterError(f"{document_name.title()} must be a regular file")
    if path.stat().st_size > MAX_CANDIDATE_REGISTRY_BYTES:
        raise ImageModelAdapterError(f"{document_name.title()} exceeds the size limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageModelAdapterError(f"Unable to parse {document_name}") from error
    if not isinstance(document, dict):
        raise ImageModelAdapterError(f"{document_name.title()} must be a JSON object")
    return document


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ImageModelAdapterError(f"Duplicate JSON key in image candidate registry: {key}")
        document[key] = value
    return document
