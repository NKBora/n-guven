from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import Any, Mapping, Protocol
import unicodedata

from anyio import to_thread

from app.schemas.analysis import (
    ConfidenceLevel,
    TextAnalysisRequest,
    TextAnalysisResponse,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_PATH = SERVICE_ROOT / "releases" / "berturk-text-origin-v1.json"
PREPROCESSING_VERSION = "tr-text-v1"
PROBABILITY_EPSILON = 1e-7
MAX_METADATA_BYTES = 1024 * 1024


class ReleaseVerificationError(ValueError):
    """Raised when a configured release cannot be proven to match its registry."""


class SyntheticProbabilityPredictor(Protocol):
    def predict_synthetic_probability(self, text: str) -> float:
        """Return the uncalibrated probability for the synthetic label."""
        ...


@dataclass(frozen=True, slots=True)
class VerifiedBerturkRelease:
    release_id: str
    model_id: str
    threshold_version: str
    temperature: float
    low_maximum: float
    high_minimum: float
    max_sequence_length: int
    labels: Mapping[str, str]
    artifact_root: Path


def load_verified_berturk_release(
    artifact_root: Path,
    *,
    release_path: Path = DEFAULT_RELEASE_PATH,
) -> VerifiedBerturkRelease:
    """Verify the registered release and every mounted artifact before model loading."""
    release = _load_json_object(release_path, "Release registry")
    if release.get("stage") != "prototype":
        raise ReleaseVerificationError("Only the registered prototype stage is supported")

    root = _require_directory(artifact_root)
    layout = _mapping(release, "artifactLayout")
    manifest_path = root / _safe_filename(layout, "modelManifest")
    calibration_path = root / _safe_filename(layout, "calibration")
    model_release = _mapping(release, "model")
    calibration_release = _mapping(release, "calibration")

    _verify_file_hash(
        manifest_path,
        str(model_release.get("manifestSha256", "")),
        "Model manifest",
    )
    _verify_file_hash(
        calibration_path,
        str(calibration_release.get("artifactSha256", "")),
        "Calibration artifact",
    )
    manifest = _load_json_object(manifest_path, "Model manifest")
    calibration = _load_json_object(calibration_path, "Calibration artifact")
    _verify_model_identity(release, manifest)
    _verify_calibration_identity(release, calibration)
    _verify_declared_artifacts(root, manifest)

    runtime = _mapping(manifest, "runtime")
    thresholds = _mapping(release, "thresholds")
    labels = _mapping(manifest, "labels")
    return VerifiedBerturkRelease(
        release_id=str(release["releaseId"]),
        model_id=str(manifest["modelId"]),
        threshold_version=str(thresholds["version"]),
        temperature=float(calibration["temperature"]),
        low_maximum=float(thresholds["lowMaximum"]),
        high_minimum=float(thresholds["highMinimum"]),
        max_sequence_length=int(runtime["maxSequenceLength"]),
        labels={str(index): str(label) for index, label in labels.items()},
        artifact_root=root,
    )


class LocalBerturkPredictor:
    """Local-only Transformers backend for a previously verified BERTurk bundle."""

    def __init__(self, release: VerifiedBerturkRelease) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ReleaseVerificationError(
                'BERTurk runtime dependencies are missing; install ".[inference]"'
            ) from error

        indexed_labels = {int(index): label for index, label in release.labels.items()}
        synthetic_indexes = [
            index for index, label in indexed_labels.items() if label == "synthetic"
        ]
        if len(synthetic_indexes) != 1:
            raise ReleaseVerificationError(
                "Verified model labels must contain one synthetic class"
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(release.artifact_root),
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(release.artifact_root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
        except (OSError, TypeError, ValueError) as error:
            raise ReleaseVerificationError(
                "Unable to load the verified local BERTurk artifact"
            ) from error
        if int(model.config.num_labels) != len(indexed_labels):
            raise ReleaseVerificationError(
                "Model output count does not match the verified label map"
            )

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._model.eval()
        self._synthetic_index = synthetic_indexes[0]
        self._max_sequence_length = release.max_sequence_length
        self._lock = Lock()

    def predict_synthetic_probability(self, text: str) -> float:
        try:
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_sequence_length,
            )
            with self._lock, self._torch.inference_mode():
                logits = self._model(**encoded).logits[0]
                probability = self._torch.softmax(logits, dim=-1)[
                    self._synthetic_index
                ]
            value = float(probability.item())
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise ReleaseVerificationError("Local BERTurk inference failed") from error
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ReleaseVerificationError("BERTurk returned an invalid probability")
        return value


class BerturkTextInferenceService:
    """Calibrated service adapter that exposes synthetic-text likelihood."""

    def __init__(
        self,
        release: VerifiedBerturkRelease,
        predictor: SyntheticProbabilityPredictor,
    ) -> None:
        self._release = release
        self._predictor = predictor

    async def analyze(self, request: TextAnalysisRequest) -> TextAnalysisResponse:
        text = preprocess_turkish_text(request.text)
        started_at = perf_counter_ns()
        raw_probability = await to_thread.run_sync(
            self._predictor.predict_synthetic_probability,
            text,
        )
        inference_ms = max(0, round((perf_counter_ns() - started_at) / 1_000_000))
        score = temperature_scale(raw_probability, self._release.temperature)
        return TextAnalysisResponse(
            analysis_id=request.analysis_id,
            score=score,
            confidence_level=_confidence_level(score, self._release),
            model_version=self._release.model_id,
            threshold_version=self._release.threshold_version,
            inference_ms=inference_ms,
            explanation=(
                "Calibrated BERTurk synthetic-text likelihood; advisory signal only."
            ),
        )


def preprocess_turkish_text(text: str) -> str:
    """Apply the exact tr-text-v1 inference preprocessing contract."""
    if "\x00" in text:
        raise ValueError("Text input must not contain NUL characters")
    normalized = text.removeprefix("\ufeff")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized)
    if not normalized:
        raise ValueError("Text is empty after preprocessing")
    return normalized


def temperature_scale(probability: float, temperature: float) -> float:
    """Apply the validation-fitted scalar temperature to a binary probability."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and within [0, 1]")
    if not math.isfinite(temperature) or not 0.05 <= temperature <= 10:
        raise ValueError("Temperature must be finite and within [0.05, 10]")
    clipped = min(max(probability, PROBABILITY_EPSILON), 1 - PROBABILITY_EPSILON)
    logit = math.log(clipped / (1 - clipped))
    return 1 / (1 + math.exp(-logit / temperature))


def _confidence_level(
    score: float,
    release: VerifiedBerturkRelease,
) -> ConfidenceLevel:
    if score <= release.low_maximum:
        return ConfidenceLevel.LOW
    if score >= release.high_minimum:
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.UNCERTAIN


def _verify_model_identity(
    release: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    selected = _mapping(release, "model")
    fine_tuning = _mapping(manifest, "fineTuning")
    runtime = _mapping(manifest, "runtime")
    upstream = _mapping(manifest, "upstream")
    labels = _mapping(manifest, "labels")
    checks = (
        (manifest.get("modelId"), selected.get("modelId"), "model id"),
        (manifest.get("adapterId"), "berturk", "adapter id"),
        (manifest.get("preprocessingVersion"), PREPROCESSING_VERSION, "preprocessing"),
        (fine_tuning.get("seed"), selected.get("seed"), "seed"),
        (runtime.get("artifactFormat"), "safetensors", "artifact format"),
        (
            upstream.get("repository"),
            "dbmdz/bert-base-turkish-cased",
            "upstream repository",
        ),
        (
            upstream.get("revision"),
            "b6e1de16c983e0f2c70664591ea3f22810072608",
            "upstream revision",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ReleaseVerificationError(f"Registered BERTurk {label} mismatch")
    if labels != {"0": "human", "1": "synthetic"}:
        raise ReleaseVerificationError("Registered BERTurk label map mismatch")

    weight_entries = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("role") == "weights"
    ]
    if len(weight_entries) != 1:
        raise ReleaseVerificationError("Model manifest must declare one weights artifact")
    weights = weight_entries[0]
    if (
        weights.get("sha256") != selected.get("weightsSha256")
        or weights.get("sizeBytes") != selected.get("weightsSizeBytes")
    ):
        raise ReleaseVerificationError("Registered BERTurk weights identity mismatch")


def _verify_calibration_identity(
    release: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> None:
    selected = _mapping(release, "calibration")
    model = _mapping(calibration, "model")
    artifacts = _mapping(calibration, "artifacts")
    checks = (
        (calibration.get("method"), "temperature-scaling", "method"),
        (calibration.get("fittedSplit"), "validation", "split"),
        (model.get("name"), "berturk", "model"),
        (calibration.get("temperature"), selected.get("temperature"), "temperature"),
        (
            artifacts.get("manifestSha256"),
            selected.get("validationManifestSha256"),
            "validation manifest",
        ),
        (
            artifacts.get("predictionsSha256"),
            selected.get("validationPredictionsSha256"),
            "validation predictions",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ReleaseVerificationError(f"Registered calibration {label} mismatch")


def _verify_declared_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ReleaseVerificationError("Model manifest has no artifacts")
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping):
            raise ReleaseVerificationError("Model manifest contains an invalid artifact")
        relative = _safe_relative_path(str(item.get("path", "")))
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise ReleaseVerificationError("Model manifest contains duplicate artifacts")
        seen.add(relative_text)
        candidate = root / relative
        _reject_symlink_components(root, relative)
        _verify_file_hash(candidate, str(item.get("sha256", "")), relative_text)
        if not candidate.resolve(strict=True).is_relative_to(root):
            raise ReleaseVerificationError(f"Artifact escapes model root: {relative_text}")
        if candidate.stat().st_size != item.get("sizeBytes"):
            raise ReleaseVerificationError(f"Artifact size mismatch: {relative_text}")


def _require_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ReleaseVerificationError(
            "Model artifact root must be an existing non-symbolic-link directory"
        )
    return path.resolve(strict=True)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ReleaseVerificationError(f"{label} exceeds the metadata size limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"{label} must be a JSON object")
    return value


def _verify_file_hash(path: Path, expected: str, label: str) -> None:
    _require_regular_file(path, label)
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ReleaseVerificationError(f"{label} has an invalid registered SHA-256")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ReleaseVerificationError(f"{label} hash mismatch")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReleaseVerificationError(
            f"{label} must be an existing non-symbolic-link file"
        )


def _reject_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ReleaseVerificationError(
                f"Artifact path contains a symbolic link: {relative.as_posix()}"
            )


def _mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise ReleaseVerificationError(f"Release metadata is missing {key}")
    return value


def _safe_filename(document: Mapping[str, Any], key: str) -> str:
    value = str(document.get(key, ""))
    if not value or Path(value).name != value:
        raise ReleaseVerificationError(f"Artifact layout contains an unsafe {key}")
    return value


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ReleaseVerificationError("Model manifest contains an unsafe artifact path")
    return path
