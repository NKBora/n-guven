/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Reproducible, offline execution of the frozen image robustness benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from nguven_evaluation.evaluation import EvaluationMetadata, evaluate_predictions
from nguven_evaluation.image_benchmark import (
    DEFAULT_IMAGE_BENCHMARK_PATH,
    image_benchmark_evidence_allowed,
    load_image_benchmark_lock,
)
from nguven_evaluation.image_model_adapters import (
    DEFAULT_IMAGE_CANDIDATES_PATH,
    CandidateImageModelAdapter,
    ImageCandidateDescriptor,
    ImageModelAdapterError,
    load_image_candidate_registry,
)
from nguven_evaluation.image_offline_predictions import (
    ImageOfflinePredictionError,
    VerifiedImageVariant,
    build_image_benchmark_predictions,
    load_image_benchmark_labels,
    load_preprocessed_image_variants,
)
from nguven_evaluation.image_transformers_backend import LocalTransformersImageBackend


class ImageBenchmarkRunError(ValueError):
    """Raised when a benchmark run would be incomplete or non-comparable."""


def run_image_benchmark(
    *,
    adapter_id: str,
    preprocessed_root: Path,
    labels_path: Path,
    artifact_root: Path,
    output_root: Path,
    run_id: str,
    git_commit: str,
    device: str = "cpu",
    seed: int = 42,
    warmup_iterations: int = 3,
    benchmark_path: Path = DEFAULT_IMAGE_BENCHMARK_PATH,
    candidate_registry_path: Path = DEFAULT_IMAGE_CANDIDATES_PATH,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Load reviewed local artifacts and execute one candidate without network access."""
    benchmark = load_image_benchmark_lock(
        benchmark_path,
        candidate_registry_path=candidate_registry_path,
    )
    candidates = load_image_candidate_registry(candidate_registry_path)
    candidate = _select_candidate(candidates, adapter_id)
    backend = LocalTransformersImageBackend(candidate, artifact_root, device=device)
    adapter = CandidateImageModelAdapter(candidate, backend)
    variants = load_preprocessed_image_variants(preprocessed_root)
    labels = load_image_benchmark_labels(labels_path)
    _validate_protocol_coverage(benchmark, variants, labels)
    _warm_up(adapter, variants, warmup_iterations)
    manifest, predictions = build_image_benchmark_predictions(
        variants,
        labels,
        adapter=adapter,
    )
    return write_image_benchmark_run(
        manifest,
        predictions,
        benchmark=benchmark,
        candidate=candidate,
        output_root=output_root,
        run_id=run_id,
        git_commit=git_commit,
        seed=seed,
        device=device,
        benchmark_sha256=_sha256_file(benchmark_path),
        candidate_registry_sha256=_sha256_file(candidate_registry_path),
        labels_sha256=_sha256_file(labels_path),
        preprocessed_manifest_sha256=_sha256_file(preprocessed_root / "manifest.jsonl"),
        now=now,
    )


def write_image_benchmark_run(
    manifest: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    benchmark: Mapping[str, Any],
    candidate: ImageCandidateDescriptor,
    output_root: Path,
    run_id: str,
    git_commit: str,
    seed: int,
    device: str,
    benchmark_sha256: str,
    candidate_registry_sha256: str,
    labels_sha256: str,
    preprocessed_manifest_sha256: str,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Evaluate and atomically publish private predictions plus traceable metadata."""
    if output_root.exists() or output_root.is_symlink():
        raise ImageBenchmarkRunError("Image benchmark output must not already exist")
    if device not in {"cpu", "mps", "cuda"}:
        raise ImageBenchmarkRunError("Image benchmark device must be cpu, mps, or cuda")
    _validate_run_identity(run_id, git_commit)

    manifest_bytes = _jsonl_bytes(manifest)
    prediction_bytes = _jsonl_bytes(predictions)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    predictions_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    created_at = now().astimezone(timezone.utc)
    dataset_version = f"{benchmark['benchmarkId']}:{benchmark['version']}"
    result = evaluate_predictions(
        manifest,
        predictions,
        metadata=EvaluationMetadata(
            run_id=run_id,
            git_commit=git_commit,
            seed=seed,
            dataset_version=dataset_version,
            model_name=candidate.adapter_id,
            model_version=candidate.revision,
        ),
        manifest_sha256=manifest_sha256,
        predictions_sha256=predictions_sha256,
        now=lambda: created_at,
    )
    run = {
        "schemaVersion": "image-benchmark-run-v1",
        "runId": run_id,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        "gitCommit": git_commit,
        "seed": seed,
        "device": device,
        "candidate": {
            "adapterId": candidate.adapter_id,
            "repository": candidate.repository,
            "revision": candidate.revision,
        },
        "benchmark": {
            "id": benchmark["benchmarkId"],
            "version": benchmark["version"],
            "evidenceAllowed": image_benchmark_evidence_allowed(benchmark),
            "claimStatus": (
                "reviewed-evidence"
                if image_benchmark_evidence_allowed(benchmark)
                else "experimental-unreviewed"
            ),
        },
        "counts": {
            "originals": int(benchmark["protocol"]["recordCount"]),
            "predictions": len(predictions),
        },
        "artifacts": {
            "benchmarkSha256": benchmark_sha256,
            "candidateRegistrySha256": candidate_registry_sha256,
            "labelsSha256": labels_sha256,
            "preprocessedManifestSha256": preprocessed_manifest_sha256,
            "evaluationManifestSha256": manifest_sha256,
            "predictionsSha256": predictions_sha256,
        },
    }
    _write_atomic_directory(
        output_root,
        {
            "manifest.jsonl": manifest_bytes,
            "predictions.jsonl": prediction_bytes,
            "result.json": _json_bytes(result),
            "run.json": _json_bytes(run),
        },
    )
    return {"run": run, "result": result}


def _select_candidate(
    candidates: Sequence[ImageCandidateDescriptor], adapter_id: str
) -> ImageCandidateDescriptor:
    matches = [candidate for candidate in candidates if candidate.adapter_id == adapter_id]
    if len(matches) != 1:
        raise ImageBenchmarkRunError(f"Unknown reviewed image adapter: {adapter_id}")
    return matches[0]


def _validate_protocol_coverage(
    benchmark: Mapping[str, Any],
    variants: Sequence[VerifiedImageVariant],
    labels: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_originals = int(benchmark["protocol"]["recordCount"])
    expected_predictions = expected_originals * len(benchmark["protocol"]["transformations"])
    if len(labels) != expected_originals:
        raise ImageBenchmarkRunError("Private label count differs from the frozen protocol")
    if len(variants) != expected_predictions:
        raise ImageBenchmarkRunError("Image variant count differs from the frozen protocol")


def _warm_up(
    adapter: CandidateImageModelAdapter,
    variants: Sequence[VerifiedImageVariant],
    iterations: int,
) -> None:
    if not 0 <= iterations <= 100:
        raise ImageBenchmarkRunError("Warm-up iterations must be within [0, 100]")
    if not iterations:
        return
    variant = next(
        (item for item in variants if item.transformation == "canonical"),
        variants[0],
    )
    try:
        with Image.open(variant.path) as source:
            source.load()
            image = source.convert("RGB")
        for _ in range(iterations):
            adapter.predict(image)
    except (OSError, UnidentifiedImageError, ImageModelAdapterError) as error:
        raise ImageBenchmarkRunError("Image model warm-up failed") from error


def _validate_run_identity(run_id: str, git_commit: str) -> None:
    if not run_id or len(run_id) > 256 or any(character.isspace() for character in run_id):
        raise ImageBenchmarkRunError("Run id must be a non-empty token of at most 256 characters")
    if not 7 <= len(git_commit) <= 64 or any(
        character not in "0123456789abcdefABCDEF" for character in git_commit
    ):
        raise ImageBenchmarkRunError("Git commit must contain 7 to 64 hexadecimal characters")


def _write_atomic_directory(output_root: Path, files: Mapping[str, bytes]) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    os.chmod(temporary_root, 0o700)
    try:
        for name, content in files.items():
            destination = temporary_root / name
            with destination.open("xb") as stream:
                stream.write(content)
            os.chmod(destination, 0o600)
        os.replace(temporary_root, output_root)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if not records:
        raise ImageBenchmarkRunError("Image benchmark artifact cannot be empty")
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ImageBenchmarkRunError("Image benchmark provenance artifact is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
