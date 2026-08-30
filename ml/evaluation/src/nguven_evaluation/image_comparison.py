/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Fail-closed robustness comparison for reviewed image detector runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from nguven_evaluation.evaluation import validate_result
from nguven_evaluation.image_benchmark import (
    DEFAULT_IMAGE_BENCHMARK_PATH,
    image_benchmark_evidence_allowed,
    load_image_benchmark_lock,
)
from nguven_evaluation.image_model_adapters import (
    DEFAULT_IMAGE_CANDIDATES_PATH,
    load_image_candidate_registry,
)


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_COMPARISON_SCHEMA_PATH = (
    EVALUATION_ROOT / "image" / "comparisons" / "schema.json"
)
MAX_IMAGE_RUN_METADATA_BYTES = 2 * 1024 * 1024


class ImageModelComparisonError(ValueError):
    """Raised when image benchmark runs are incomplete, altered, or incomparable."""


def load_image_benchmark_run(root: Path) -> dict[str, Any]:
    """Load one private run and verify every artifact hash used by its result."""
    if root.is_symlink() or not root.is_dir():
        raise ImageModelComparisonError("Image benchmark run must be a regular directory")
    run_path = root / "run.json"
    result_path = root / "result.json"
    manifest_path = root / "manifest.jsonl"
    predictions_path = root / "predictions.jsonl"
    run = _load_json_object(run_path, "image benchmark run")
    result = _load_json_object(result_path, "image evaluation result")
    try:
        validate_result(result)
    except ValueError as error:
        raise ImageModelComparisonError("Image evaluation result violates its contract") from error
    if run.get("schemaVersion") != "image-benchmark-run-v1":
        raise ImageModelComparisonError("Unsupported image benchmark run schema")
    expected_hashes = {
        "evaluationManifestSha256": _sha256_file(manifest_path),
        "predictionsSha256": _sha256_file(predictions_path),
    }
    for field, digest in expected_hashes.items():
        if run.get("artifacts", {}).get(field) != digest:
            raise ImageModelComparisonError(f"Image run {field} does not match local bytes")
    if result["artifacts"]["manifestSha256"] != expected_hashes["evaluationManifestSha256"]:
        raise ImageModelComparisonError("Image result manifest hash differs from run evidence")
    if result["artifacts"]["predictionsSha256"] != expected_hashes["predictionsSha256"]:
        raise ImageModelComparisonError("Image result prediction hash differs from run evidence")
    return {
        "run": run,
        "result": result,
        "resultSha256": _sha256_file(result_path),
    }


def compare_image_benchmark_runs(
    run_roots: Sequence[Path],
    *,
    benchmark_path: Path = DEFAULT_IMAGE_BENCHMARK_PATH,
    candidate_registry_path: Path = DEFAULT_IMAGE_CANDIDATES_PATH,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    schema_path: Path = DEFAULT_IMAGE_COMPARISON_SCHEMA_PATH,
) -> dict[str, Any]:
    """Compare private run directories under the exact reviewed benchmark lock."""
    benchmark = load_image_benchmark_lock(
        benchmark_path,
        candidate_registry_path=candidate_registry_path,
    )
    candidates = load_image_candidate_registry(candidate_registry_path)
    return compare_image_run_evidence(
        [load_image_benchmark_run(root) for root in run_roots],
        benchmark=benchmark,
        benchmark_sha256=_sha256_file(benchmark_path),
        expected_candidates={candidate.adapter_id: candidate.revision for candidate in candidates},
        now=now,
        schema_path=schema_path,
    )


def compare_image_run_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    benchmark: Mapping[str, Any],
    benchmark_sha256: str,
    expected_candidates: Mapping[str, str],
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    schema_path: Path = DEFAULT_IMAGE_COMPARISON_SCHEMA_PATH,
) -> dict[str, Any]:
    """Rank complete evidence using the frozen robustness-first selection order."""
    if len(evidence) != len(expected_candidates):
        raise ImageModelComparisonError("Comparison requires one run for every reviewed candidate")
    protocol = benchmark["protocol"]
    expected_transformations = tuple(protocol["transformations"])
    expected_originals = int(protocol["recordCount"])
    expected_predictions = expected_originals * len(expected_transformations)
    dataset_version = f"{benchmark['benchmarkId']}:{benchmark['version']}"
    shared_artifacts: tuple[str, str, str] | None = None
    seen_candidates: set[str] = set()
    seen_run_ids: set[str] = set()
    candidate_reports: list[dict[str, Any]] = []

    for package in evidence:
        run = package["run"]
        result = package["result"]
        try:
            validate_result(dict(result))
        except ValueError as error:
            raise ImageModelComparisonError("Image evaluation result violates its contract") from error
        adapter_id = str(run["candidate"]["adapterId"])
        if adapter_id in seen_candidates or adapter_id not in expected_candidates:
            raise ImageModelComparisonError("Image comparison candidate coverage is invalid")
        seen_candidates.add(adapter_id)
        revision = str(run["candidate"]["revision"])
        if revision != expected_candidates[adapter_id]:
            raise ImageModelComparisonError("Image candidate revision differs from the registry")
        if result["model"] != {"name": adapter_id, "version": revision}:
            raise ImageModelComparisonError("Image result model identity differs from run evidence")
        run_id = str(run["runId"])
        if run_id in seen_run_ids or result["run"]["id"] != run_id:
            raise ImageModelComparisonError("Image comparison run identity is invalid")
        seen_run_ids.add(run_id)
        if run["benchmark"]["id"] != benchmark["benchmarkId"] or run["benchmark"]["version"] != benchmark["version"]:
            raise ImageModelComparisonError("Image run benchmark identity differs from the lock")
        if run["artifacts"]["benchmarkSha256"] != benchmark_sha256:
            raise ImageModelComparisonError("Image run was not produced from this benchmark lock")
        if result["dataset"] != {"version": dataset_version, "recordCount": expected_predictions}:
            raise ImageModelComparisonError("Image result coverage differs from the frozen protocol")
        if run["counts"] != {"originals": expected_originals, "predictions": expected_predictions}:
            raise ImageModelComparisonError("Image run counts differ from the frozen protocol")

        comparable_artifacts = (
            str(run["artifacts"]["candidateRegistrySha256"]),
            str(run["artifacts"]["labelsSha256"]),
            str(run["artifacts"]["preprocessedManifestSha256"]),
        )
        if shared_artifacts is None:
            shared_artifacts = comparable_artifacts
        elif shared_artifacts != comparable_artifacts:
            raise ImageModelComparisonError("Image candidates did not use identical benchmark inputs")

        transformation_metrics = _transformation_metrics(
            result,
            expected_transformations=expected_transformations,
            expected_records=expected_originals,
        )
        metrics = result["metrics"]
        required_metrics = {
            "macroF1",
            "accuracy",
            "prAuc",
            "brierScore",
            "highConfidenceFalsePositiveRate",
            "p95InferenceMs",
        }
        if not required_metrics.issubset(metrics):
            raise ImageModelComparisonError("Image result lacks a required selection metric")
        worst = min(float(item["macroF1"]) for item in transformation_metrics)
        checks = {
            "macroF1": float(metrics["macroF1"]) >= float(protocol["acceptanceTargets"]["minimumMacroF1"]),
            "worstTransformationMacroF1": worst >= float(protocol["acceptanceTargets"]["minimumWorstTransformationMacroF1"]),
            "highConfidenceFalsePositiveRate": float(metrics["highConfidenceFalsePositiveRate"]) <= float(protocol["acceptanceTargets"]["maximumHighConfidenceFalsePositiveRate"]),
            "p95InferenceMs": float(metrics["p95InferenceMs"]) <= float(protocol["acceptanceTargets"]["maximumP95InferenceMs"]),
        }
        candidate_reports.append(
            {
                "rank": 0,
                "adapterId": adapter_id,
                "modelRevision": revision,
                "runId": run_id,
                "metrics": {
                    "macroF1": float(metrics["macroF1"]),
                    "accuracy": float(metrics["accuracy"]),
                    "prAuc": float(metrics["prAuc"]),
                    "brierScore": float(metrics["brierScore"]),
                    "highConfidenceFalsePositiveRate": float(metrics["highConfidenceFalsePositiveRate"]),
                    "p95InferenceMs": float(metrics["p95InferenceMs"]),
                    "worstTransformationMacroF1": worst,
                },
                "transformations": transformation_metrics,
                "acceptance": {"status": "passed" if all(checks.values()) else "failed", "checks": checks},
                "artifacts": {
                    "resultSha256": str(package["resultSha256"]),
                    "predictionsSha256": str(result["artifacts"]["predictionsSha256"]),
                },
            }
        )

    if seen_candidates != set(expected_candidates):
        raise ImageModelComparisonError("Comparison is missing a reviewed image candidate")
    candidate_reports.sort(key=_ranking_key)
    rank_by_performance: dict[tuple[float, float, float, float], int] = {}
    for candidate in candidate_reports:
        performance = _performance_key(candidate)
        rank_by_performance.setdefault(performance, len(rank_by_performance) + 1)
        candidate["rank"] = rank_by_performance[performance]
    qualified = [item for item in candidate_reports if item["acceptance"]["status"] == "passed"]
    leaders: list[str] = []
    if qualified:
        best = _performance_key(qualified[0])
        leaders = [item["adapterId"] for item in qualified if _performance_key(item) == best]
    if not leaders:
        selection_status = "no-qualified-candidate"
    elif len(leaders) > 1:
        selection_status = "tied"
    elif image_benchmark_evidence_allowed(benchmark):
        selection_status = "selected"
    else:
        selection_status = "experimental-leader"

    report = {
        "schemaVersion": "image-model-comparison-v1",
        "createdAt": now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "benchmark": {
            "id": benchmark["benchmarkId"],
            "version": benchmark["version"],
            "sha256": benchmark_sha256,
            "recordCount": expected_originals,
            "variantCount": expected_predictions,
            "evidenceAllowed": image_benchmark_evidence_allowed(benchmark),
        },
        "protocol": {
            "transformations": list(expected_transformations),
            "selectionOrder": list(protocol["selectionOrder"]),
            "acceptanceTargets": dict(protocol["acceptanceTargets"]),
        },
        "candidates": candidate_reports,
        "selection": {"status": selection_status, "leaders": leaders},
    }
    _validate_comparison(report, schema_path)
    return report


def write_image_comparison(report: Mapping[str, Any], output_path: Path) -> None:
    """Atomically write a text-free comparison report with owner-only permissions."""
    if output_path.exists() or output_path.is_symlink():
        raise ImageModelComparisonError("Image comparison output must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _transformation_metrics(
    result: Mapping[str, Any], *, expected_transformations: Sequence[str], expected_records: int
) -> list[dict[str, Any]]:
    slices = [item for item in result.get("slices", []) if item.get("dimension") == "transformation"]
    by_key = {str(item["key"]): item for item in slices}
    if len(by_key) != len(slices) or set(by_key) != set(expected_transformations):
        raise ImageModelComparisonError("Image result lacks the complete transformation slices")
    if any(int(item["recordCount"]) != expected_records for item in by_key.values()):
        raise ImageModelComparisonError("Image transformation slice coverage is incomplete")
    return [
        {
            "name": transformation,
            "recordCount": expected_records,
            "macroF1": float(by_key[transformation]["macroF1"]),
            "falsePositiveRate": float(by_key[transformation]["falsePositiveRate"]),
        }
        for transformation in expected_transformations
    ]


def _performance_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float]:
    metrics = candidate["metrics"]
    return (
        float(metrics["worstTransformationMacroF1"]),
        float(metrics["macroF1"]),
        -float(metrics["highConfidenceFalsePositiveRate"]),
        -float(metrics["p95InferenceMs"]),
    )


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    performance = _performance_key(candidate)
    return (-performance[0], -performance[1], -performance[2], -performance[3], str(candidate["adapterId"]))


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_IMAGE_RUN_METADATA_BYTES:
        raise ImageModelComparisonError(f"{label.title()} is unavailable or exceeds limits")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageModelComparisonError(f"Unable to parse {label}") from error
    if not isinstance(document, dict):
        raise ImageModelComparisonError(f"{label.title()} must be an object")
    return document


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ImageModelComparisonError("Image comparison artifact must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_comparison(report: Mapping[str, Any], schema_path: Path) -> None:
    schema = _load_json_object(schema_path, "image comparison schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(report),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise ImageModelComparisonError(f"Image comparison violates schema at {location}")
