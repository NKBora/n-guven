"""Immutable candidate experiment specifications and execution gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from nguven_evaluation.benchmark import benchmark_evidence_allowed
from nguven_evaluation.model_adapters import CANDIDATES

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_SCHEMA_PATH = EVALUATION_ROOT / "experiments" / "schema.json"
DEFAULT_BERTURK_EXPERIMENT_PATH = EVALUATION_ROOT / "experiments" / "berturk-v1.json"
DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH = (
    EVALUATION_ROOT / "experiments" / "modernbert-tr-v1.json"
)
DEFAULT_MAX_EXPERIMENT_BYTES = 1024 * 1024


class ExperimentContractError(ValueError):
    """Raised when an experiment could violate candidate or evidence controls."""


def load_experiment_spec(
    path: Path,
    *,
    schema_path: Path = DEFAULT_EXPERIMENT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Load and validate one candidate-specific experiment specification."""
    document = _load_json_object(path, "experiment specification")
    schema = _load_json_object(schema_path, "experiment schema")
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        locations = sorted(
            ".".join(str(part) for part in error.absolute_path) or "<document>"
            for error in errors
        )
        raise ExperimentContractError(
            "Experiment specification validation failed at: " + ", ".join(locations)
        )
    _validate_candidate_identity(document)
    _validate_execution_state(document)
    return document


def validate_experiment_inputs(
    specification: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    require_execution_ready: bool = False,
) -> None:
    """Bind an experiment to the exact shared plan and benchmark release."""
    expected_benchmark = specification["benchmark"]
    if benchmark["benchmarkId"] != expected_benchmark["benchmarkId"]:
        raise ExperimentContractError("Experiment benchmark id mismatch")
    if benchmark["version"] != expected_benchmark["version"]:
        raise ExperimentContractError("Experiment benchmark version mismatch")
    if plan["dataset"]["version"] != f"{benchmark['benchmarkId']}-{benchmark['version']}":
        raise ExperimentContractError("Fine-tuning plan dataset version mismatch")
    protocol = specification["protocol"]
    if list(plan["protocol"]["seeds"]) != list(protocol["seeds"]):
        raise ExperimentContractError("Experiment seed order differs from the shared plan")
    if int(plan["protocol"]["maxSequenceLength"]) != int(protocol["maxSequenceLength"]):
        raise ExperimentContractError("Experiment sequence length differs from the shared plan")
    plan_candidates = {str(candidate["adapterId"]): candidate for candidate in plan["candidates"]}
    candidate = plan_candidates.get(str(specification["adapterId"]))
    if candidate is None or candidate["repository"] != specification["upstream"]["repository"]:
        raise ExperimentContractError("Experiment candidate differs from the shared plan")
    if candidate["revision"] != specification["upstream"]["revision"]:
        raise ExperimentContractError("Experiment candidate revision differs from the shared plan")

    if require_execution_ready:
        execution = specification["execution"]
        if execution["status"] != "ready" or execution["allowed"] is not True:
            raise ExperimentContractError("Experiment is not approved for execution")
        if not benchmark_evidence_allowed(benchmark):
            raise ExperimentContractError("Benchmark is not approved for result evidence")
        materialized = benchmark["release"]["materializedArtifact"]
        if materialized["manifestSha256"] != plan["dataset"]["manifestSha256"]:
            raise ExperimentContractError("Plan manifest hash differs from benchmark release")
        if materialized["preprocessedSha256"] != plan["dataset"]["preprocessedSha256"]:
            raise ExperimentContractError("Plan preprocessing hash differs from benchmark release")


def experiment_execution_allowed(
    specification: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> bool:
    """Return true only when both candidate and benchmark reviews are complete."""
    execution = specification["execution"]
    return bool(
        execution["status"] == "ready"
        and execution["allowed"] is True
        and benchmark_evidence_allowed(benchmark)
    )


def validate_comparable_experiments(
    specifications: list[Mapping[str, Any]],
) -> None:
    """Reject any candidate-specific drift that would invalidate the comparison."""
    if len(specifications) != 2:
        raise ExperimentContractError("Comparison requires exactly two experiments")
    by_adapter = {str(item["adapterId"]): item for item in specifications}
    if set(by_adapter) != set(CANDIDATES):
        raise ExperimentContractError(
            "Comparison requires exactly BERTurk and ModernBERT-TR experiments"
        )
    if len({str(item["experimentId"]) for item in specifications}) != 2:
        raise ExperimentContractError("Comparison experiment ids must be unique")

    baseline = by_adapter["berturk"]
    candidate = by_adapter["modernbert-tr"]
    for field, label in (
        ("benchmark", "benchmark"),
        ("protocol", "training protocol"),
        ("acceptance", "acceptance thresholds"),
    ):
        if baseline[field] != candidate[field]:
            raise ExperimentContractError(
                f"Candidate experiments have different {label}"
            )
    baseline_execution = baseline["execution"]
    candidate_execution = candidate["execution"]
    if (
        baseline_execution["status"] != candidate_execution["status"]
        or baseline_execution["allowed"] != candidate_execution["allowed"]
    ):
        raise ExperimentContractError("Candidate experiment execution states differ")


def _validate_candidate_identity(specification: Mapping[str, Any]) -> None:
    adapter_id = str(specification["adapterId"])
    candidate = CANDIDATES[adapter_id]
    upstream = specification["upstream"]
    expected = (candidate.repository, candidate.revision, candidate.weights_license)
    actual = (upstream["repository"], upstream["revision"], upstream["weightsLicense"])
    if actual != expected:
        raise ExperimentContractError(
            f"Experiment upstream identity mismatch for {adapter_id}"
        )


def _validate_execution_state(specification: Mapping[str, Any]) -> None:
    execution = specification["execution"]
    status = execution["status"]
    if status == "awaiting-reviewed-data" and (
        execution["allowed"] is True or execution["resultArtifact"] is not None
    ):
        raise ExperimentContractError(
            "Awaiting experiment cannot run or claim a result artifact"
        )
    if status == "ready" and (
        execution["allowed"] is not True or execution["resultArtifact"] is not None
    ):
        raise ExperimentContractError(
            "Ready experiment must allow execution without claiming results"
        )
    if status == "completed" and (
        execution["allowed"] is True or execution["resultArtifact"] is None
    ):
        raise ExperimentContractError(
            "Completed experiment must be closed and bind a result artifact"
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentContractError(f"{label.title()} must be a regular file")
    if path.stat().st_size > DEFAULT_MAX_EXPERIMENT_BYTES:
        raise ExperimentContractError(f"{label.title()} exceeds the size limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentContractError(f"{label.title()} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ExperimentContractError(f"{label.title()} must be a JSON object")
    return document
