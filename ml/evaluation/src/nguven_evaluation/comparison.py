"""Evidence-preserving aggregation for fair fine-tuned model comparisons."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from math import sqrt
from statistics import fmean, pstdev, stdev
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from nguven_evaluation.evaluation import validate_result
from nguven_evaluation.finetuning import load_finetuning_plan, write_private_json

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_SCHEMA_PATH = EVALUATION_ROOT / "comparisons" / "schema.json"
METRIC_NAMES = (
    "accuracy",
    "macroPrecision",
    "macroRecall",
    "macroF1",
    "meanInferenceMs",
)
ADVANCED_METRIC_NAMES = (
    "prAuc",
    "brierScore",
    "ece",
    "highConfidenceFalsePositiveRate",
    "p95InferenceMs",
)
ACCEPTANCE_TARGETS = {
    "minimumMacroF1": 0.8,
    "maximumHighConfidenceFalsePositiveRate": 0.05,
    "maximumP95LatencyMs": 3000.0,
}


class ModelComparisonError(ValueError):
    """Raised when candidate results are not comparable under one protocol."""


def load_evaluation_result(path: Path) -> dict[str, Any]:
    """Load one explicit local evaluation result without following symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise ModelComparisonError("Evaluation result must be a regular non-symbolic-link file")
    if path.stat().st_size > 1024 * 1024:
        raise ModelComparisonError("Evaluation result exceeds the size limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelComparisonError("Evaluation result must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ModelComparisonError("Evaluation result must be a JSON object")
    try:
        validate_result(document)
    except ValueError as error:
        raise ModelComparisonError("Evaluation result violates its contract") from error
    return document


def compare_evaluation_results(
    results: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    schema_path: Path = DEFAULT_COMPARISON_SCHEMA_PATH,
) -> dict[str, Any]:
    """Aggregate identical seed sets and rank candidates without hiding variance."""
    expected_candidates = {str(item["adapterId"]) for item in plan["candidates"]}
    expected_seeds = {int(seed) for seed in plan["protocol"]["seeds"]}
    if len(expected_candidates) < 2:
        raise ModelComparisonError("Comparison plan must contain at least two candidates")
    advanced_complete = [
        all(metric in result.get("metrics", {}) for metric in ADVANCED_METRIC_NAMES)
        for result in results
    ]
    advanced_any = [
        any(metric in result.get("metrics", {}) for metric in ADVANCED_METRIC_NAMES)
        for result in results
    ]
    if any(advanced_any) and not all(advanced_complete):
        raise ModelComparisonError(
            "Advanced metric coverage must be identical across every candidate and seed"
        )
    metric_names = METRIC_NAMES + ADVANCED_METRIC_NAMES if all(advanced_complete) else METRIC_NAMES

    grouped: dict[str, list[Mapping[str, Any]]] = {
        candidate: [] for candidate in expected_candidates
    }
    seen_run_ids: set[str] = set()
    seen_prediction_hashes: set[str] = set()
    shared_dataset: tuple[str, int, str] | None = None
    for result in results:
        try:
            validate_result(dict(result))
        except ValueError as error:
            raise ModelComparisonError("Evaluation result violates its contract") from error
        adapter_id = str(result["model"]["name"])
        if adapter_id not in grouped:
            raise ModelComparisonError(f"Unexpected comparison candidate: {adapter_id}")
        if result["dataset"]["version"] != plan["dataset"]["version"]:
            raise ModelComparisonError("Evaluation result dataset version differs from the plan")
        run_id = str(result["run"]["id"])
        if run_id in seen_run_ids:
            raise ModelComparisonError(f"Duplicate evaluation run id: {run_id}")
        seen_run_ids.add(run_id)
        prediction_hash = str(result["artifacts"]["predictionsSha256"]).lower()
        if prediction_hash in seen_prediction_hashes:
            raise ModelComparisonError("Comparison results reuse a prediction artifact")
        seen_prediction_hashes.add(prediction_hash)
        dataset_identity = (
            str(result["dataset"]["version"]),
            int(result["dataset"]["recordCount"]),
            str(result["artifacts"]["manifestSha256"]).lower(),
        )
        if shared_dataset is None:
            shared_dataset = dataset_identity
        elif dataset_identity != shared_dataset:
            raise ModelComparisonError("Comparison results do not use the same test dataset")
        grouped[adapter_id].append(result)

    if shared_dataset is None:
        raise ModelComparisonError("No evaluation results were supplied")

    candidate_reports: list[dict[str, Any]] = []
    for adapter_id in sorted(grouped):
        candidate_results = grouped[adapter_id]
        if not candidate_results:
            raise ModelComparisonError(f"Missing evaluation results for: {adapter_id}")
        seeds = [int(result["run"]["seed"]) for result in candidate_results]
        if len(seeds) != len(set(seeds)):
            raise ModelComparisonError(f"Duplicate seed result for: {adapter_id}")
        if set(seeds) != expected_seeds:
            raise ModelComparisonError(f"Seed coverage mismatch for: {adapter_id}")
        versions = {str(result["model"]["version"]) for result in candidate_results}
        if len(versions) != 1:
            raise ModelComparisonError(f"Multiple model versions supplied for: {adapter_id}")
        ordered_results = sorted(candidate_results, key=lambda result: int(result["run"]["seed"]))
        metrics = {
            metric: _summarize(
                [float(result["metrics"][metric]) for result in ordered_results]
            )
            for metric in metric_names
        }
        candidate_reports.append(
            {
                "rank": 0,
                "adapterId": adapter_id,
                "modelVersion": versions.pop(),
                "runIds": [str(result["run"]["id"]) for result in ordered_results],
                "metrics": metrics,
                "predictionArtifacts": [
                    str(result["artifacts"]["predictionsSha256"]).lower()
                    for result in ordered_results
                ],
            }
        )

    candidate_reports.sort(key=_ranking_key)
    performance_keys = [_performance_key(candidate) for candidate in candidate_reports]
    rank_by_performance: dict[tuple[float, float, float], int] = {}
    for candidate, performance in zip(candidate_reports, performance_keys):
        rank_by_performance.setdefault(performance, len(rank_by_performance) + 1)
        candidate["rank"] = rank_by_performance[performance]
    best_performance = performance_keys[0]
    leaders = [
        candidate["adapterId"]
        for candidate in candidate_reports
        if _performance_key(candidate) == best_performance
    ]
    report = {
        "schemaVersion": "1.0",
        "createdAt": now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "planId": plan["planId"],
        "dataset": {
            "version": shared_dataset[0],
            "recordCount": shared_dataset[1],
            "manifestSha256": shared_dataset[2],
        },
        "protocol": {
            "seeds": sorted(expected_seeds),
            "primaryMetric": "macroF1",
            "tieBreakers": ["accuracy", "meanInferenceMs"],
        },
        "candidates": candidate_reports,
        "selection": {
            "status": "selected" if len(leaders) == 1 else "tied",
            "leaders": leaders,
        },
        "acceptance": _acceptance_report(
            candidate_reports, advanced_metrics=all(advanced_complete)
        ),
    }
    _validate_comparison(report, schema_path=schema_path)
    return report


def compare_result_files(
    result_paths: Sequence[Path],
    *,
    plan_path: Path,
) -> dict[str, Any]:
    """Load a plan and result files before building a comparison report."""
    plan = load_finetuning_plan(plan_path)
    results = [load_evaluation_result(path) for path in result_paths]
    return compare_evaluation_results(results, plan=plan)


def write_comparison_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Write the validated evidence report atomically."""
    write_private_json(report, output_path)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    mean = fmean(values)
    if len(values) == 1:
        margin = 0.0
    else:
        margin = _t_critical_95(len(values) - 1) * stdev(values) / sqrt(len(values))
    return {
        "mean": mean,
        "populationStdDev": pstdev(values),
        "confidence95Low": max(0.0, mean - margin),
        "confidence95High": mean + margin,
    }


def _acceptance_report(
    candidates: Sequence[Mapping[str, Any]], *, advanced_metrics: bool
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        checks: dict[str, bool | None] = {
            "macroF1": float(metrics["macroF1"]["mean"])
            >= ACCEPTANCE_TARGETS["minimumMacroF1"],
            "highConfidenceFalsePositiveRate": None,
            "p95InferenceMs": None,
        }
        if advanced_metrics:
            checks["highConfidenceFalsePositiveRate"] = (
                float(metrics["highConfidenceFalsePositiveRate"]["mean"])
                <= ACCEPTANCE_TARGETS["maximumHighConfidenceFalsePositiveRate"]
            )
            checks["p95InferenceMs"] = (
                float(metrics["p95InferenceMs"]["mean"])
                <= ACCEPTANCE_TARGETS["maximumP95LatencyMs"]
            )
        values = list(checks.values())
        status = (
            "insufficient-evidence"
            if any(value is None for value in values)
            else "passed" if all(values) else "failed"
        )
        reports.append(
            {
                "adapterId": candidate["adapterId"],
                "status": status,
                "checks": checks,
            }
        )
    return {"targets": dict(ACCEPTANCE_TARGETS), "candidates": reports}


def _t_critical_95(degrees_of_freedom: int) -> float:
    values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
    }
    return values.get(degrees_of_freedom, 1.96)


def _performance_key(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    return (
        float(metrics["macroF1"]["mean"]),
        float(metrics["accuracy"]["mean"]),
        -float(metrics["meanInferenceMs"]["mean"]),
    )


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, str]:
    performance = _performance_key(candidate)
    return (-performance[0], -performance[1], -performance[2], str(candidate["adapterId"]))


def _validate_comparison(report: Mapping[str, Any], *, schema_path: Path) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelComparisonError("Unable to load the comparison schema") from error
    Draft202012Validator.check_schema(schema)
    errors = list(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(report)
    )
    if errors:
        fields = sorted(
            ".".join(str(part) for part in error.absolute_path) or "<comparison>"
            for error in errors
        )
        raise ModelComparisonError(
            "Generated comparison violates its schema at: " + ", ".join(fields)
        )
