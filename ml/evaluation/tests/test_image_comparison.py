/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nguven_evaluation.image_benchmark import load_image_benchmark_lock
from nguven_evaluation.image_comparison import (
    ImageModelComparisonError,
    compare_image_run_evidence,
)
from nguven_evaluation.image_model_adapters import load_image_candidate_registry
from nguven_evaluation.image_preprocessing import ROBUSTNESS_TRANSFORMATIONS


def _benchmark() -> dict:
    benchmark = deepcopy(load_image_benchmark_lock())
    benchmark["protocol"]["recordCount"] = 2
    benchmark["source"]["subsets"][0]["recordCount"] = 1
    benchmark["source"]["subsets"][1]["recordCount"] = 1
    benchmark["release"] = {
        "status": "source-locked",
        "evidenceStatus": "not-materialized",
        "resultsAllowed": False,
        "materializedArtifact": None,
    }
    return benchmark


def _package(
    adapter_id: str,
    revision: str,
    *,
    macro_f1: float,
    worst: float,
    false_positive_rate: float,
    latency: float,
) -> dict:
    run_id = f"{adapter_id}-v1"
    result = {
        "schemaVersion": "1.0",
        "run": {"id": run_id, "createdAt": "2026-08-30T12:00:00Z", "gitCommit": "abcdef1", "seed": 42},
        "dataset": {"version": "image-origin-robustness:v1", "recordCount": 12},
        "model": {"name": adapter_id, "version": revision},
        "metrics": {
            "accuracy": macro_f1,
            "macroPrecision": macro_f1,
            "macroRecall": macro_f1,
            "macroF1": macro_f1,
            "meanInferenceMs": latency / 2,
            "prAuc": macro_f1,
            "brierScore": 0.1,
            "ece": 0.03,
            "highConfidenceFalsePositiveRate": false_positive_rate,
            "p95InferenceMs": latency,
        },
        "slices": [
            {
                "dimension": "transformation",
                "key": transformation,
                "recordCount": 2,
                "macroF1": worst if transformation == "jpeg-q70" else macro_f1,
                "falsePositiveRate": false_positive_rate,
            }
            for transformation in ROBUSTNESS_TRANSFORMATIONS
        ],
        "artifacts": {"manifestSha256": "1" * 64, "predictionsSha256": ("2" if adapter_id.startswith("sig") else "3") * 64},
    }
    return {
        "run": {
            "schemaVersion": "image-benchmark-run-v1",
            "runId": run_id,
            "candidate": {"adapterId": adapter_id, "revision": revision},
            "benchmark": {"id": "image-origin-robustness", "version": "v1"},
            "counts": {"originals": 2, "predictions": 12},
            "artifacts": {
                "benchmarkSha256": "a" * 64,
                "candidateRegistrySha256": "b" * 64,
                "labelsSha256": "c" * 64,
                "preprocessedManifestSha256": "d" * 64,
            },
        },
        "result": result,
        "resultSha256": ("4" if adapter_id.startswith("sig") else "5") * 64,
    }


def _evidence() -> tuple[list[dict], dict[str, str]]:
    candidates = load_image_candidate_registry()
    revisions = {candidate.adapter_id: candidate.revision for candidate in candidates}
    return [
        _package("siglip2-aiornot", revisions["siglip2-aiornot"], macro_f1=0.88, worst=0.81, false_positive_rate=0.03, latency=40),
        _package("vit-cifake", revisions["vit-cifake"], macro_f1=0.90, worst=0.75, false_positive_rate=0.02, latency=30),
    ], revisions


def test_ranks_worst_transformation_before_overall_score() -> None:
    evidence, revisions = _evidence()

    report = compare_image_run_evidence(
        evidence,
        benchmark=_benchmark(),
        benchmark_sha256="a" * 64,
        expected_candidates=revisions,
        now=lambda: datetime(2026, 8, 30, tzinfo=timezone.utc),
    )

    assert [item["adapterId"] for item in report["candidates"]] == ["siglip2-aiornot", "vit-cifake"]
    assert report["selection"] == {"status": "experimental-leader", "leaders": ["siglip2-aiornot"]}
    assert all(item["acceptance"]["status"] == "passed" for item in report["candidates"])


def test_rejects_different_preprocessed_inputs() -> None:
    evidence, revisions = _evidence()
    evidence[1]["run"]["artifacts"]["preprocessedManifestSha256"] = "e" * 64

    with pytest.raises(ImageModelComparisonError, match="identical benchmark inputs"):
        compare_image_run_evidence(evidence, benchmark=_benchmark(), benchmark_sha256="a" * 64, expected_candidates=revisions)


def test_rejects_missing_transformation_slice() -> None:
    evidence, revisions = _evidence()
    evidence[0]["result"]["slices"].pop()

    with pytest.raises(ImageModelComparisonError, match="complete transformation"):
        compare_image_run_evidence(evidence, benchmark=_benchmark(), benchmark_sha256="a" * 64, expected_candidates=revisions)


def test_does_not_select_candidate_that_fails_acceptance() -> None:
    evidence, revisions = _evidence()
    for package in evidence:
        package["result"]["metrics"]["highConfidenceFalsePositiveRate"] = 0.2

    report = compare_image_run_evidence(evidence, benchmark=_benchmark(), benchmark_sha256="a" * 64, expected_candidates=revisions)

    assert report["selection"] == {"status": "no-qualified-candidate", "leaders": []}
