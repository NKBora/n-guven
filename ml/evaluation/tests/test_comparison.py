from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.comparison import (
    ModelComparisonError,
    compare_evaluation_results,
    load_evaluation_result,
    write_comparison_report,
)
from nguven_evaluation.finetuning import build_finetuning_plan, write_private_json


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def plan(tmp_path: Path, seeds: list[int] | None = None) -> dict[str, Any]:
    manifest = tmp_path / "manifest.jsonl"
    preprocessed = tmp_path / "preprocessed.jsonl"
    manifest.write_text("synthetic-manifest\n", encoding="utf-8")
    preprocessed.write_text("synthetic-preprocessed\n", encoding="utf-8")
    return build_finetuning_plan(
        plan_id="comparison-plan-v1",
        dataset_version="synthetic-dataset-v1",
        manifest_path=manifest,
        preprocessed_path=preprocessed,
        seeds=seeds or [17, 42],
        epochs=3,
        train_batch_size=8,
        evaluation_batch_size=16,
        learning_rate=0.00002,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_sequence_length=512,
        early_stopping_patience=2,
    )


def result(
    adapter_id: str,
    seed: int,
    *,
    macro_f1: float,
    accuracy: float,
    inference_ms: float,
    manifest_hash: str | None = None,
) -> dict[str, Any]:
    run_id = f"{adapter_id}-seed-{seed}"
    return {
        "schemaVersion": "1.0",
        "run": {
            "id": run_id,
            "createdAt": "2026-08-29T00:00:00Z",
            "gitCommit": "abcdef1234567",
            "seed": seed,
        },
        "dataset": {
            "version": "synthetic-dataset-v1",
            "recordCount": 20,
        },
        "model": {
            "name": adapter_id,
            "version": f"{adapter_id}-text-origin-v1",
        },
        "metrics": {
            "accuracy": accuracy,
            "macroPrecision": macro_f1,
            "macroRecall": macro_f1,
            "macroF1": macro_f1,
            "meanInferenceMs": inference_ms,
        },
        "artifacts": {
            "manifestSha256": manifest_hash or digest("shared-test-manifest"),
            "predictionsSha256": digest(run_id),
        },
    }


def candidate_results() -> list[dict[str, Any]]:
    return [
        result("berturk", 17, macro_f1=0.80, accuracy=0.82, inference_ms=5.0),
        result("berturk", 42, macro_f1=0.90, accuracy=0.88, inference_ms=6.0),
        result("modernbert-tr", 17, macro_f1=0.84, accuracy=0.83, inference_ms=8.0),
        result("modernbert-tr", 42, macro_f1=0.88, accuracy=0.89, inference_ms=8.5),
    ]


def test_comparison_aggregates_seed_variance_and_ranks_candidates(tmp_path: Path) -> None:
    report = compare_evaluation_results(
        candidate_results(),
        plan=plan(tmp_path),
        now=lambda: datetime(2026, 8, 29, tzinfo=timezone.utc),
    )

    assert report["createdAt"] == "2026-08-29T00:00:00Z"
    assert report["selection"] == {"status": "selected", "leaders": ["modernbert-tr"]}
    assert [candidate["adapterId"] for candidate in report["candidates"]] == [
        "modernbert-tr",
        "berturk",
    ]
    assert [candidate["rank"] for candidate in report["candidates"]] == [1, 2]
    berturk = report["candidates"][1]
    assert berturk["metrics"]["macroF1"]["mean"] == pytest.approx(0.85)
    assert berturk["metrics"]["macroF1"]["populationStdDev"] == pytest.approx(0.05)


def test_comparison_reports_performance_tie_without_forcing_selection(tmp_path: Path) -> None:
    results = [
        result(adapter, seed, macro_f1=0.8, accuracy=0.8, inference_ms=5.0)
        for adapter in ("berturk", "modernbert-tr")
        for seed in (17, 42)
    ]

    report = compare_evaluation_results(results, plan=plan(tmp_path))

    assert report["selection"] == {
        "status": "tied",
        "leaders": ["berturk", "modernbert-tr"],
    }
    assert [candidate["rank"] for candidate in report["candidates"]] == [1, 1]


def test_comparison_requires_identical_seed_coverage(tmp_path: Path) -> None:
    with pytest.raises(ModelComparisonError, match="Seed coverage mismatch"):
        compare_evaluation_results(candidate_results()[:-1], plan=plan(tmp_path))


def test_comparison_requires_identical_test_dataset(tmp_path: Path) -> None:
    results = candidate_results()
    results[-1]["artifacts"]["manifestSha256"] = digest("different-test-manifest")

    with pytest.raises(ModelComparisonError, match="same test dataset"):
        compare_evaluation_results(results, plan=plan(tmp_path))


def test_comparison_rejects_reused_predictions(tmp_path: Path) -> None:
    results = candidate_results()
    results[-1]["artifacts"]["predictionsSha256"] = results[0]["artifacts"][
        "predictionsSha256"
    ]

    with pytest.raises(ModelComparisonError, match="reuse a prediction artifact"):
        compare_evaluation_results(results, plan=plan(tmp_path))


def test_comparison_rejects_unplanned_candidate(tmp_path: Path) -> None:
    results = candidate_results()
    results[0]["model"]["name"] = "unreviewed-model"

    with pytest.raises(ModelComparisonError, match="Unexpected comparison candidate"):
        compare_evaluation_results(results, plan=plan(tmp_path))


def test_load_result_rejects_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text(json.dumps(candidate_results()[0]), encoding="utf-8")
    link = tmp_path / "result-link.json"
    link.symlink_to(target)

    with pytest.raises(ModelComparisonError, match="non-symbolic-link"):
        load_evaluation_result(link)


def test_write_comparison_report_is_owner_only(tmp_path: Path) -> None:
    report = compare_evaluation_results(candidate_results(), plan=plan(tmp_path))
    output = tmp_path / "reports" / "comparison.json"

    write_comparison_report(report, output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_comparison_cli_writes_valid_report(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write_private_json(plan(tmp_path), plan_path)
    result_paths: list[Path] = []
    for index, item in enumerate(candidate_results()):
        path = tmp_path / f"result-{index}.json"
        path.write_text(json.dumps(item), encoding="utf-8")
        result_paths.append(path)
    output = tmp_path / "comparison.json"
    arguments = ["compare-models", "--plan", str(plan_path), "--output", str(output)]
    for path in result_paths:
        arguments.extend(["--result", str(path)])

    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["selection"]["leaders"] == [
        "modernbert-tr"
    ]
