/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nguven_evaluation.image_benchmark import load_image_benchmark_lock
from nguven_evaluation.image_benchmark_runner import (
    ImageBenchmarkRunError,
    write_image_benchmark_run,
)
from nguven_evaluation.image_model_adapters import load_image_candidate_registry
from nguven_evaluation.image_preprocessing import ROBUSTNESS_TRANSFORMATIONS


def _records() -> tuple[list[dict], list[dict]]:
    manifest = []
    predictions = []
    for source_id, label, score in (
        ("human-1", "human", 0.9),
        ("synthetic-1", "synthetic", 0.8),
    ):
        for transformation in ROBUSTNESS_TRANSFORMATIONS:
            record_id = f"{source_id}--{transformation}"
            manifest.append(
                {
                    "id": record_id,
                    "label": label,
                    "sourceGroup": "human-source" if label == "human" else "ai-source",
                    "generatorFamily": None if label == "human" else "generator-a",
                    "transformation": transformation,
                }
            )
            predictions.append(
                {
                    "id": record_id,
                    "predictedLabel": label,
                    "score": score,
                    "inferenceMs": 2.0,
                }
            )
    return manifest, predictions


def _benchmark() -> dict:
    benchmark = deepcopy(load_image_benchmark_lock())
    benchmark["protocol"]["recordCount"] = 2
    benchmark["source"]["subsets"][0]["recordCount"] = 1
    benchmark["source"]["subsets"][1]["recordCount"] = 1
    return benchmark


def test_writes_hash_bound_owner_only_run_atomically(tmp_path: Path) -> None:
    manifest, predictions = _records()
    candidate = load_image_candidate_registry()[0]
    output = tmp_path / "run"

    evidence = write_image_benchmark_run(
        manifest,
        predictions,
        benchmark=_benchmark(),
        candidate=candidate,
        output_root=output,
        run_id="siglip2-test-v1",
        git_commit="abcdef1",
        seed=42,
        device="cpu",
        benchmark_sha256="a" * 64,
        candidate_registry_sha256="b" * 64,
        labels_sha256="c" * 64,
        preprocessed_manifest_sha256="d" * 64,
        now=lambda: datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc),
    )

    assert set(item.name for item in output.iterdir()) == {
        "manifest.jsonl",
        "predictions.jsonl",
        "result.json",
        "run.json",
    }
    run = json.loads((output / "run.json").read_text())
    assert run["benchmark"]["claimStatus"] == "experimental-unreviewed"
    assert run["counts"] == {"originals": 2, "predictions": 12}
    assert run["artifacts"]["predictionsSha256"] == hashlib.sha256(
        (output / "predictions.jsonl").read_bytes()
    ).hexdigest()
    assert evidence["result"]["metrics"]["macroF1"] == 1.0
    assert (output.stat().st_mode & 0o777) == 0o700
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in output.iterdir())


def test_rejects_existing_output_without_overwriting(tmp_path: Path) -> None:
    manifest, predictions = _records()
    candidate = load_image_candidate_registry()[0]
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(ImageBenchmarkRunError, match="must not already exist"):
        write_image_benchmark_run(
            manifest,
            predictions,
            benchmark=_benchmark(),
            candidate=candidate,
            output_root=output,
            run_id="siglip2-test-v1",
            git_commit="abcdef1",
            seed=42,
            device="cpu",
            benchmark_sha256="a" * 64,
            candidate_registry_sha256="b" * 64,
            labels_sha256="c" * 64,
            preprocessed_manifest_sha256="d" * 64,
        )


def test_rejects_non_hexadecimal_git_commit(tmp_path: Path) -> None:
    manifest, predictions = _records()
    candidate = load_image_candidate_registry()[0]

    with pytest.raises(ImageBenchmarkRunError, match="Git commit"):
        write_image_benchmark_run(
            manifest,
            predictions,
            benchmark=_benchmark(),
            candidate=candidate,
            output_root=tmp_path / "run",
            run_id="siglip2-test-v1",
            git_commit="not-a-sha",
            seed=42,
            device="cpu",
            benchmark_sha256="a" * 64,
            candidate_registry_sha256="b" * 64,
            labels_sha256="c" * 64,
            preprocessed_manifest_sha256="d" * 64,
        )
