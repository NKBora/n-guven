from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nguven_evaluation.image_benchmark import (
    ImageBenchmarkContractError,
    image_benchmark_evidence_allowed,
    load_image_benchmark_lock,
)


def _write_lock(tmp_path: Path, lock: dict) -> Path:
    path = tmp_path / "image-benchmark.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_repository_lock_is_external_frozen_and_evidence_gated() -> None:
    lock = load_image_benchmark_lock()

    assert lock["source"]["repository"] == "thu-coai/Syncred-Bench"
    assert lock["source"]["revision"] == "2747c4b2c901b24865c16c4f8c73a220675f9820"
    assert lock["source"]["license"] == "Apache-2.0"
    assert lock["protocol"]["frozenTest"] is True
    assert lock["protocol"]["recordCount"] == 1050
    assert "competitions/aiornot" in lock["source"]["excludedTrainingRepositories"]
    assert "CIFAKE" in lock["source"]["excludedTrainingRepositories"]
    assert image_benchmark_evidence_allowed(lock) is False


def test_lock_rejects_candidate_registry_drift(tmp_path: Path) -> None:
    registry = tmp_path / "candidates.json"
    registry.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ImageBenchmarkContractError, match="registry hash"):
        load_image_benchmark_lock(candidate_registry_path=registry)


def test_lock_rejects_missing_robustness_transformation(tmp_path: Path) -> None:
    lock = deepcopy(load_image_benchmark_lock())
    lock["protocol"]["transformations"][-1] = "canonical"

    with pytest.raises(ImageBenchmarkContractError, match="schema"):
        load_image_benchmark_lock(_write_lock(tmp_path, lock))


def test_lock_rejects_unreviewed_result_enablement(tmp_path: Path) -> None:
    lock = deepcopy(load_image_benchmark_lock())
    lock["release"]["resultsAllowed"] = True

    with pytest.raises(ImageBenchmarkContractError, match="before reviewed materialization"):
        load_image_benchmark_lock(_write_lock(tmp_path, lock))


def test_lock_rejects_incorrect_subset_count(tmp_path: Path) -> None:
    lock = deepcopy(load_image_benchmark_lock())
    lock["source"]["subsets"][0]["recordCount"] = 449

    with pytest.raises(ImageBenchmarkContractError, match="subset counts"):
        load_image_benchmark_lock(_write_lock(tmp_path, lock))
