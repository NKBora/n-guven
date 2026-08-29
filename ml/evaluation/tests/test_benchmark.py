from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nguven_evaluation.benchmark import (
    BenchmarkContractError,
    benchmark_evidence_allowed,
    load_benchmark_lock,
)


def write_lock(tmp_path: Path, lock: dict) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_repository_benchmark_locks_sources_without_claiming_results() -> None:
    lock = load_benchmark_lock()

    assert lock["benchmarkId"] == "text-origin-tr"
    assert lock["sampling"]["targetRecordCount"] == 12000
    assert {source["label"] for source in lock["sources"]} == {"human", "synthetic"}
    assert len(
        {
            source["generatorFamily"]
            for source in lock["sources"]
            if source["label"] == "synthetic"
        }
    ) >= 2
    assert benchmark_evidence_allowed(lock) is False


def test_synthetic_sources_require_generator_provenance(tmp_path: Path) -> None:
    lock = load_benchmark_lock()
    invalid = deepcopy(lock)
    invalid["sources"][1]["generatorModel"] = None

    with pytest.raises(BenchmarkContractError, match="model and family provenance"):
        load_benchmark_lock(write_lock(tmp_path, invalid))


def test_unreviewed_release_cannot_enable_results(tmp_path: Path) -> None:
    lock = load_benchmark_lock()
    invalid = deepcopy(lock)
    invalid["release"]["resultsAllowed"] = True

    with pytest.raises(BenchmarkContractError, match="before reviewed materialization"):
        load_benchmark_lock(write_lock(tmp_path, invalid))


def test_source_revisions_must_be_unique(tmp_path: Path) -> None:
    lock = load_benchmark_lock()
    invalid = deepcopy(lock)
    invalid["sources"][2]["repository"] = invalid["sources"][1]["repository"]
    invalid["sources"][2]["revision"] = invalid["sources"][1]["revision"]

    with pytest.raises(BenchmarkContractError, match="source revisions must be unique"):
        load_benchmark_lock(write_lock(tmp_path, invalid))
