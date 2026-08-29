from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from nguven_evaluation.benchmark import load_benchmark_lock
from nguven_evaluation.experiments import (
    DEFAULT_BERTURK_EXPERIMENT_PATH,
    DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH,
    ExperimentContractError,
    experiment_execution_allowed,
    load_experiment_spec,
    validate_comparable_experiments,
    validate_experiment_inputs,
)
from nguven_evaluation.model_adapters import BERTURK, MODERNBERT_TR


def shared_plan() -> dict:
    return {
        "dataset": {
            "version": "text-origin-tr-v1",
            "manifestSha256": "92f0ce2a53a41b1b8e793582b851817bb6d182c61e026a3adcb2cb755badbb2a",
            "preprocessedSha256": "bf46915e2f67bd70fe583f7ceeeae8c296c2337a6b7e5a0a87e849b1e07d806f",
        },
        "candidates": [
            {"adapterId": item.adapter_id, "repository": item.repository, "revision": item.revision}
            for item in (BERTURK, MODERNBERT_TR)
        ],
        "protocol": {"seeds": [17, 42, 71], "maxSequenceLength": 128},
    }


def test_berturk_experiment_records_completed_validation_runs() -> None:
    specification = load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH)
    benchmark = load_benchmark_lock()

    validate_experiment_inputs(specification, plan=shared_plan(), benchmark=benchmark)
    assert specification["adapterId"] == "berturk"
    assert specification["protocol"]["seeds"] == [17, 42, 71]
    assert specification["execution"]["status"] == "completed"
    assert specification["execution"]["resultArtifact"]["runCount"] == 3
    assert experiment_execution_allowed(specification, benchmark) is False


def test_berturk_evidence_matches_completed_experiment_hash() -> None:
    specification = load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH)
    evidence_path = DEFAULT_BERTURK_EXPERIMENT_PATH.parent / "evidence" / "berturk-v1.json"
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == specification["execution"][
        "resultArtifact"
    ]["sha256"]
    assert {item["seed"] for item in evidence["selection"]} == {17, 42, 71}
    assert evidence["testSplitAccessed"] is False


def test_berturk_experiment_rejects_upstream_drift(tmp_path: Path) -> None:
    specification = load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH)
    invalid = deepcopy(specification)
    invalid["upstream"]["revision"] = "0" * 40
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="upstream identity mismatch"):
        load_experiment_spec(path)


def test_execution_requires_reviewed_materialized_benchmark() -> None:
    specification = load_experiment_spec(DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH)
    specification = deepcopy(specification)
    specification["execution"] = {
        "status": "ready",
        "allowed": True,
        "resultArtifact": None,
    }
    benchmark = load_benchmark_lock()
    benchmark = deepcopy(benchmark)
    benchmark["release"]["evidenceStatus"] = "not-materialized"

    with pytest.raises(ExperimentContractError, match="not approved for result evidence"):
        validate_experiment_inputs(
            specification,
            plan=shared_plan(),
            benchmark=benchmark,
            require_execution_ready=True,
        )


def test_experiment_rejects_seed_drift() -> None:
    specification = load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH)
    benchmark = load_benchmark_lock()
    plan = shared_plan()
    plan["protocol"]["seeds"] = [17, 42, 99]

    with pytest.raises(ExperimentContractError, match="seed order differs"):
        validate_experiment_inputs(specification, plan=plan, benchmark=benchmark)


def test_modernbert_experiment_is_pinned_to_reviewed_identity() -> None:
    specification = load_experiment_spec(DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH)

    assert specification["adapterId"] == "modernbert-tr"
    assert specification["upstream"]["revision"] == MODERNBERT_TR.revision
    assert specification["protocol"]["seeds"] == [17, 42, 71]
    assert specification["execution"]["status"] == "completed"
    assert specification["execution"]["resultArtifact"]["runCount"] == 3


def test_modernbert_evidence_matches_completed_experiment_hash() -> None:
    specification = load_experiment_spec(DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH)
    evidence_path = (
        DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH.parent
        / "evidence"
        / "modernbert-tr-v1.json"
    )
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == specification["execution"][
        "resultArtifact"
    ]["sha256"]
    assert {item["seed"] for item in evidence["selection"]} == {17, 42, 71}
    assert evidence["testSplitAccessed"] is False


def test_candidate_experiments_are_strictly_comparable() -> None:
    validate_comparable_experiments(
        [
            load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH),
            load_experiment_spec(DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH),
        ]
    )


def test_candidate_comparison_rejects_protocol_drift() -> None:
    berturk = load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH)
    modernbert = load_experiment_spec(DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH)
    modernbert = deepcopy(modernbert)
    modernbert["protocol"]["seeds"] = [17, 42, 99]

    with pytest.raises(ExperimentContractError, match="different training protocol"):
        validate_comparable_experiments([berturk, modernbert])


def test_candidate_comparison_rejects_acceptance_drift() -> None:
    berturk = load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH)
    modernbert = load_experiment_spec(DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH)
    modernbert = deepcopy(modernbert)
    modernbert["acceptance"]["minimumMacroF1"] = 0.7

    with pytest.raises(ExperimentContractError, match="different acceptance thresholds"):
        validate_comparable_experiments([berturk, modernbert])
