from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from nguven_evaluation.finetuning import build_finetuning_plan, write_private_json
from nguven_evaluation.training import (
    TrainingExecutionError,
    TrainingStageRequest,
    TrainingStageResult,
    build_training_requests,
    execute_candidate_training,
    load_training_splits,
)


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def training_record(split: str, label: str) -> dict[str, Any]:
    text = f"{split} için {label} örneği"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "id": f"{split}-{label}",
        "text": text,
        "label": label,
        "labelId": 0 if label == "human" else 1,
        "split": split,
        "preprocessingVersion": "tr-text-v1",
        "inputContentHash": f"sha256:{digest}",
        "outputContentHash": f"sha256:{digest}",
    }


def setup_execution(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    preprocessed = tmp_path / "preprocessed.jsonl"
    manifest.write_text("manifest\n", encoding="utf-8")
    preprocessed.write_text("preprocessed\n", encoding="utf-8")
    plan = build_finetuning_plan(
        plan_id="text-origin-plan-v1",
        dataset_version="text-origin-tr-v1",
        manifest_path=manifest,
        preprocessed_path=preprocessed,
        seeds=[17, 42, 71],
        epochs=3,
        train_batch_size=8,
        gradient_accumulation_steps=1,
        evaluation_batch_size=16,
        learning_rate=0.00002,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_sequence_length=512,
        early_stopping_patience=2,
    )
    plan_path = tmp_path / "plan.json"
    write_private_json(plan, plan_path)
    training_root = tmp_path / "training"
    training_root.mkdir()
    for split in ("train", "validation"):
        write_jsonl(
            training_root / f"{split}.jsonl",
            [training_record(split, "human"), training_record(split, "synthetic")],
        )
    return plan_path, training_root


class FakeBackend:
    def train(
        self,
        request: TrainingStageRequest,
        *,
        candidate: Any,
        protocol: Mapping[str, Any],
        train_records: Sequence[Mapping[str, Any]],
        validation_records: Sequence[Mapping[str, Any]],
    ) -> TrainingStageResult:
        assert {record["split"] for record in train_records} == {"train"}
        assert {record["split"] for record in validation_records} == {"validation"}
        export = request.output_directory / "export"
        export.mkdir()
        (export / "model.safetensors").write_bytes(b"safe-fixture")
        score = 0.7 if request.stage == "linear-probe" else 0.8
        return TrainingStageResult(
            stage=request.stage,
            seed=request.seed,
            validation_macro_f1=score,
            best_epoch=2.0,
            training_seconds=1.25,
            artifact_directory=str(export.relative_to(request.output_directory.parents[1])),
        )


def provenance() -> dict[str, Any]:
    return {
        "experiment": {"experimentId": "berturk-v1"},
        "experiment_sha256": "1" * 64,
        "benchmark": {
            "benchmarkId": "text-origin-tr",
            "version": "v1",
            "release": {
                "materializedArtifact": {
                    "manifestSha256": "2" * 64,
                    "preprocessedSha256": "3" * 64,
                }
            },
        },
        "environment_lock": {
            "environmentId": "fixture-v1",
            "execution": {"device": "cpu"},
        },
        "environment_sha256": "4" * 64,
    }


def test_training_requests_are_seeded_and_stage_ordered(tmp_path: Path) -> None:
    plan_path, _ = setup_execution(tmp_path)
    plan = json.loads(plan_path.read_text())
    requests = build_training_requests(plan, adapter_id="berturk", output_root=tmp_path)

    assert [(request.seed, request.stage) for request in requests] == [
        (17, "linear-probe"),
        (17, "fine-tune"),
        (42, "linear-probe"),
        (42, "fine-tune"),
        (71, "linear-probe"),
        (71, "fine-tune"),
    ]


def test_training_loader_rejects_test_material(tmp_path: Path) -> None:
    _, training_root = setup_execution(tmp_path)
    (training_root / "test.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(TrainingExecutionError, match="forbidden test material"):
        load_training_splits(training_root)


def test_candidate_execution_is_atomic_and_selects_validation_winner(tmp_path: Path) -> None:
    plan_path, training_root = setup_execution(tmp_path)
    output = tmp_path / "berturk-run"

    execution = execute_candidate_training(
        plan_path=plan_path,
        training_root=training_root,
        adapter_id="berturk",
        run_id="berturk-v1",
        git_commit="abcdef1234567",
        output_root=output,
        backend=FakeBackend(),
        **provenance(),
    )

    assert len(execution["stages"]) == 6
    assert {item["selectedStage"] for item in execution["selection"]} == {"fine-tune"}
    assert execution["experiment"]["experimentId"] == "berturk-v1"
    assert execution["benchmark"]["benchmarkId"] == "text-origin-tr"
    assert execution["environment"]["environmentId"] == "fixture-v1"
    assert (output / "execution.json").is_file()
    assert not any("test" in path.name for path in output.rglob("*"))


def test_candidate_execution_rejects_overwrite(tmp_path: Path) -> None:
    plan_path, training_root = setup_execution(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(TrainingExecutionError, match="must not already exist"):
        execute_candidate_training(
            plan_path=plan_path,
            training_root=training_root,
            adapter_id="berturk",
            run_id="berturk-v1",
            git_commit="abcdef1",
            output_root=output,
            backend=FakeBackend(),
        )
