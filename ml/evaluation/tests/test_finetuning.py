from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.finetuning import (
    FineTuningReadinessError,
    build_finetuning_plan,
    load_finetuning_plan,
    load_label_ontology,
    prepare_finetuning_package,
    write_private_finetuning_package,
    write_private_json,
)
from nguven_evaluation.integrity import compute_text_content_hash


def manifest_record(record_id: str, label: str, split: str) -> dict[str, Any]:
    text = f"özel metin {record_id}"
    is_synthetic = label == "synthetic"
    return {
        "id": record_id,
        "source": f"source-{record_id}",
        "sourceGroup": f"source-group-{record_id}",
        "sourceUrl": f"https://example.invalid/{record_id}",
        "accessedAt": "2026-08-29T00:00:00Z",
        "license": "synthetic-test-only",
        "contentHash": compute_text_content_hash(text),
        "language": "tr",
        "contentType": "text/plain",
        "label": label,
        "labelSource": "synthetic-fixture",
        "generatorModel": f"generator-{record_id}" if is_synthetic else None,
        "generatorFamily": f"family-{record_id}" if is_synthetic else None,
        "transformation": "none",
        "intendedUse": "contract-testing-only",
        "split": split,
    }


def preprocessed_record(manifest: dict[str, Any]) -> dict[str, Any]:
    text = f"özel metin {manifest['id']}"
    content_hash = compute_text_content_hash(text)
    return {
        "id": manifest["id"],
        "text": text,
        "preprocessingVersion": "tr-text-v1",
        "inputContentHash": manifest["contentHash"],
        "outputContentHash": content_hash,
        "inputCharacters": len(text),
        "outputCharacters": len(text),
    }


def dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = [
        manifest_record(f"{split}-{label}", label, split)
        for split in ("train", "validation", "test")
        for label in ("human", "synthetic")
    ]
    return manifest, [preprocessed_record(record) for record in manifest]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def plan_for_files(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest, preprocessed = dataset()
    manifest_path = tmp_path / "manifest.jsonl"
    preprocessed_path = tmp_path / "preprocessed.jsonl"
    write_jsonl(manifest_path, manifest)
    write_jsonl(preprocessed_path, preprocessed)
    plan = build_finetuning_plan(
        plan_id="comparison-plan-v1",
        dataset_version="synthetic-dataset-v1",
        manifest_path=manifest_path,
        preprocessed_path=preprocessed_path,
        seeds=[17, 42],
        epochs=3,
        train_batch_size=8,
        evaluation_batch_size=16,
        learning_rate=0.00002,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_sequence_length=512,
        early_stopping_patience=2,
    )
    return plan, manifest_path, preprocessed_path


def test_build_plan_pins_dataset_and_both_candidates(tmp_path: Path) -> None:
    plan, manifest_path, preprocessed_path = plan_for_files(tmp_path)

    assert plan["dataset"]["manifestSha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert plan["dataset"]["preprocessedSha256"] == hashlib.sha256(
        preprocessed_path.read_bytes()
    ).hexdigest()
    assert {candidate["adapterId"] for candidate in plan["candidates"]} == {
        "berturk",
        "modernbert-tr",
    }
    assert plan["protocol"]["testSplitPolicy"] == "untouched-until-final-evaluation"


def test_load_plan_binds_exact_dataset_artifacts(tmp_path: Path) -> None:
    plan, manifest_path, preprocessed_path = plan_for_files(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_private_json(plan, plan_path)

    assert load_finetuning_plan(
        plan_path,
        manifest_path=manifest_path,
        preprocessed_path=preprocessed_path,
    ) == plan

    preprocessed_path.write_text("changed", encoding="utf-8")
    with pytest.raises(FineTuningReadinessError, match="preprocessing hash mismatch"):
        load_finetuning_plan(plan_path, preprocessed_path=preprocessed_path)


def test_prepare_package_isolates_test_and_omits_text_from_summary(tmp_path: Path) -> None:
    manifest, preprocessed = dataset()
    plan, _, _ = plan_for_files(tmp_path)

    package = prepare_finetuning_package(
        manifest,
        preprocessed,
        plan=plan,
        ontology=load_label_ontology(),
    )

    assert set(package.training_records) == {"train", "validation"}
    assert {record["split"] for record in package.training_records["train"]} == {"train"}
    assert {record["split"] for record in package.test_records} == {"test"}
    summary_text = json.dumps(package.summary, ensure_ascii=False)
    assert "özel metin" not in summary_text
    assert package.summary["splitCounts"]["test"] == {"human": 1, "synthetic": 1}


def test_write_package_uses_owner_only_permissions(tmp_path: Path) -> None:
    manifest, preprocessed = dataset()
    plan, _, _ = plan_for_files(tmp_path)
    package = prepare_finetuning_package(
        manifest,
        preprocessed,
        plan=plan,
        ontology=load_label_ontology(),
    )
    output = tmp_path / "readiness"

    write_private_finetuning_package(package, output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "training" / "train.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "evaluation" / "test.jsonl").stat().st_mode) == 0o600
    assert not (output / "training" / "test.jsonl").exists()
    with pytest.raises(FineTuningReadinessError, match="must not already exist"):
        write_private_finetuning_package(package, output)


def test_prepare_rejects_unknown_label(tmp_path: Path) -> None:
    manifest, preprocessed = dataset()
    manifest[0]["label"] = "unknown"
    plan, _, _ = plan_for_files(tmp_path)

    with pytest.raises(FineTuningReadinessError, match="Unsupported label"):
        prepare_finetuning_package(
            manifest,
            preprocessed,
            plan=plan,
            ontology=load_label_ontology(),
        )


def test_prepare_requires_every_label_in_every_split(tmp_path: Path) -> None:
    manifest, preprocessed = dataset()
    removed = manifest.pop()
    preprocessed = [record for record in preprocessed if record["id"] != removed["id"]]
    plan, _, _ = plan_for_files(tmp_path)

    with pytest.raises(FineTuningReadinessError, match="test:synthetic"):
        prepare_finetuning_package(
            manifest,
            preprocessed,
            plan=plan,
            ontology=load_label_ontology(),
        )


def test_prepare_rejects_coverage_and_hash_mismatches(tmp_path: Path) -> None:
    manifest, preprocessed = dataset()
    plan, _, _ = plan_for_files(tmp_path)

    with pytest.raises(FineTuningReadinessError, match="coverage mismatch"):
        prepare_finetuning_package(
            manifest,
            preprocessed[:-1],
            plan=plan,
            ontology=load_label_ontology(),
        )

    preprocessed[0]["inputContentHash"] = "sha256:" + "0" * 64
    with pytest.raises(FineTuningReadinessError, match="content hash mismatch"):
        prepare_finetuning_package(
            manifest,
            preprocessed,
            plan=plan,
            ontology=load_label_ontology(),
        )


def test_prepare_rechecks_cross_split_leakage(tmp_path: Path) -> None:
    manifest, preprocessed = dataset()
    manifest[0]["sourceGroup"] = manifest[-1]["sourceGroup"]
    plan, _, _ = plan_for_files(tmp_path)

    with pytest.raises(ValueError, match="spans splits"):
        prepare_finetuning_package(
            manifest,
            preprocessed,
            plan=plan,
            ontology=load_label_ontology(),
        )


def test_cli_creates_plan_and_materializes_isolated_package(tmp_path: Path) -> None:
    manifest, preprocessed = dataset()
    manifest_path = tmp_path / "manifest.jsonl"
    preprocessed_path = tmp_path / "preprocessed.jsonl"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "readiness"
    write_jsonl(manifest_path, manifest)
    write_jsonl(preprocessed_path, preprocessed)

    assert main(
        [
            "create-finetuning-plan",
            str(manifest_path),
            "--preprocessed",
            str(preprocessed_path),
            "--output",
            str(plan_path),
            "--plan-id",
            "comparison-plan-v1",
            "--dataset-version",
            "synthetic-dataset-v1",
            "--seed",
            "17",
        ]
    ) == 0
    assert main(
        [
            "prepare-finetuning-data",
            str(manifest_path),
            "--preprocessed",
            str(preprocessed_path),
            "--plan",
            str(plan_path),
            "--output",
            str(output),
        ]
    ) == 0

    assert (output / "training" / "train.jsonl").is_file()
    assert (output / "training" / "validation.jsonl").is_file()
    assert (output / "evaluation" / "test.jsonl").is_file()
