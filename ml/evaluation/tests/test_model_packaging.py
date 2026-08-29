from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.finetuning import build_finetuning_plan, write_private_json
from nguven_evaluation.model_artifacts import verify_model_artifacts
from nguven_evaluation.model_packaging import (
    ModelPackagingError,
    package_finetuned_model,
    write_model_manifest,
)


def make_plan(tmp_path: Path) -> Path:
    manifest = tmp_path / "dataset.jsonl"
    preprocessed = tmp_path / "preprocessed.jsonl"
    manifest.write_text("synthetic-manifest\n", encoding="utf-8")
    preprocessed.write_text("synthetic-preprocessed\n", encoding="utf-8")
    plan = build_finetuning_plan(
        plan_id="comparison-plan-v1",
        dataset_version="synthetic-dataset-v1",
        manifest_path=manifest,
        preprocessed_path=preprocessed,
        seeds=[17, 42],
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
    path = tmp_path / "plan.json"
    write_private_json(plan, path)
    return path


def make_export(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"synthetic-safe-tensors")
    (root / "config.json").write_text(
        json.dumps({"id2label": {"0": "human", "1": "synthetic"}}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    return root


def package(tmp_path: Path, *, seed: int = 17, max_length: int = 512):
    return package_finetuned_model(
        make_export(tmp_path),
        model_id="berturk-text-origin-v1",
        adapter_id="berturk",
        framework_version="4.48.0",
        max_sequence_length=max_length,
        plan_path=make_plan(tmp_path),
        seed=seed,
        git_commit="abcdef1234567",
        intended_use="Offline Turkish human and synthetic text comparison.",
        limitations="Synthetic fixture; not suitable for a production decision.",
    )


def test_package_creates_verifiable_manifest_with_finetuning_provenance(
    tmp_path: Path,
) -> None:
    artifact_root = make_export(tmp_path)
    plan_path = make_plan(tmp_path)

    manifest = package_finetuned_model(
        artifact_root,
        model_id="berturk-text-origin-v1",
        adapter_id="berturk",
        framework_version="4.48.0",
        max_sequence_length=512,
        plan_path=plan_path,
        seed=17,
        git_commit="abcdef1234567",
        intended_use="Offline Turkish human and synthetic text comparison.",
        limitations="Synthetic fixture; not suitable for a production decision.",
    )
    report = verify_model_artifacts(manifest, artifact_root=artifact_root)

    assert report.artifact_count == 4
    assert manifest["fineTuning"]["planId"] == "comparison-plan-v1"
    assert manifest["fineTuning"]["seed"] == 17
    assert manifest["labels"] == {"0": "human", "1": "synthetic"}
    assert {artifact["path"] for artifact in manifest["artifacts"]} == {
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
    }


def test_write_model_manifest_is_owner_only(tmp_path: Path) -> None:
    manifest = package(tmp_path)
    output = tmp_path / "manifests" / "model.json"

    write_model_manifest(manifest, output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "optimizer.pt", "model.py"])
def test_package_rejects_unsafe_or_unreviewed_files(tmp_path: Path, filename: str) -> None:
    artifact_root = make_export(tmp_path)
    (artifact_root / filename).write_bytes(b"unsafe")

    with pytest.raises(ModelPackagingError, match="Unsupported file"):
        package_finetuned_model(
            artifact_root,
            model_id="berturk-text-origin-v1",
            adapter_id="berturk",
            framework_version="4.48.0",
            max_sequence_length=512,
            plan_path=make_plan(tmp_path),
            seed=17,
            git_commit="abcdef1234567",
            intended_use="Offline Turkish human and synthetic text comparison.",
            limitations="Synthetic fixture; not suitable for a production decision.",
        )


def test_package_rejects_symbolic_links(tmp_path: Path) -> None:
    artifact_root = make_export(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (artifact_root / "tokenizer.json").symlink_to(outside)

    with pytest.raises(ModelPackagingError, match="symbolic link"):
        package_finetuned_model(
            artifact_root,
            model_id="berturk-text-origin-v1",
            adapter_id="berturk",
            framework_version="4.48.0",
            max_sequence_length=512,
            plan_path=make_plan(tmp_path),
            seed=17,
            git_commit="abcdef1234567",
            intended_use="Offline Turkish human and synthetic text comparison.",
            limitations="Synthetic fixture; not suitable for a production decision.",
        )


def test_package_rejects_incompatible_labels(tmp_path: Path) -> None:
    artifact_root = make_export(tmp_path)
    (artifact_root / "config.json").write_text(
        json.dumps({"id2label": {"0": "LABEL_0", "1": "LABEL_1"}}),
        encoding="utf-8",
    )

    with pytest.raises(ModelPackagingError, match="label mapping"):
        package_finetuned_model(
            artifact_root,
            model_id="berturk-text-origin-v1",
            adapter_id="berturk",
            framework_version="4.48.0",
            max_sequence_length=512,
            plan_path=make_plan(tmp_path),
            seed=17,
            git_commit="abcdef1234567",
            intended_use="Offline Turkish human and synthetic text comparison.",
            limitations="Synthetic fixture; not suitable for a production decision.",
        )


def test_package_binds_seed_and_sequence_length_to_plan(tmp_path: Path) -> None:
    with pytest.raises(ModelPackagingError, match="seed"):
        package(tmp_path, seed=99)

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ModelPackagingError, match="sequence length"):
        package(other, max_length=256)


def test_package_cli_writes_manifest_outside_export(tmp_path: Path) -> None:
    artifact_root = make_export(tmp_path)
    plan_path = make_plan(tmp_path)
    output = tmp_path / "manifests" / "berturk.json"

    assert main(
        [
            "package-finetuned-model",
            str(artifact_root),
            "--output",
            str(output),
            "--model-id",
            "berturk-text-origin-v1",
            "--adapter-id",
            "berturk",
            "--framework-version",
            "4.48.0",
            "--max-sequence-length",
            "512",
            "--plan",
            str(plan_path),
            "--seed",
            "17",
            "--git-commit",
            "abcdef1234567",
            "--intended-use",
            "Offline Turkish human and synthetic text comparison.",
            "--limitations",
            "Synthetic fixture; not suitable for a production decision.",
        ]
    ) == 0
    assert output.is_file()
