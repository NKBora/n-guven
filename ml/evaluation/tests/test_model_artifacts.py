from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nguven_evaluation.model_adapters import BERTURK
from nguven_evaluation.model_artifacts import (
    ModelArtifactError,
    load_model_artifact_manifest,
    verify_model_artifacts,
)


def valid_manifest(artifact: bytes = b"synthetic-model") -> dict[str, object]:
    return {
        "schemaVersion": "1.0",
        "modelId": "synthetic-berturk-v1",
        "adapterId": "berturk",
        "task": "text-classification",
        "languages": ["tr"],
        "preprocessingVersion": "tr-text-v1",
        "upstream": {
            "provider": "synthetic test fixture",
            "repository": "dbmdz/bert-base-turkish-cased",
            "revision": BERTURK.revision,
            "tokenizerRepository": "dbmdz/bert-base-turkish-cased",
            "tokenizerRevision": BERTURK.revision,
            "weightsLicense": {
                "spdxId": "MIT",
                "url": "https://spdx.org/licenses/MIT.html",
            },
            "codeLicense": {
                "spdxId": "Apache-2.0",
                "url": "https://spdx.org/licenses/Apache-2.0.html",
            },
        },
        "fineTuning": {
            "planId": "synthetic-comparison-v1",
            "planSha256": "1" * 64,
            "datasetManifestSha256": "2" * 64,
            "preprocessedSha256": "3" * 64,
            "seed": 42,
            "gitCommit": "abcdef1234567",
        },
        "runtime": {
            "framework": "transformers",
            "frameworkVersion": "4.48.0",
            "pythonVersion": "3.12.8",
            "artifactFormat": "safetensors",
            "maxSequenceLength": 512,
        },
        "labels": {"0": "human", "1": "synthetic"},
        "artifacts": [
            {
                "path": "model.safetensors",
                "role": "weights",
                "mediaType": "application/vnd.safetensors",
                "sha256": hashlib.sha256(artifact).hexdigest(),
                "sizeBytes": len(artifact),
            }
        ],
        "intendedUse": "Synthetic contract fixture for offline classification tests only.",
        "limitations": "This fixture is not a trained model and must never be used as evidence.",
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_and_verify_model_artifacts(tmp_path: Path) -> None:
    artifact = b"synthetic-model"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "model.safetensors").write_bytes(artifact)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, valid_manifest(artifact))

    manifest = load_model_artifact_manifest(manifest_path)
    report = verify_model_artifacts(manifest, artifact_root=artifact_root)

    assert report.model_id == "synthetic-berturk-v1"
    assert report.artifact_count == 1
    assert report.total_bytes == len(artifact)


def test_manifest_rejects_mutable_upstream_revision(tmp_path: Path) -> None:
    manifest = valid_manifest()
    manifest["upstream"]["revision"] = "main"  # type: ignore[index]
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)

    with pytest.raises(ModelArtifactError, match="upstream.revision"):
        load_model_artifact_manifest(manifest_path)


def test_manifest_rejects_parent_directory_artifact_path(tmp_path: Path) -> None:
    manifest = valid_manifest()
    manifest["artifacts"][0]["path"] = "../model.safetensors"  # type: ignore[index]
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)

    with pytest.raises(ModelArtifactError, match="artifacts.0.path"):
        load_model_artifact_manifest(manifest_path)


def test_verify_rejects_hash_mismatch_without_exposing_bytes(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "model.safetensors").write_bytes(b"tampered-model")
    manifest = valid_manifest(b"expected-model")
    manifest["artifacts"][0]["sizeBytes"] = len(b"tampered-model")  # type: ignore[index]

    with pytest.raises(ModelArtifactError, match="hash mismatch") as error:
        verify_model_artifacts(manifest, artifact_root=artifact_root)

    assert "tampered-model" not in str(error.value)


def test_verify_rejects_symbolic_link_artifact(tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"synthetic-model")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "model.safetensors").symlink_to(outside)

    with pytest.raises(ModelArtifactError, match="symbolic-link path component"):
        verify_model_artifacts(valid_manifest(), artifact_root=artifact_root)


def test_verify_rejects_symbolic_link_directory_component(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = b"synthetic-model"
    (outside / "model.safetensors").write_bytes(artifact)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "nested").symlink_to(outside, target_is_directory=True)
    manifest = valid_manifest(artifact)
    manifest["artifacts"][0]["path"] = "nested/model.safetensors"  # type: ignore[index]

    with pytest.raises(ModelArtifactError, match="symbolic-link path component"):
        verify_model_artifacts(manifest, artifact_root=artifact_root)
