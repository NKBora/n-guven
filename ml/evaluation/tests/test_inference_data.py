from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nguven_evaluation.experiments import (
    DEFAULT_BERTURK_EXPERIMENT_PATH,
    DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH,
    load_experiment_spec,
)
from nguven_evaluation.inference_data import (
    InferenceDataError,
    prepare_inference_split,
    write_private_inference_split,
)
from nguven_evaluation.integrity import compute_text_content_hash


def manifest(record_id: str, *, split: str, label: str, text: str) -> dict:
    return {
        "id": record_id,
        "source": f"source-{label}",
        "sourceUrl": "https://example.com/data",
        "sourceGroup": f"group-{split}-{label}",
        "contentHash": compute_text_content_hash(text),
        "label": label,
        "labelSource": "reviewed-test-fixture",
        "language": "tr",
        "contentType": "text/plain",
        "split": split,
        "license": "MIT",
        "accessedAt": "2026-08-29T00:00:00Z",
        "transformation": "none",
        "intendedUse": "unit test",
    }


def preprocessed(record_id: str, text: str) -> dict:
    content_hash = compute_text_content_hash(text)
    return {
        "id": record_id,
        "inputCharacters": len(text),
        "inputContentHash": content_hash,
        "outputCharacters": len(text),
        "outputContentHash": content_hash,
        "preprocessingVersion": "tr-text-v1",
        "text": text,
    }


def fixtures() -> tuple[list[dict], list[dict]]:
    rows = [
        ("v-human", "validation", "human", "İnsan doğrulama metni."),
        ("v-synthetic", "validation", "synthetic", "Yapay doğrulama metni."),
        ("t-human", "test", "human", "İnsan test metni."),
        ("t-synthetic", "test", "synthetic", "Yapay test metni."),
    ]
    return (
        [manifest(record_id, split=split, label=label, text=text) for record_id, split, label, text in rows],
        [preprocessed(record_id, text) for record_id, _, _, text in rows],
    )


def completed_experiments() -> list[dict]:
    return [
        load_experiment_spec(DEFAULT_BERTURK_EXPERIMENT_PATH),
        load_experiment_spec(DEFAULT_MODERNBERT_TR_EXPERIMENT_PATH),
    ]


def test_prepare_validation_split_does_not_require_experiment_gate() -> None:
    manifests, texts = fixtures()
    package = prepare_inference_split(manifests, texts, split="validation")

    assert [record["id"] for record in package.manifest_records] == [
        "v-human",
        "v-synthetic",
    ]
    assert [record["id"] for record in package.preprocessed_records] == [
        "v-human",
        "v-synthetic",
    ]


def test_prepare_test_split_requires_two_completed_candidates() -> None:
    manifests, texts = fixtures()
    experiments = completed_experiments()
    experiments[1] = deepcopy(experiments[1])
    experiments[1]["execution"] = {
        "status": "ready",
        "allowed": True,
        "resultArtifact": None,
    }

    with pytest.raises(InferenceDataError, match="completed"):
        prepare_inference_split(
            manifests,
            texts,
            split="test",
            experiments=experiments,
        )


def test_prepare_test_split_after_candidate_completion() -> None:
    manifests, texts = fixtures()
    package = prepare_inference_split(
        manifests,
        texts,
        split="test",
        experiments=completed_experiments(),
    )

    assert len(package.manifest_records) == 2
    assert {record["label"] for record in package.manifest_records} == {
        "human",
        "synthetic",
    }


def test_prepare_split_rejects_coverage_mismatch() -> None:
    manifests, texts = fixtures()

    with pytest.raises(InferenceDataError, match="identical coverage"):
        prepare_inference_split(manifests, texts[:-1], split="validation")


def test_write_private_inference_split_is_atomic_and_hash_bound(tmp_path: Path) -> None:
    manifests, texts = fixtures()
    package = prepare_inference_split(manifests, texts, split="validation")
    output = tmp_path / "validation"

    provenance = write_private_inference_split(
        package,
        output,
        source_manifest_sha256="a" * 64,
        source_preprocessed_sha256="b" * 64,
    )

    assert provenance["recordCount"] == 2
    assert json.loads((output / "provenance.json").read_text()) == provenance
    assert (output / "manifest.jsonl").stat().st_mode & 0o077 == 0
    with pytest.raises(InferenceDataError, match="must not already exist"):
        write_private_inference_split(
            package,
            output,
            source_manifest_sha256="a" * 64,
            source_preprocessed_sha256="b" * 64,
        )
