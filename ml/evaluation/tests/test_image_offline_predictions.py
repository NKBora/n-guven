from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from nguven_evaluation.image_model_adapters import ImagePrediction
from nguven_evaluation.image_offline_predictions import (
    ImageOfflinePredictionError,
    build_image_benchmark_predictions,
    load_image_benchmark_labels,
    load_preprocessed_image_variants,
)
from nguven_evaluation.image_preprocessing import ROBUSTNESS_TRANSFORMATIONS


class FakeAdapter:
    def predict(self, image: Image.Image) -> ImagePrediction:
        return ImagePrediction("human", 0.8)


def _private_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "preprocessed"
    root.mkdir()
    records = []
    for transformation in ROBUSTNESS_TRANSFORMATIONS:
        path = root / f"{transformation}.png"
        Image.new("RGB", (64, 64), "white").save(path)
        data = path.read_bytes()
        records.append(
            {
                "id": "fp_450-0000",
                "sourceSha256": "a" * 64,
                "preprocessingVersion": "image-preprocessing-v1",
                "transformation": transformation,
                "path": path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "sizeBytes": len(data),
                "width": 64,
                "height": 64,
                "format": "png",
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "id": "fp_450-0000",
                "label": "human",
                "sourceGroup": "fp_450",
                "generatorFamily": None,
                "contentCategory": "NEWS",
                "circulationStyle": "Captured",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root, labels


def test_builds_complete_stable_robustness_predictions(tmp_path: Path) -> None:
    root, label_path = _private_fixture(tmp_path)
    variants = load_preprocessed_image_variants(root)
    labels = load_image_benchmark_labels(label_path)
    ticks = iter(range(0, len(variants) * 2_000_000, 1_000_000))

    manifest, predictions = build_image_benchmark_predictions(
        variants,
        labels,
        adapter=FakeAdapter(),
        clock_ns=lambda: next(ticks),
    )

    assert len(manifest) == len(ROBUSTNESS_TRANSFORMATIONS)
    assert {item["transformation"] for item in manifest} == set(ROBUSTNESS_TRANSFORMATIONS)
    assert all(item["predictedLabel"] == "human" for item in predictions)
    assert all(item["inferenceMs"] == 1.0 for item in predictions)


def test_rejects_incomplete_robustness_suite(tmp_path: Path) -> None:
    root, _ = _private_fixture(tmp_path)
    lines = (root / "manifest.jsonl").read_text().splitlines()
    (root / "manifest.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ImageOfflinePredictionError, match="complete robustness suite"):
        load_preprocessed_image_variants(root)


def test_rejects_label_coverage_mismatch(tmp_path: Path) -> None:
    root, _ = _private_fixture(tmp_path)
    variants = load_preprocessed_image_variants(root)

    with pytest.raises(ImageOfflinePredictionError, match="different coverage"):
        build_image_benchmark_predictions(variants, {}, adapter=FakeAdapter())


def test_rejects_changed_variant_bytes(tmp_path: Path) -> None:
    root, _ = _private_fixture(tmp_path)
    (root / "canonical.png").write_bytes(b"changed")

    with pytest.raises(ImageOfflinePredictionError, match="size mismatch"):
        load_preprocessed_image_variants(root)
