/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from nguven_evaluation.image_benchmark import load_image_benchmark_lock
from nguven_evaluation.image_benchmark_materialization import (
    ImageBenchmarkMaterializationError,
    fetch_locked_image_sources,
    materialize_image_benchmark,
)
from nguven_evaluation.image_dataset_inputs import load_image_dataset_inputs


def _png(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


def _mpo() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(
        output,
        format="MPO",
        save_all=True,
        append_images=[Image.new("RGB", (64, 64), "black")],
    )
    return output.getvalue()


def _small_lock(cache: Path) -> tuple[dict, list[Path]]:
    lock = deepcopy(load_image_benchmark_lock())
    files = [
        ("fp_450/real.parquet", b"real-source"),
        ("syncred_600/synthetic.parquet", b"synthetic-source"),
    ]
    paths: list[Path] = []
    artifacts = []
    for relative, content in files:
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths.append(path)
        artifacts.append(
            {
                "path": relative,
                "sizeBytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    lock["source"]["artifacts"] = artifacts
    lock["source"]["subsets"][0]["recordCount"] = 1
    lock["source"]["subsets"][1]["recordCount"] = 1
    lock["protocol"]["recordCount"] = 2
    return lock, paths


def test_materializes_private_images_labels_and_hashes(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock, paths = _small_lock(cache)

    def rows(path: Path):
        subset = path.parent.name
        yield {
            "subset": subset,
            "image": {"bytes": _png("white" if subset == "fp_450" else "black")},
            "prefix": "NEWS",
            "style": "Captured",
        }

    output = tmp_path / "private"
    summary = materialize_image_benchmark(lock, paths, output, row_reader=rows)

    assert summary["recordCount"] == 2
    assert summary["labelCounts"] == {"human": 1, "synthetic": 1}
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    assert {item["label"] for item in labels} == {"human", "synthetic"}
    verified = load_image_dataset_inputs(
        output / "inputs.jsonl",
        image_root=output / "images",
    )
    assert len(verified) == 2


def test_local_fetch_rejects_changed_source_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock, _ = _small_lock(cache)
    (cache / "fp_450" / "real.parquet").write_bytes(b"wrong-bytes")

    with pytest.raises(ImageBenchmarkMaterializationError, match="mismatch"):
        fetch_locked_image_sources(lock, cache)


def test_materialization_rejects_subset_count_drift(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock, paths = _small_lock(cache)

    def rows(path: Path):
        if path.parent.name == "fp_450":
            return iter(())
        return iter(
            [
                {
                    "subset": "syncred_600",
                    "image": {"bytes": _png("black")},
                    "prefix": "NEWS",
                    "style": "Captured",
                }
            ]
        )

    with pytest.raises(ImageBenchmarkMaterializationError, match="subset counts"):
        materialize_image_benchmark(lock, paths, tmp_path / "private", row_reader=rows)


def test_materialization_rejects_external_image_paths(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock, paths = _small_lock(cache)

    def rows(path: Path):
        yield {
            "subset": path.parent.name,
            "image": {"path": "/untrusted/image.png", "bytes": None},
            "prefix": "NEWS",
            "style": "Captured",
        }

    with pytest.raises(ImageBenchmarkMaterializationError, match="embedded bytes"):
        materialize_image_benchmark(lock, paths, tmp_path / "private", row_reader=rows)


def test_materializes_reviewed_two_view_mpo_without_changing_source_bytes(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    lock, paths = _small_lock(cache)
    mpo = _mpo()

    def rows(path: Path):
        yield {
            "subset": path.parent.name,
            "image": {"bytes": mpo if path.parent.name == "fp_450" else _png("black")},
            "prefix": "NEWS",
            "style": "Captured",
        }

    output = tmp_path / "private"
    materialize_image_benchmark(lock, paths, output, row_reader=rows)

    stored = output / "images" / "fp_450-0000" / "original.mpo"
    assert stored.read_bytes() == mpo
