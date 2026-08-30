/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import hashlib
import io
from copy import deepcopy
from pathlib import Path

from PIL import Image

from nguven_evaluation.image_benchmark import load_image_benchmark_lock
from nguven_evaluation.image_benchmark_materialization import materialize_image_benchmark
from nguven_evaluation.image_benchmark_runner import write_image_benchmark_run
from nguven_evaluation.image_comparison import (
    compare_image_run_evidence,
    load_image_benchmark_run,
)
from nguven_evaluation.image_dataset_inputs import load_image_dataset_inputs
from nguven_evaluation.image_model_adapters import (
    CandidateImageModelAdapter,
    ImagePrediction,
    load_image_candidate_registry,
)
from nguven_evaluation.image_offline_predictions import (
    build_image_benchmark_predictions,
    load_image_benchmark_labels,
    load_preprocessed_image_variants,
)
from nguven_evaluation.image_preprocessing import write_preprocessed_image_dataset


class PixelBackend:
    def predict(self, image: Image.Image) -> ImagePrediction:
        pixel = image.getpixel((image.width // 2, image.height // 2))
        label = "human" if sum(pixel) > 384 else "synthetic"
        return ImagePrediction(label, 0.9)


def _png(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


def _small_source_lock(tmp_path: Path) -> tuple[dict, list[Path]]:
    benchmark = deepcopy(load_image_benchmark_lock())
    sources = []
    artifacts = []
    for relative, content in (
        ("fp_450/real.parquet", b"fixture-real"),
        ("syncred_600/synthetic.parquet", b"fixture-synthetic"),
    ):
        path = tmp_path / "sources" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        sources.append(path)
        artifacts.append(
            {
                "path": relative,
                "sizeBytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    benchmark["source"]["artifacts"] = artifacts
    benchmark["source"]["subsets"][0]["recordCount"] = 1
    benchmark["source"]["subsets"][1]["recordCount"] = 1
    benchmark["protocol"]["recordCount"] = 2
    return benchmark, sources


def test_private_image_pipeline_is_reproducible_end_to_end(tmp_path: Path) -> None:
    benchmark, sources = _small_source_lock(tmp_path)

    def rows(path: Path):
        subset = path.parent.name
        yield {
            "subset": subset,
            "image": {"bytes": _png("white" if subset == "fp_450" else "black")},
            "prefix": "NEWS",
            "style": "Captured",
        }

    materialized = tmp_path / "private"
    materialize_image_benchmark(benchmark, sources, materialized, row_reader=rows)
    verified = load_image_dataset_inputs(
        materialized / "inputs.jsonl",
        image_root=materialized / "images",
    )
    preprocessed = tmp_path / "preprocessed"
    write_preprocessed_image_dataset(verified, preprocessed)
    variants = load_preprocessed_image_variants(preprocessed)
    labels = load_image_benchmark_labels(materialized / "labels.jsonl")

    candidates = load_image_candidate_registry()
    run_roots = []
    for candidate in candidates:
        adapter = CandidateImageModelAdapter(candidate, PixelBackend())
        manifest, predictions = build_image_benchmark_predictions(
            variants,
            labels,
            adapter=adapter,
        )
        output = tmp_path / f"run-{candidate.adapter_id}"
        write_image_benchmark_run(
            manifest,
            predictions,
            benchmark=benchmark,
            candidate=candidate,
            output_root=output,
            run_id=f"fixture-{candidate.adapter_id}",
            git_commit="abcdef1",
            seed=42,
            device="cpu",
            benchmark_sha256="a" * 64,
            candidate_registry_sha256="b" * 64,
            labels_sha256=hashlib.sha256((materialized / "labels.jsonl").read_bytes()).hexdigest(),
            preprocessed_manifest_sha256=hashlib.sha256((preprocessed / "manifest.jsonl").read_bytes()).hexdigest(),
        )
        run_roots.append(output)

    evidence = [load_image_benchmark_run(root) for root in run_roots]
    report = compare_image_run_evidence(
        evidence,
        benchmark=benchmark,
        benchmark_sha256="a" * 64,
        expected_candidates={candidate.adapter_id: candidate.revision for candidate in candidates},
    )

    assert len(variants) == 12
    assert all(package["result"]["metrics"]["macroF1"] == 1.0 for package in evidence)
    assert report["selection"]["status"] == "experimental-leader"
    assert len(report["selection"]["leaders"]) == 1
