from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nguven_evaluation.materialization import (
    BenchmarkMaterializationError,
    fetch_locked_sources,
    materialize_benchmark,
    write_materialized_benchmark,
)


def _source(
    source_id: str,
    *,
    label: str,
    sample_count: int,
    generator: str | None,
) -> dict:
    return {
        "sourceId": source_id,
        "repository": f"fixture/{source_id}",
        "revision": "a" * 40,
        "configuration": None,
        "split": "train",
        "textField": "query",
        "extraction": "synthetic-query",
        "sampleCount": sample_count,
        "artifacts": [],
        "label": label,
        "license": ["MIT"],
        "sourceUrl": f"https://example.invalid/{source_id}",
        "generatorModel": generator,
        "generatorFamily": generator,
        "intendedUse": "fixture",
        "eligibility": "benchmark-candidate",
    }


def _records(path: Path, prefix: str, count: int) -> None:
    records = [
        {
            "pid": f"{prefix}-{index}",
            "query": (
                f"{prefix} örneği {index}: Bu kayıt deterministik benchmark "
                "materyalizasyon denetimini güvenli ve tekrarlanabilir biçimde sınar."
            ),
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_materialization_balances_labels_sources_and_splits(tmp_path: Path) -> None:
    sources = [
        _source("human-source", label="human", sample_count=8, generator=None),
        _source("generator-a", label="synthetic", sample_count=4, generator="family-a"),
        _source("generator-b", label="synthetic", sample_count=4, generator="family-b"),
    ]
    paths: dict[str, list[Path]] = {}
    for source in sources:
        path = tmp_path / f"{source['sourceId']}.jsonl"
        _records(path, source["sourceId"], 12)
        paths[source["sourceId"]] = [path]

    lock = {
        "benchmarkId": "fixture",
        "version": "v1",
        "preprocessingVersion": "tr-text-v1",
        "sampling": {
            "seed": 19,
            "targetRecordCount": 16,
            "splitRatios": {"train": 0.5, "validation": 0.25, "test": 0.25},
        },
        "sources": sources,
    }
    first = materialize_benchmark(
        lock,
        paths,
        accessed_at="2026-08-29T12:00:00+03:00",
    )
    second = materialize_benchmark(
        lock,
        paths,
        accessed_at="2026-08-29T12:00:00+03:00",
    )

    assert first.manifest_records == second.manifest_records
    assert first.input_records == second.input_records
    assert first.summary["labelCounts"] == {"human": 8, "synthetic": 8}
    assert first.summary["sourceCounts"] == {
        "generator-a": 4,
        "generator-b": 4,
        "human-source": 8,
    }
    assert first.summary["splitLabelCounts"] == {
        "test:human": 2,
        "test:synthetic": 2,
        "train:human": 4,
        "train:synthetic": 4,
        "validation:human": 2,
        "validation:synthetic": 2,
    }
    assert all("<think" not in item["text"].lower() for item in first.input_records)

    output = tmp_path / "private-release"
    write_materialized_benchmark(first, output)
    assert (output / "manifest.jsonl").is_file()
    assert (output / "preprocessed.jsonl").is_file()


def test_fetch_rejects_mismatched_locked_hash(tmp_path: Path) -> None:
    source = _source("generator-a", label="synthetic", sample_count=1, generator="a")
    cached = tmp_path / "generator-a" / "data.jsonl"
    cached.parent.mkdir(parents=True)
    cached.write_text("safe fixture", encoding="utf-8")
    source["artifacts"] = [
        {
            "path": "data.jsonl",
            "size": cached.stat().st_size,
            "sha256": hashlib.sha256(b"different").hexdigest(),
        }
    ]

    with pytest.raises(BenchmarkMaterializationError, match="SHA-256 mismatch"):
        fetch_locked_sources({"sources": [source]}, tmp_path)
