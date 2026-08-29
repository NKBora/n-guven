"""Versioned source locks and evidence gates for the Turkish text benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_PATH = EVALUATION_ROOT / "benchmarks" / "text-origin-tr-v1.json"
DEFAULT_BENCHMARK_SCHEMA_PATH = EVALUATION_ROOT / "benchmarks" / "schema.json"
DEFAULT_MAX_BENCHMARK_BYTES = 1024 * 1024


class BenchmarkContractError(ValueError):
    """Raised when a benchmark lock could permit unreviewed evidence."""


def load_benchmark_lock(
    path: Path = DEFAULT_BENCHMARK_PATH,
    *,
    schema_path: Path = DEFAULT_BENCHMARK_SCHEMA_PATH,
) -> dict[str, Any]:
    """Load and validate one immutable benchmark source lock."""
    document = _load_json_object(path, "benchmark lock")
    schema = _load_json_object(schema_path, "benchmark schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda item: list(item.path),
    )
    if errors:
        locations = [
            ".".join(str(part) for part in error.absolute_path) or "<document>"
            for error in errors
        ]
        raise BenchmarkContractError(
            "Benchmark lock validation failed at: " + ", ".join(locations)
        )
    _validate_semantics(document)
    return document


def benchmark_evidence_allowed(lock: Mapping[str, Any]) -> bool:
    """Return true only for a reviewed, hash-bound materialized release."""
    release = lock["release"]
    return bool(
        release["status"] == "materialized"
        and release["evidenceStatus"] == "reviewed"
        and release["resultsAllowed"] is True
        and release["materializedArtifact"] is not None
    )


def _validate_semantics(lock: Mapping[str, Any]) -> None:
    sources = list(lock["sources"])
    source_ids = [str(item["sourceId"]) for item in sources]
    if len(source_ids) != len(set(source_ids)):
        raise BenchmarkContractError("Benchmark source ids must be unique")
    identities = [(str(item["repository"]), str(item["revision"])) for item in sources]
    if len(identities) != len(set(identities)):
        raise BenchmarkContractError("Benchmark source revisions must be unique")

    labels = {str(item["label"]) for item in sources}
    if labels != {"human", "synthetic"}:
        raise BenchmarkContractError("Benchmark must contain human and synthetic sources")

    synthetic_families: set[str] = set()
    for source in sources:
        label = str(source["label"])
        generator_model = source["generatorModel"]
        generator_family = source["generatorFamily"]
        if label == "human" and (generator_model is not None or generator_family is not None):
            raise BenchmarkContractError("Human sources cannot declare a generator")
        if label == "synthetic":
            if not generator_model or not generator_family:
                raise BenchmarkContractError(
                    "Synthetic sources require model and family provenance"
                )
            synthetic_families.add(str(generator_family))

    minimum = int(lock["sampling"]["minimumGeneratorFamilies"])
    if len(synthetic_families) < minimum:
        raise BenchmarkContractError(
            f"Benchmark requires at least {minimum} synthetic generator families"
        )

    release = lock["release"]
    allowed = benchmark_evidence_allowed(lock)
    if release["resultsAllowed"] is True and not allowed:
        raise BenchmarkContractError(
            "Benchmark results cannot be enabled before reviewed materialization"
        )
    if release["status"] == "source-locked" and release["materializedArtifact"] is not None:
        raise BenchmarkContractError(
            "Source-locked benchmark cannot claim a materialized artifact"
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkContractError(f"{label.title()} must be a regular file")
    if path.stat().st_size > DEFAULT_MAX_BENCHMARK_BYTES:
        raise BenchmarkContractError(f"{label.title()} exceeds the size limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkContractError(f"{label.title()} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise BenchmarkContractError(f"{label.title()} must be a JSON object")
    return document
