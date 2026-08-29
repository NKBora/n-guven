"""Deterministic, hash-verified materialization of the Turkish text benchmark."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import shutil
import tempfile
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from nguven_evaluation.integrity import compute_text_content_hash
from nguven_evaluation.offline_preprocessing import build_preprocessed_records
from nguven_evaluation.preprocessing import DEFAULT_PREPROCESSING_VERSION
from nguven_evaluation.splitting import audit_manifest

MIN_TEXT_CHARACTERS = 80
MAX_TEXT_CHARACTERS = 2048
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
SPLIT_ORDER = ("train", "validation", "test")


class BenchmarkMaterializationError(ValueError):
    """Raised when reviewed source material cannot produce the frozen benchmark."""


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    source_id: str
    original_id: str
    text: str


@dataclass(frozen=True, slots=True)
class MaterializedBenchmark:
    manifest_records: list[dict[str, Any]]
    input_records: list[dict[str, str]]
    preprocessed_records: list[dict[str, Any]]
    summary: dict[str, Any]


def fetch_locked_sources(
    lock: Mapping[str, Any],
    cache_root: Path,
    *,
    allow_network: bool = False,
) -> dict[str, list[Path]]:
    """Fetch every pinned source artifact and verify its declared SHA-256."""
    if cache_root.is_symlink():
        raise BenchmarkMaterializationError("Source cache must not be a symbolic link")
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, list[Path]] = {}
    for source in lock["sources"]:
        source_id = str(source["sourceId"])
        source_paths: list[Path] = []
        for artifact in source["artifacts"]:
            relative = Path(str(artifact["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise BenchmarkMaterializationError("Source artifact path is unsafe")
            destination = cache_root / source_id / relative.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                if not allow_network:
                    raise BenchmarkMaterializationError(
                        f"Missing locked source artifact with network disabled: {source_id}/{relative}"
                    )
                _download_source_artifact(
                    repository=str(source["repository"]),
                    revision=str(source["revision"]),
                    artifact_path=relative.as_posix(),
                    destination=destination,
                )
            _verify_regular_file(
                destination,
                expected_size=int(artifact["size"]),
                expected_sha256=str(artifact["sha256"]),
            )
            source_paths.append(destination)
        resolved[source_id] = source_paths
    return resolved


def materialize_benchmark(
    lock: Mapping[str, Any],
    source_paths: Mapping[str, Sequence[Path]],
    *,
    accessed_at: str,
) -> MaterializedBenchmark:
    """Create balanced private records and non-sensitive audit evidence."""
    _validate_accessed_at(accessed_at)
    seed = int(lock["sampling"]["seed"])
    sources = {str(source["sourceId"]): source for source in lock["sources"]}
    missing = sorted(set(sources) - set(source_paths))
    if missing:
        raise BenchmarkMaterializationError(
            "Missing materialization source path(s): " + ", ".join(missing)
        )

    selected_by_source: dict[str, list[SourceCandidate]] = {}
    seen_hashes: set[str] = set()
    for source_id, source in sources.items():
        if source["label"] == "human":
            continue
        candidates = _load_candidates(source, source_paths[source_id])
        selected_by_source[source_id] = _select_candidates(
            candidates,
            count=int(source["sampleCount"]),
            seed=seed,
            seen_hashes=seen_hashes,
        )

    synthetic_lengths = [
        len(candidate.text)
        for source_id in sorted(selected_by_source)
        for candidate in selected_by_source[source_id]
    ]
    if not synthetic_lengths:
        raise BenchmarkMaterializationError("Benchmark contains no synthetic records")

    for source_id, source in sources.items():
        if source["label"] != "human":
            continue
        candidates = _load_candidates(source, source_paths[source_id])
        selected = _select_candidates(
            candidates,
            count=int(source["sampleCount"]),
            seed=seed,
            seen_hashes=seen_hashes,
            target_lengths=synthetic_lengths,
        )
        selected_by_source[source_id] = selected

    manifest_records: list[dict[str, Any]] = []
    input_records: list[dict[str, str]] = []
    for source_id in sorted(selected_by_source):
        source = sources[source_id]
        split_candidates = _assign_exact_splits(
            selected_by_source[source_id],
            ratios=lock["sampling"]["splitRatios"],
            seed=seed,
        )
        for split, candidate in split_candidates:
            content_hash = compute_text_content_hash(candidate.text)
            record_id = f"{source_id}-{content_hash.removeprefix('sha256:')[:20]}"
            manifest_records.append(
                {
                    "id": record_id,
                    "source": source_id,
                    "sourceGroup": f"{source_id}:{candidate.original_id}",
                    "sourceUrl": source["sourceUrl"],
                    "accessedAt": accessed_at,
                    "license": " OR ".join(source["license"]),
                    "contentHash": content_hash,
                    "language": "tr",
                    "contentType": "text/plain",
                    "label": source["label"],
                    "labelSource": f"pinned-generator-provenance:{source_id}",
                    "generatorModel": source["generatorModel"],
                    "generatorFamily": source["generatorFamily"],
                    "transformation": "custom:benchmark-excerpt-v1",
                    "intendedUse": "N-Güven Turkish text-origin benchmark v1",
                    "split": split,
                }
            )
            input_records.append({"id": record_id, "text": candidate.text})

    manifest_records.sort(key=lambda item: str(item["id"]))
    input_records.sort(key=lambda item: item["id"])
    audit_manifest(manifest_records)
    preprocessed_records = build_preprocessed_records(
        manifest_records,
        input_records,
        version=DEFAULT_PREPROCESSING_VERSION,
    )
    summary = _build_summary(lock, manifest_records, input_records, accessed_at=accessed_at)
    if summary["recordCount"] != int(lock["sampling"]["targetRecordCount"]):
        raise BenchmarkMaterializationError("Materialized record count differs from source lock")
    return MaterializedBenchmark(
        manifest_records=manifest_records,
        input_records=input_records,
        preprocessed_records=preprocessed_records,
        summary=summary,
    )


def write_materialized_benchmark(
    materialized: MaterializedBenchmark,
    output_root: Path,
) -> None:
    """Atomically write private text artifacts and a non-sensitive summary."""
    if output_root.exists() or output_root.is_symlink():
        raise BenchmarkMaterializationError("Materialization output must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        os.chmod(temporary_root, 0o700)
        _write_jsonl(temporary_root / "manifest.jsonl", materialized.manifest_records)
        _write_jsonl(temporary_root / "input.jsonl", materialized.input_records)
        _write_jsonl(temporary_root / "preprocessed.jsonl", materialized.preprocessed_records)
        _write_json(temporary_root / "summary.json", materialized.summary)
        os.replace(temporary_root, output_root)
    except OSError as error:
        raise BenchmarkMaterializationError("Unable to publish benchmark materialization") from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def sha256_regular_file(path: Path) -> str:
    _verify_regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_candidates(
    source: Mapping[str, Any],
    paths: Sequence[Path],
) -> Iterator[SourceCandidate]:
    extraction = str(source["extraction"])
    source_id = str(source["sourceId"])
    if extraction == "wikipedia-article":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise BenchmarkMaterializationError(
                "Install the materialization extra to read pinned Parquet sources"
            ) from error
        for path in paths:
            _verify_regular_file(path)
            table = parquet.ParquetFile(path)
            for batch in table.iter_batches(columns=["id", "text"], batch_size=2048):
                data = batch.to_pydict()
                for original_id, raw_text in zip(data["id"], data["text"], strict=True):
                    text = _normalize_text(raw_text)
                    if text:
                        yield SourceCandidate(source_id, str(original_id), text)
        return

    for path in paths:
        _verify_regular_file(path)
        with path.open("r", encoding="utf-8") as source_file:
            for line_number, raw_line in enumerate(source_file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise BenchmarkMaterializationError(
                        f"Invalid JSON in {source_id} at line {line_number}"
                    ) from error
                if extraction == "assistant-final-answer":
                    original_id = str(record.get("id", line_number))
                    text = _extract_final_answer(record)
                elif extraction == "synthetic-query":
                    original_id = str(record.get("pid", line_number)) + f":{line_number}"
                    text = _normalize_text(record.get(str(source["textField"])))
                else:
                    raise BenchmarkMaterializationError(
                        f"Unsupported source extraction contract: {extraction}"
                    )
                if text:
                    yield SourceCandidate(source_id, original_id, text)


def _extract_final_answer(record: Mapping[str, Any]) -> str | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    assistant = next(
        (
            item.get("content")
            for item in reversed(messages)
            if isinstance(item, Mapping) and item.get("role") == "assistant"
        ),
        None,
    )
    if not isinstance(assistant, str):
        return None
    without_reasoning = re.sub(r"<think>.*?</think>", " ", assistant, flags=re.DOTALL | re.IGNORECASE)
    if "<think" in without_reasoning.lower() or "</think" in without_reasoning.lower():
        return None
    return _normalize_text(without_reasoning)


def _normalize_text(value: Any, *, target_length: int | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", " ", value).strip()
    limit = min(target_length or MAX_TEXT_CHARACTERS, MAX_TEXT_CHARACTERS)
    if len(normalized) > limit:
        boundary = normalized.rfind(" ", 0, limit + 1)
        normalized = normalized[: boundary if boundary >= MIN_TEXT_CHARACTERS else limit].strip()
    if len(normalized) < MIN_TEXT_CHARACTERS:
        return None
    return normalized


def _select_candidates(
    candidates: Iterable[SourceCandidate],
    *,
    count: int,
    seed: int,
    seen_hashes: set[str],
    target_lengths: Sequence[int] | None = None,
) -> list[SourceCandidate]:
    ranked: list[tuple[int, int, SourceCandidate]] = []
    local_hashes: set[str] = set()
    targets = list(target_lengths or ())
    source_id = "unknown"
    for sequence, candidate in enumerate(candidates):
        source_id = candidate.source_id
        text = candidate.text
        if targets:
            target_index = int.from_bytes(
                hashlib.sha256(f"{seed}:{candidate.original_id}".encode()).digest()[:8],
                "big",
            ) % len(targets)
            text = _normalize_text(text, target_length=targets[target_index]) or ""
        content_hash = compute_text_content_hash(text) if text else ""
        if not text or content_hash in seen_hashes or content_hash in local_hashes:
            continue
        local_hashes.add(content_hash)
        rank = int.from_bytes(hashlib.sha256(
            f"{seed}:{candidate.source_id}:{candidate.original_id}:{content_hash}".encode()
        ).digest(), "big")
        entry = (-rank, sequence, SourceCandidate(candidate.source_id, candidate.original_id, text))
        if len(ranked) < count:
            heapq.heappush(ranked, entry)
        elif entry[0] > ranked[0][0]:
            heapq.heapreplace(ranked, entry)
    selected = [
        candidate
        for _, _, candidate in sorted(ranked, key=lambda item: -item[0])
    ]
    if len(selected) != count:
        raise BenchmarkMaterializationError(
            f"Source {source_id} "
            f"provided {len(selected)} eligible unique records; required {count}"
        )
    seen_hashes.update(compute_text_content_hash(candidate.text) for candidate in selected)
    return selected


def _assign_exact_splits(
    candidates: Sequence[SourceCandidate],
    *,
    ratios: Mapping[str, Any],
    seed: int,
) -> list[tuple[str, SourceCandidate]]:
    ranked = sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{seed}:split:{item.source_id}:{item.original_id}".encode()
        ).digest(),
    )
    counts = {
        "train": int(len(ranked) * float(ratios["train"])),
        "validation": int(len(ranked) * float(ratios["validation"])),
    }
    counts["test"] = len(ranked) - counts["train"] - counts["validation"]
    assigned: list[tuple[str, SourceCandidate]] = []
    offset = 0
    for split in SPLIT_ORDER:
        end = offset + counts[split]
        assigned.extend((split, candidate) for candidate in ranked[offset:end])
        offset = end
    return assigned


def _build_summary(
    lock: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, str]],
    *,
    accessed_at: str,
) -> dict[str, Any]:
    input_by_id = {item["id"]: item["text"] for item in inputs}
    lengths = sorted(len(input_by_id[str(item["id"])]) for item in manifest)
    label_counts = Counter(str(item["label"]) for item in manifest)
    split_counts = Counter(f"{item['split']}:{item['label']}" for item in manifest)
    source_counts = Counter(str(item["source"]) for item in manifest)
    generator_counts = Counter(
        str(item["generatorFamily"])
        for item in manifest
        if item.get("generatorFamily") is not None
    )
    return {
        "schemaVersion": "1.0",
        "benchmarkId": lock["benchmarkId"],
        "benchmarkVersion": lock["version"],
        "materializedAt": accessed_at,
        "samplingSeed": lock["sampling"]["seed"],
        "recordCount": len(manifest),
        "labelCounts": dict(sorted(label_counts.items())),
        "splitLabelCounts": dict(sorted(split_counts.items())),
        "sourceCounts": dict(sorted(source_counts.items())),
        "generatorFamilyCounts": dict(sorted(generator_counts.items())),
        "characterLengths": {
            "minimum": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p95": lengths[int((len(lengths) - 1) * 0.95)],
            "maximum": lengths[-1],
        },
        "controls": {
            "contentDeduplication": "sha256-exact",
            "lengthMatching": "synthetic-distribution-targeted",
            "hiddenReasoningExcluded": True,
            "rawTextCommitted": False,
        },
    }


def _download_source_artifact(
    *,
    repository: str,
    revision: str,
    artifact_path: str,
    destination: Path,
) -> None:
    url = (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/"
        f"{artifact_path}?download=true"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            request = urllib.request.Request(url, headers={"User-Agent": "n-guven-benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output, length=DOWNLOAD_CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    except (OSError, urllib.error.URLError) as error:
        raise BenchmarkMaterializationError(
            f"Unable to download pinned source artifact: {repository}/{artifact_path}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_regular_file(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkMaterializationError(f"Source artifact must be a regular file: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise BenchmarkMaterializationError(f"Source artifact size mismatch: {path}")
    if expected_sha256 is not None:
        hasher = hashlib.sha256()
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(DOWNLOAD_CHUNK_BYTES), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected_sha256:
            raise BenchmarkMaterializationError(f"Source artifact SHA-256 mismatch: {path}")


def _validate_accessed_at(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BenchmarkMaterializationError("Access time must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise BenchmarkMaterializationError("Access time must include a timezone")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        json.dump(document, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
