/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Hash-verified private materialization of the frozen SynCred image benchmark."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import urllib.request
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image, UnidentifiedImageError

from nguven_evaluation.image_dataset_inputs import DEFAULT_MAX_IMAGE_BYTES


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_LABEL_SCHEMA_PATH = (
    EVALUATION_ROOT / "image" / "benchmarks" / "labels.schema.json"
)
SUPPORTED_FORMAT_SUFFIXES = {
    "JPEG": ".jpg",
    "MPO": ".mpo",
    "PNG": ".png",
    "WEBP": ".webp",
}


class ImageBenchmarkMaterializationError(ValueError):
    """Raised when locked source bytes cannot produce the private benchmark."""


RowReader = Callable[[Path], Iterable[Mapping[str, Any]]]


def fetch_locked_image_sources(
    lock: Mapping[str, Any],
    cache_root: Path,
    *,
    allow_network: bool = False,
) -> tuple[Path, ...]:
    """Fetch only pinned Parquet artifacts and verify every byte before use."""
    if cache_root.is_symlink():
        raise ImageBenchmarkMaterializationError("Image source cache must not be a symlink")
    cache_root.mkdir(parents=True, exist_ok=True)
    repository = str(lock["source"]["repository"])
    revision = str(lock["source"]["revision"])
    paths: list[Path] = []
    for artifact in lock["source"]["artifacts"]:
        relative = Path(str(artifact["path"]))
        destination = cache_root / relative
        if destination.is_file() and not destination.is_symlink():
            _verify_locked_file(destination, artifact)
            paths.append(destination)
            continue
        if not allow_network:
            raise ImageBenchmarkMaterializationError(
                f"Locked image source is unavailable locally: {relative.as_posix()}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{repository}/resolve/{revision}/"
            f"{relative.as_posix()}"
        )
        _download_locked_file(url, destination, artifact)
        paths.append(destination)
    return tuple(paths)


def materialize_image_benchmark(
    lock: Mapping[str, Any],
    source_paths: Iterable[Path],
    output_root: Path,
    *,
    row_reader: RowReader | None = None,
    label_schema_path: Path = DEFAULT_IMAGE_LABEL_SCHEMA_PATH,
) -> dict[str, Any]:
    """Extract exact embedded image bytes and publish private manifests atomically."""
    if output_root.exists() or output_root.is_symlink():
        raise ImageBenchmarkMaterializationError("Image benchmark output must not exist")
    if row_reader is None:
        row_reader = _iter_parquet_rows
    source_paths = tuple(source_paths)
    schema = _load_json_object(label_schema_path, "image label schema")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    expected_artifacts = {str(item["path"]): item for item in lock["source"]["artifacts"]}
    supplied = {path.as_posix(): path for path in source_paths}
    by_name = {path.name: path for path in source_paths}
    if len(by_name) != len(source_paths):
        by_name = {}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    os.chmod(temporary_root, 0o700)
    inputs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    try:
        for relative_name, artifact in expected_artifacts.items():
            source_path = supplied.get(relative_name) or by_name.get(Path(relative_name).name)
            if source_path is None:
                matches = [path for path in source_paths if path.as_posix().endswith(relative_name)]
                source_path = matches[0] if len(matches) == 1 else None
            if source_path is None:
                raise ImageBenchmarkMaterializationError(
                    f"Missing locked image source: {relative_name}"
                )
            _verify_locked_file(source_path, artifact)
            expected_subset = Path(relative_name).parts[0]
            for row in row_reader(source_path):
                if str(row.get("subset")) != expected_subset:
                    raise ImageBenchmarkMaterializationError(
                        f"Parquet subset mismatch in {relative_name}"
                    )
                sequence = counts[expected_subset]
                record_id = f"{expected_subset}-{sequence:04d}"
                image_value = row.get("image")
                if not isinstance(image_value, Mapping) or not isinstance(image_value.get("bytes"), bytes):
                    raise ImageBenchmarkMaterializationError("Parquet image must contain embedded bytes")
                image_bytes = image_value["bytes"]
                suffix = _validate_image_bytes(image_bytes)
                relative_image = Path(record_id) / f"original{suffix}"
                destination = temporary_root / "images" / relative_image
                destination.parent.mkdir(parents=True, mode=0o700)
                destination.write_bytes(image_bytes)
                os.chmod(destination, 0o600)
                digest = hashlib.sha256(image_bytes).hexdigest()
                inputs.append(
                    {
                        "id": record_id,
                        "path": relative_image.as_posix(),
                        "sha256": digest,
                        "sizeBytes": len(image_bytes),
                    }
                )
                label = "human" if expected_subset == "fp_450" else "synthetic"
                label_record = {
                    "id": record_id,
                    "label": label,
                    "sourceGroup": expected_subset,
                    "generatorFamily": None if label == "human" else "syncred-multi-generator",
                    "contentCategory": str(row.get("prefix") or "unknown"),
                    "circulationStyle": str(row.get("style") or "unknown"),
                }
                if list(validator.iter_errors(label_record)):
                    raise ImageBenchmarkMaterializationError("Generated image label violates schema")
                labels.append(label_record)
                counts[expected_subset] += 1

        expected_counts = {
            str(item["configuration"]): int(item["recordCount"])
            for item in lock["source"]["subsets"]
        }
        if dict(counts) != expected_counts:
            raise ImageBenchmarkMaterializationError(
                f"Image benchmark subset counts differ: {dict(counts)}"
            )
        inputs_path = temporary_root / "inputs.jsonl"
        labels_path = temporary_root / "labels.jsonl"
        _write_jsonl(inputs_path, inputs)
        _write_jsonl(labels_path, labels)
        summary = {
            "benchmarkId": lock["benchmarkId"],
            "version": lock["version"],
            "sourceRevision": lock["source"]["revision"],
            "recordCount": len(inputs),
            "labelCounts": dict(sorted(Counter(item["label"] for item in labels).items())),
            "inputsSha256": _sha256_file(inputs_path),
            "labelsSha256": _sha256_file(labels_path),
        }
        summary_path = temporary_root / "summary.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(summary_path, 0o600)
        os.replace(temporary_root, output_root)
        return summary
    except (OSError, ImageBenchmarkMaterializationError):
        raise
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _iter_parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImageBenchmarkMaterializationError(
            'Image benchmark materialization requires: pip install -e ".[materialization]"'
        ) from error
    parquet_file = parquet.ParquetFile(path)
    columns = ["subset", "image", "prefix", "style"]
    for batch in parquet_file.iter_batches(batch_size=16, columns=columns):
        yield from batch.to_pylist()


def _validate_image_bytes(data: bytes) -> str:
    if not data or len(data) > DEFAULT_MAX_IMAGE_BYTES:
        raise ImageBenchmarkMaterializationError("Embedded image size is outside limits")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                if image_format not in SUPPORTED_FORMAT_SUFFIXES:
                    raise ImageBenchmarkMaterializationError("Unsupported embedded image format")
                if image_format == "MPO" and int(getattr(image, "n_frames", 1)) != 2:
                    raise ImageBenchmarkMaterializationError(
                        "MPO benchmark image must contain exactly two views"
                    )
                if image_format != "MPO" and bool(getattr(image, "is_animated", False)):
                    raise ImageBenchmarkMaterializationError("Animated benchmark images are unsupported")
                image.verify()
                return SUPPORTED_FORMAT_SUFFIXES[image_format]
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError) as error:
        raise ImageBenchmarkMaterializationError("Unable to verify embedded image") from error


def _download_locked_file(url: str, destination: Path, artifact: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            os.chmod(temporary, 0o600)
            request = urllib.request.Request(url, headers={"User-Agent": "n-guven-image-benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, target, length=1024 * 1024)
        _verify_locked_file(temporary, artifact)
        os.replace(temporary, destination)
        temporary = None
    except (OSError, ValueError) as error:
        raise ImageBenchmarkMaterializationError("Unable to download locked image source") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_locked_file(path: Path, artifact: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ImageBenchmarkMaterializationError("Locked image source must be a regular file")
    if path.stat().st_size != int(artifact["sizeBytes"]):
        raise ImageBenchmarkMaterializationError(f"Locked image source size mismatch: {path.name}")
    if _sha256_file(path) != str(artifact["sha256"]):
        raise ImageBenchmarkMaterializationError(f"Locked image source hash mismatch: {path.name}")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    os.chmod(path, 0o600)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ImageBenchmarkMaterializationError(f"{label.title()} must be a regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageBenchmarkMaterializationError(f"Unable to parse {label}") from error
    if not isinstance(document, dict):
        raise ImageBenchmarkMaterializationError(f"{label.title()} must be a JSON object")
    return document
