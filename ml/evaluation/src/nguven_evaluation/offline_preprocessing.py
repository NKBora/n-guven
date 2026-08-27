"""Secure offline orchestration for versioned text preprocessing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator

from nguven_evaluation.integrity import verify_dataset_content_hashes
from nguven_evaluation.preprocessing import (
    DEFAULT_PREPROCESSING_VERSION,
    preprocess_turkish_text,
)

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESSED_SCHEMA_PATH = EVALUATION_ROOT / "preprocessed" / "schema.json"


class OfflinePreprocessingError(ValueError):
    """Raised when an offline preprocessing artifact cannot be produced safely."""


def build_preprocessed_records(
    manifest_records: Sequence[dict[str, Any]],
    input_records: Sequence[dict[str, Any]],
    *,
    version: str = DEFAULT_PREPROCESSING_VERSION,
    schema_path: Path = DEFAULT_PREPROCESSED_SCHEMA_PATH,
) -> list[dict[str, Any]]:
    """Verify raw inputs, preprocess them, and validate the output contract."""
    verify_dataset_content_hashes(manifest_records, input_records)
    input_by_id = {str(record["id"]): record for record in input_records}

    output_records: list[dict[str, Any]] = []
    for record_id in sorted(input_by_id):
        result = preprocess_turkish_text(
            str(input_by_id[record_id]["text"]),
            version=version,
        )
        output_records.append(
            {
                "id": record_id,
                "text": result.text,
                "preprocessingVersion": result.preprocessing_version,
                "inputContentHash": result.input_content_hash,
                "outputContentHash": result.output_content_hash,
                "inputCharacters": result.input_characters,
                "outputCharacters": result.output_characters,
            }
        )

    _validate_output_records(output_records, schema_path=schema_path)
    return output_records


def write_private_jsonl(
    records: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    force: bool = False,
) -> None:
    """Atomically write local-only JSONL with owner-read/write permissions."""
    if output_path.suffix.lower() != ".jsonl":
        raise OfflinePreprocessingError("Preprocessing output must use the .jsonl extension")
    if output_path.is_symlink():
        raise OfflinePreprocessingError(
            f"Preprocessing output must not be a symbolic link: {output_path}"
        )
    if output_path.exists() and not force:
        raise OfflinePreprocessingError(
            f"Output already exists; use --force to replace it: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            os.chmod(temporary_path, 0o600)
            for record in records:
                temporary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as error:
        raise OfflinePreprocessingError(
            f"Unable to write preprocessing output: {output_path}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ensure_distinct_artifact_paths(output_path: Path, protected_paths: Sequence[Path]) -> None:
    """Prevent an output path from replacing any source or schema artifact."""
    output_resolved = output_path.resolve(strict=False)
    for protected_path in protected_paths:
        if output_resolved == protected_path.resolve(strict=False):
            raise OfflinePreprocessingError(
                f"Preprocessing output must differ from input artifacts: {output_path}"
            )


def _validate_output_records(
    records: Sequence[dict[str, Any]],
    *,
    schema_path: Path,
) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise OfflinePreprocessingError(
            f"Unable to load preprocessing output schema: {schema_path}"
        ) from error
    if not isinstance(schema, dict):
        raise OfflinePreprocessingError(
            f"Preprocessing output schema must be a JSON object: {schema_path}"
        )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    invalid_records: list[int] = []
    for record_number, record in enumerate(records, start=1):
        if any(validator.iter_errors(record)):
            invalid_records.append(record_number)
    if invalid_records:
        raise OfflinePreprocessingError(
            "Generated preprocessing records violate the output schema at record(s): "
            + ", ".join(str(number) for number in invalid_records)
        )
