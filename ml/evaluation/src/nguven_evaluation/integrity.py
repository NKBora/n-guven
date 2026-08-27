"""Integrity checks joining dataset manifests to local text inputs."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Sequence


class DatasetIntegrityError(ValueError):
    """Raised when manifest and local input records fail integrity checks."""


@dataclass(frozen=True)
class IntegrityReport:
    """Non-sensitive summary of a successful integrity verification."""

    verified_record_count: int


def compute_text_content_hash(text: str) -> str:
    """Return the manifest hash for the exact decoded text encoded as UTF-8."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def verify_dataset_content_hashes(
    manifest_records: Sequence[dict[str, Any]],
    input_records: Sequence[dict[str, Any]],
) -> IntegrityReport:
    """Verify one-to-one ID coverage, text media type, and SHA-256 content hashes."""
    manifest_by_id = _index_unique(manifest_records, artifact_name="manifest")
    input_by_id = _index_unique(input_records, artifact_name="input")

    issues: list[str] = []
    missing_ids = sorted(set(manifest_by_id) - set(input_by_id))
    unexpected_ids = sorted(set(input_by_id) - set(manifest_by_id))
    if missing_ids:
        issues.append("missing input id(s): " + ", ".join(missing_ids))
    if unexpected_ids:
        issues.append("unexpected input id(s): " + ", ".join(unexpected_ids))

    shared_ids = sorted(set(manifest_by_id) & set(input_by_id))
    non_text_ids: list[str] = []
    mismatch_ids: list[str] = []
    for record_id in shared_ids:
        manifest_record = manifest_by_id[record_id]
        input_record = input_by_id[record_id]
        content_type = str(manifest_record["contentType"]).lower()
        if not content_type.startswith("text/"):
            non_text_ids.append(record_id)
            continue

        expected_hash = str(manifest_record["contentHash"]).lower()
        actual_hash = compute_text_content_hash(str(input_record["text"]))
        if not hmac.compare_digest(expected_hash, actual_hash):
            mismatch_ids.append(record_id)

    if non_text_ids:
        issues.append("non-text manifest record id(s): " + ", ".join(non_text_ids))
    if mismatch_ids:
        issues.append("content hash mismatch for id(s): " + ", ".join(mismatch_ids))

    if issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise DatasetIntegrityError(f"Dataset content integrity verification failed:\n{details}")

    return IntegrityReport(verified_record_count=len(shared_ids))


def _index_unique(
    records: Sequence[dict[str, Any]],
    *,
    artifact_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for record in records:
        record_id = str(record["id"])
        if record_id in indexed:
            duplicates.add(record_id)
        indexed[record_id] = record
    if duplicates:
        raise DatasetIntegrityError(
            f"Dataset {artifact_name} contains duplicate id(s): "
            + ", ".join(sorted(duplicates))
        )
    return indexed
