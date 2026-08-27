from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.integrity import (
    DatasetIntegrityError,
    compute_text_content_hash,
    verify_dataset_content_hashes,
)


def manifest_record(
    record_id: str,
    text: str,
    *,
    content_type: str = "text/plain; charset=utf-8",
) -> dict[str, object]:
    return {
        "id": record_id,
        "source": "synthetic-test-fixture",
        "sourceUrl": f"https://example.invalid/{record_id}",
        "accessedAt": "2026-08-27T12:00:00Z",
        "license": "test-only",
        "contentHash": compute_text_content_hash(text),
        "language": "tr",
        "contentType": content_type,
        "label": "synthetic",
        "labelSource": "test-fixture",
        "generatorModel": "fixture-generator-v1",
        "transformation": "none",
        "intendedUse": "Automated testing only",
        "split": "train",
    }


def input_record(record_id: str, text: str) -> dict[str, str]:
    return {"id": record_id, "text": text}


def write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_compute_text_hash_uses_exact_utf8_bytes() -> None:
    text = "Türkçe içerik: İ, ı, ş, ğ"

    assert compute_text_content_hash(text) == (
        "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def test_verify_dataset_content_hashes_accepts_complete_matching_records() -> None:
    text = "Bütünlük kontrolü için sentetik metin."

    report = verify_dataset_content_hashes(
        [manifest_record("sample-001", text)],
        [input_record("sample-001", text)],
    )

    assert report.verified_record_count == 1


def test_verify_dataset_content_hashes_rejects_mismatch_without_exposing_text() -> None:
    private_text = "BU-HAM-METIN-HATA-MESAJINA-GIRMEMELI"

    with pytest.raises(DatasetIntegrityError) as captured:
        verify_dataset_content_hashes(
            [manifest_record("sample-001", "Beklenen metin")],
            [input_record("sample-001", private_text)],
        )

    assert "content hash mismatch for id(s): sample-001" in str(captured.value)
    assert private_text not in str(captured.value)


def test_verify_dataset_content_hashes_rejects_coverage_mismatch() -> None:
    with pytest.raises(DatasetIntegrityError) as captured:
        verify_dataset_content_hashes(
            [manifest_record("missing", "Eksik metin")],
            [input_record("unexpected", "Fazla metin")],
        )

    assert "missing input id(s): missing" in str(captured.value)
    assert "unexpected input id(s): unexpected" in str(captured.value)


def test_verify_dataset_content_hashes_rejects_duplicate_manifest_ids() -> None:
    text = "Tekil olması gereken kayıt."
    duplicate = manifest_record("duplicate", text)

    with pytest.raises(DatasetIntegrityError, match="manifest contains duplicate id"):
        verify_dataset_content_hashes(
            [duplicate, dict(duplicate)],
            [input_record("duplicate", text)],
        )


def test_verify_dataset_content_hashes_rejects_non_text_manifest_record() -> None:
    text = "Metin girdisi bir görsel manifestine bağlanamaz."

    with pytest.raises(DatasetIntegrityError, match="non-text manifest record"):
        verify_dataset_content_hashes(
            [manifest_record("sample-001", text, content_type="image/png")],
            [input_record("sample-001", text)],
        )


def test_cli_verifies_hashes_without_printing_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_text = "CLI-CIKTISINDA-GORUNMEMELI"
    manifest_path = write_json(
        tmp_path / "manifest.json",
        [manifest_record("sample-001", private_text)],
    )
    input_path = write_json(
        tmp_path / "input.json",
        [input_record("sample-001", private_text)],
    )

    exit_code = main(
        [
            "verify-content-hashes",
            str(manifest_path),
            "--input",
            str(input_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Verified 1 content hash" in output
    assert private_text not in output
