from __future__ import annotations

import json
from pathlib import Path

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.manifests import ManifestValidationError, load_manifest


def valid_record(record_id: str = "sample-001") -> dict[str, object]:
    return {
        "id": record_id,
        "source": "synthetic-test-fixture",
        "sourceUrl": "https://example.invalid/datasets/synthetic-test-fixture",
        "accessedAt": "2026-08-27T12:00:00Z",
        "license": "test-only",
        "contentHash": f"sha256:{'a' * 64}",
        "language": "tr",
        "contentType": "text/plain",
        "label": "synthetic",
        "labelSource": "test-fixture",
        "generatorModel": "fixture-generator-v1",
        "transformation": "none",
        "intendedUse": "Automated testing only",
        "split": "train",
    }


def write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_load_manifest_accepts_json_array(tmp_path: Path) -> None:
    manifest = write_json(tmp_path / "manifest.json", [valid_record()])

    records = load_manifest(manifest)

    assert records == [valid_record()]


def test_load_manifest_accepts_json_lines(tmp_path: Path) -> None:
    records = [valid_record("sample-001"), valid_record("sample-002")]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    assert load_manifest(manifest) == records


def test_load_manifest_reports_record_and_field_for_schema_error(tmp_path: Path) -> None:
    record = valid_record()
    record["contentHash"] = "not-a-sha256"
    manifest = write_json(tmp_path / "manifest.json", [record])

    with pytest.raises(ManifestValidationError, match=r"record 1, contentHash"):
        load_manifest(manifest)


def test_load_manifest_rejects_empty_collection(tmp_path: Path) -> None:
    manifest = write_json(tmp_path / "manifest.json", [])

    with pytest.raises(ManifestValidationError, match="contains no records"):
        load_manifest(manifest)


def test_cli_returns_nonzero_for_invalid_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = write_json(tmp_path / "manifest.json", [{"id": "incomplete"}])

    exit_code = main(["validate-manifest", str(manifest)])

    assert exit_code == 1
    assert "failed schema validation" in capsys.readouterr().out
