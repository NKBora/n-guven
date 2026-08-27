from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.integrity import DatasetIntegrityError, compute_text_content_hash
from nguven_evaluation.offline_preprocessing import (
    OfflinePreprocessingError,
    build_preprocessed_records,
    ensure_distinct_artifact_paths,
    write_private_jsonl,
)


def manifest_record(record_id: str, text: str) -> dict[str, object]:
    return {
        "id": record_id,
        "source": "synthetic-test-fixture",
        "sourceUrl": f"https://example.invalid/{record_id}",
        "accessedAt": "2026-08-28T00:00:00Z",
        "license": "test-only",
        "contentHash": compute_text_content_hash(text),
        "language": "tr",
        "contentType": "text/plain; charset=utf-8",
        "label": "synthetic",
        "labelSource": "test-fixture",
        "generatorModel": "fixture-generator-v1",
        "transformation": "none",
        "intendedUse": "Automated testing only",
        "split": "train",
    }


def write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_build_preprocessed_records_is_sorted_and_traceable() -> None:
    text_a = "\ufeffBirinci\r\nmetin"
    text_b = "İkinci metin"
    manifest = [manifest_record("b", text_b), manifest_record("a", text_a)]
    inputs = [{"id": "b", "text": text_b}, {"id": "a", "text": text_a}]

    records = build_preprocessed_records(manifest, inputs)

    assert [record["id"] for record in records] == ["a", "b"]
    assert records[0]["text"] == "Birinci\nmetin"
    assert records[0]["preprocessingVersion"] == "tr-text-v1"
    assert records[0]["inputContentHash"] == compute_text_content_hash(text_a)
    assert records[0]["outputContentHash"] == compute_text_content_hash("Birinci\nmetin")


def test_build_preprocessed_records_requires_verified_input_hashes() -> None:
    with pytest.raises(DatasetIntegrityError, match="content hash mismatch"):
        build_preprocessed_records(
            [manifest_record("sample-001", "Beklenen")],
            [{"id": "sample-001", "text": "Farklı"}],
        )


def test_write_private_jsonl_uses_owner_only_permissions(tmp_path: Path) -> None:
    output_path = tmp_path / "preprocessed.jsonl"
    records = [{"id": "sample-001", "text": "Sentetik test"}]

    write_private_jsonl(records, output_path)

    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert json.loads(output_path.read_text(encoding="utf-8")) == records[0]


def test_write_private_jsonl_refuses_overwrite_without_force(tmp_path: Path) -> None:
    output_path = tmp_path / "preprocessed.jsonl"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(OfflinePreprocessingError, match="already exists"):
        write_private_jsonl([], output_path)

    assert output_path.read_text(encoding="utf-8") == "existing"


def test_write_private_jsonl_requires_jsonl_extension(tmp_path: Path) -> None:
    with pytest.raises(OfflinePreprocessingError, match=".jsonl extension"):
        write_private_jsonl([], tmp_path / "preprocessed.json")


def test_ensure_distinct_artifact_paths_rejects_input_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"

    with pytest.raises(OfflinePreprocessingError, match="must differ"):
        ensure_distinct_artifact_paths(input_path, [input_path])


def test_cli_preprocesses_without_printing_raw_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_text = "\ufeffCLI-CIKTISINDA-GORUNMEMELI\r\n"
    manifest_path = write_json(
        tmp_path / "manifest.json",
        [manifest_record("sample-001", private_text)],
    )
    input_path = write_json(
        tmp_path / "input.json",
        [{"id": "sample-001", "text": private_text}],
    )
    output_path = tmp_path / "preprocessed.jsonl"

    exit_code = main(
        [
            "preprocess-text",
            str(manifest_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert artifact["text"] == "CLI-CIKTISINDA-GORUNMEMELI\n"
    assert private_text not in output
    assert "Wrote 1 preprocessed record" in output
