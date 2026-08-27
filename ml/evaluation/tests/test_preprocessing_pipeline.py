from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.integrity import compute_text_content_hash
from nguven_evaluation.offline_preprocessing import (
    OfflinePreprocessingError,
    build_preprocessed_records,
    write_private_jsonl,
)


def manifest_record(record_id: str, text: str) -> dict[str, object]:
    return {
        "id": record_id,
        "source": "synthetic-pipeline-fixture",
        "sourceUrl": f"https://example.invalid/pipeline/{record_id}",
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


def run_preprocessing(
    manifest_path: Path,
    input_path: Path,
    output_path: Path,
    *extra_arguments: str,
) -> int:
    return main(
        [
            "preprocess-text",
            str(manifest_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            *extra_arguments,
        ]
    )


def test_integrity_hashes_raw_text_before_unicode_normalization() -> None:
    decomposed = "Tu\u0308rkc\u0327e"
    composed = "Türkçe"

    decomposed_output = build_preprocessed_records(
        [manifest_record("sample", decomposed)],
        [{"id": "sample", "text": decomposed}],
    )[0]
    composed_output = build_preprocessed_records(
        [manifest_record("sample", composed)],
        [{"id": "sample", "text": composed}],
    )[0]

    assert decomposed_output["inputContentHash"] != composed_output["inputContentHash"]
    assert decomposed_output["text"] == composed_output["text"]
    assert decomposed_output["outputContentHash"] == composed_output["outputContentHash"]


def test_cli_output_is_deterministic_across_record_order(tmp_path: Path) -> None:
    records = [("b", "İkinci\r\nmetin"), ("a", "\ufeffBirinci metin")]
    manifest_path = write_json(
        tmp_path / "manifest.json",
        [manifest_record(record_id, text) for record_id, text in records],
    )
    first_input = write_json(
        tmp_path / "first-input.json",
        [{"id": record_id, "text": text} for record_id, text in records],
    )
    second_input = write_json(
        tmp_path / "second-input.json",
        [{"id": record_id, "text": text} for record_id, text in reversed(records)],
    )
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    assert run_preprocessing(manifest_path, first_input, first_output) == 0
    assert run_preprocessing(manifest_path, second_input, second_output) == 0

    assert first_output.read_bytes() == second_output.read_bytes()


def test_cli_hash_mismatch_writes_no_artifact_and_redacts_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_text = "BU-HAM-METIN-CIKTIDA-GORUNMEMELI"
    manifest_path = write_json(
        tmp_path / "manifest.json",
        [manifest_record("sample-001", "Beklenen metin")],
    )
    input_path = write_json(
        tmp_path / "input.json",
        [{"id": "sample-001", "text": private_text}],
    )
    output_path = tmp_path / "output.jsonl"

    exit_code = run_preprocessing(manifest_path, input_path, output_path)

    assert exit_code == 1
    assert not output_path.exists()
    assert private_text not in capsys.readouterr().out


def test_cli_refuses_to_replace_input_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = "Kaynak dosya korunmalı."
    manifest_path = write_json(
        tmp_path / "manifest.json",
        [manifest_record("sample-001", text)],
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"id": "sample-001", "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    original_bytes = input_path.read_bytes()

    exit_code = run_preprocessing(
        manifest_path,
        input_path,
        input_path,
        "--force",
    )

    assert exit_code == 1
    assert input_path.read_bytes() == original_bytes
    assert "must differ from input artifacts" in capsys.readouterr().out


def test_force_replacement_restores_owner_only_permissions(tmp_path: Path) -> None:
    output_path = tmp_path / "output.jsonl"
    output_path.write_text("old", encoding="utf-8")
    output_path.chmod(0o644)

    write_private_jsonl([{"id": "sample", "text": "new"}], output_path, force=True)

    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert "new" in output_path.read_text(encoding="utf-8")


def test_writer_refuses_symlink_output_without_touching_target(tmp_path: Path) -> None:
    target_path = tmp_path / "target.jsonl"
    target_path.write_text("protected", encoding="utf-8")
    output_link = tmp_path / "output.jsonl"
    output_link.symlink_to(target_path)

    with pytest.raises(OfflinePreprocessingError, match="symbolic link"):
        write_private_jsonl([{"id": "sample", "text": "replacement"}], output_link, force=True)

    assert target_path.read_text(encoding="utf-8") == "protected"


def test_cli_rejects_invalid_utf8_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = write_json(
        tmp_path / "manifest.json",
        [manifest_record("sample-001", "Geçerli metin")],
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_bytes(b'{"id":"sample-001","text":"invalid-\xff"}\n')
    output_path = tmp_path / "output.jsonl"

    exit_code = run_preprocessing(manifest_path, input_path, output_path)

    assert exit_code == 1
    assert not output_path.exists()
    assert "must be valid UTF-8" in capsys.readouterr().out
