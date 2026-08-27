from __future__ import annotations

import json
from pathlib import Path

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.dataset_inputs import DatasetInputError, load_dataset_input


def write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_dataset_input_accepts_json_array(tmp_path: Path) -> None:
    records = [
        {"id": "sample-001", "text": "Türkçe sentetik test metni."},
        {"id": "sample-002", "text": "Yalnızca test kapsamında kullanılır."},
    ]
    input_path = write_json(tmp_path / "input.json", records)

    assert load_dataset_input(input_path) == records


def test_load_dataset_input_accepts_json_lines(tmp_path: Path) -> None:
    records = [
        {"id": "sample-001", "text": "Birinci test kaydı."},
        {"id": "sample-002", "text": "İkinci test kaydı."},
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    assert load_dataset_input(input_path) == records


def test_load_dataset_input_rejects_duplicate_ids(tmp_path: Path) -> None:
    input_path = write_json(
        tmp_path / "input.json",
        [
            {"id": "duplicate", "text": "Birinci metin."},
            {"id": "duplicate", "text": "İkinci metin."},
        ],
    )

    with pytest.raises(DatasetInputError, match="duplicate id"):
        load_dataset_input(input_path)


def test_validation_error_never_exposes_text_content(tmp_path: Path) -> None:
    sensitive_text = "BU-HAM-METIN-HATA-MESAJINA-GIRMEMELI"
    input_path = write_json(
        tmp_path / "input.json",
        [{"id": "sample-001", "text": [sensitive_text]}],
    )

    with pytest.raises(DatasetInputError) as captured:
        load_dataset_input(input_path)

    assert sensitive_text not in str(captured.value)
    assert "record 1, text: has an invalid type" in str(captured.value)


def test_load_dataset_input_rejects_symbolic_links(tmp_path: Path) -> None:
    target = write_json(
        tmp_path / "actual.json",
        [{"id": "sample-001", "text": "Test metni."}],
    )
    link = tmp_path / "linked.json"
    link.symlink_to(target)

    with pytest.raises(DatasetInputError, match="must not be a symbolic link"):
        load_dataset_input(link)


def test_load_dataset_input_enforces_file_size_limit(tmp_path: Path) -> None:
    input_path = write_json(
        tmp_path / "input.json",
        [{"id": "sample-001", "text": "Boyut sınırı testi."}],
    )

    with pytest.raises(DatasetInputError, match="safety limit"):
        load_dataset_input(input_path, max_file_bytes=8)


def test_cli_reports_only_count_and_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    private_text = "CLI-CIKTISINDA-GORUNMEMELI"
    input_path = write_json(
        tmp_path / "input.json",
        [{"id": "sample-001", "text": private_text}],
    )

    exit_code = main(["validate-input", str(input_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Validated 1 local input record" in output
    assert private_text not in output
