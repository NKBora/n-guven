from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nguven_evaluation.cli import main
from nguven_evaluation.image_dataset_inputs import (
    ImageDatasetInputError,
    load_image_dataset_inputs,
)


def _write_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return path


def _record(
    path: str,
    content: bytes,
    *,
    record_id: str = "image-001",
) -> dict[str, object]:
    return {
        "id": record_id,
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "sizeBytes": len(content),
    }


def test_loads_hash_verified_private_images(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    content = b"fixture-image-bytes"
    (image_root / "sample.jpg").write_bytes(content)
    manifest = _write_manifest(
        tmp_path / "input.json",
        [_record("sample.jpg", content)],
    )

    verified = load_image_dataset_inputs(manifest, image_root=image_root)

    assert len(verified) == 1
    assert verified[0].record_id == "image-001"
    assert verified[0].path == (image_root / "sample.jpg").resolve()
    assert verified[0].sha256 == hashlib.sha256(content).hexdigest()


def test_loads_json_lines_manifest(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    content = b"json-lines-image"
    (image_root / "sample.webp").write_bytes(content)
    record = _record("sample.webp", content)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    verified = load_image_dataset_inputs(manifest, image_root=image_root)

    assert [item.record_id for item in verified] == ["image-001"]


def test_rejects_tampered_image_bytes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    original = b"original-image"
    image = image_root / "sample.png"
    image.write_bytes(original)
    manifest = _write_manifest(
        tmp_path / "input.json",
        [_record("sample.png", original)],
    )
    image.write_bytes(b"tampered-image")

    with pytest.raises(ImageDatasetInputError, match="size mismatch|SHA-256 mismatch"):
        load_image_dataset_inputs(manifest, image_root=image_root)


def test_rejects_path_traversal_without_exposing_external_content(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    sensitive = b"PRIVATE-IMAGE-CONTENT-MUST-NOT-APPEAR"
    (tmp_path / "outside.jpg").write_bytes(sensitive)
    manifest = _write_manifest(
        tmp_path / "input.json",
        [_record("../outside.jpg", sensitive)],
    )

    with pytest.raises(ImageDatasetInputError) as captured:
        load_image_dataset_inputs(manifest, image_root=image_root)

    assert sensitive.decode() not in str(captured.value)


def test_rejects_symbolic_linked_image(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    content = b"outside-image"
    target = tmp_path / "outside.jpg"
    target.write_bytes(content)
    link = image_root / "linked.jpg"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symbolic links are unavailable on this platform")
    manifest = _write_manifest(
        tmp_path / "input.json",
        [_record("linked.jpg", content)],
    )

    with pytest.raises(ImageDatasetInputError, match="symbolic link"):
        load_image_dataset_inputs(manifest, image_root=image_root)


def test_rejects_duplicate_paths(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    content = b"same-image"
    (image_root / "sample.jpg").write_bytes(content)
    manifest = _write_manifest(
        tmp_path / "input.json",
        [
            _record("sample.jpg", content, record_id="image-001"),
            _record("sample.jpg", content, record_id="image-002"),
        ],
    )

    with pytest.raises(ImageDatasetInputError, match="duplicate path"):
        load_image_dataset_inputs(manifest, image_root=image_root)


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    first = b"first"
    second = b"second"
    (image_root / "first.jpg").write_bytes(first)
    (image_root / "second.jpg").write_bytes(second)
    manifest = _write_manifest(
        tmp_path / "input.json",
        [
            _record("first.jpg", first, record_id="same-id"),
            _record("second.jpg", second, record_id="same-id"),
        ],
    )

    with pytest.raises(ImageDatasetInputError, match="duplicate id"):
        load_image_dataset_inputs(manifest, image_root=image_root)


def test_enforces_per_file_and_total_byte_limits(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    first = b"first-image"
    second = b"second-image"
    (image_root / "first.jpg").write_bytes(first)
    (image_root / "second.jpg").write_bytes(second)
    manifest = _write_manifest(
        tmp_path / "input.json",
        [
            _record("first.jpg", first, record_id="image-001"),
            _record("second.jpg", second, record_id="image-002"),
        ],
    )

    with pytest.raises(ImageDatasetInputError, match="per-file safety limit"):
        load_image_dataset_inputs(
            manifest,
            image_root=image_root,
            max_image_bytes=len(first) - 1,
        )
    with pytest.raises(ImageDatasetInputError, match="total safety limit"):
        load_image_dataset_inputs(
            manifest,
            image_root=image_root,
            max_total_bytes=len(first) + len(second) - 1,
        )


def test_cli_reports_count_without_reading_image_content_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    private_content = b"PRIVATE-IMAGE-BYTES-MUST-NOT-APPEAR"
    (image_root / "sample.jpg").write_bytes(private_content)
    manifest = _write_manifest(
        tmp_path / "input.json",
        [_record("sample.jpg", private_content)],
    )

    exit_code = main(
        [
            "validate-image-input",
            str(manifest),
            "--image-root",
            str(image_root),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Verified 1 private image input" in output
    assert private_content.decode() not in output
