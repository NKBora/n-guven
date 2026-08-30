/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from nguven_evaluation.cli import main
from nguven_evaluation.image_dataset_inputs import VerifiedImageInput
from nguven_evaluation.image_preprocessing import (
    DEFAULT_IMAGE_PREPROCESSING_VERSION,
    ImagePreprocessingError,
    build_image_variants,
    write_preprocessed_image_dataset,
    ROBUSTNESS_TRANSFORMATIONS,
)


def _create_image(
    path: Path,
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (80, 64),
    color: object = (12, 120, 220),
    format: str = "PNG",
) -> bytes:
    image = Image.new(mode, size, color=color)
    image.save(path, format=format)
    return path.read_bytes()


def _verified(path: Path, *, record_id: str = "image-001") -> VerifiedImageInput:
    data = path.read_bytes()
    return VerifiedImageInput(
        record_id=record_id,
        path=path.resolve(),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def test_builds_deterministic_robustness_variants(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _create_image(source)

    first = build_image_variants(source)
    second = build_image_variants(source)

    assert [item.transformation for item in first] == [
        "canonical",
        "jpeg-q90",
        "jpeg-q70",
        "resize-75",
        "center-crop-90",
        "screenshot-v1",
    ]
    assert [item.sha256 for item in first] == [item.sha256 for item in second]
    assert {item.format for item in first} == {"png", "jpeg"}


def test_flattens_transparency_on_white_and_strips_metadata(tmp_path: Path) -> None:
    source = tmp_path / "transparent.png"
    _create_image(source, mode="RGBA", color=(255, 0, 0, 0))

    canonical = build_image_variants(source)[0]
    output = tmp_path / "canonical.png"
    output.write_bytes(canonical.data)
    with Image.open(output) as image:
        image.load()
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.info == {}


@pytest.mark.parametrize("size", [(31, 64), (64, 31)])
def test_rejects_images_below_minimum_dimensions(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    source = tmp_path / "small.png"
    _create_image(source, size=size)

    with pytest.raises(ImagePreprocessingError, match="below the minimum"):
        build_image_variants(source)


def test_rejects_unsupported_or_corrupt_images(tmp_path: Path) -> None:
    source = tmp_path / "not-an-image.jpg"
    source.write_bytes(b"not an image")

    with pytest.raises(ImagePreprocessingError, match="decode image safely"):
        build_image_variants(source)


def test_atomically_writes_hash_bound_private_dataset(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    _create_image(first)
    _create_image(second, format="JPEG", color=(220, 40, 30))
    output = tmp_path / "preprocessed"

    manifest = write_preprocessed_image_dataset(
        [_verified(second, record_id="image-002"), _verified(first)],
        output,
    )

    assert len(manifest) == 12
    assert manifest[0]["id"] == "image-001"
    assert {item["preprocessingVersion"] for item in manifest} == {
        DEFAULT_IMAGE_PREPROCESSING_VERSION
    }
    for item in manifest:
        artifact = output / str(item["path"])
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == item["sha256"]
    lines = (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    assert all(json.loads(line)["sourceSha256"] for line in lines)


def test_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _create_image(source)
    output = tmp_path / "preprocessed"
    output.mkdir()

    with pytest.raises(ImagePreprocessingError, match="must not already exist"):
        write_preprocessed_image_dataset([_verified(source)], output)


def test_rechecks_image_identity_immediately_before_preprocessing(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _create_image(source)
    verified = _verified(source)
    source.write_bytes(source.read_bytes() + b"tampered")

    with pytest.raises(ImagePreprocessingError, match="size changed"):
        write_preprocessed_image_dataset([verified], tmp_path / "output")


def test_rejects_duplicate_input_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _create_image(first)
    _create_image(second, color=(1, 2, 3))

    with pytest.raises(ImagePreprocessingError, match="ids must be unique"):
        write_preprocessed_image_dataset(
            [_verified(first), _verified(second)],
            tmp_path / "output",
        )


def test_preprocesses_primary_view_of_two_view_mpo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.mpo"
    output = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(
        output,
        format="MPO",
        save_all=True,
        append_images=[Image.new("RGB", (64, 64), "black")],
    )
    path.write_bytes(output.getvalue())

    records = write_preprocessed_image_dataset(
        [_verified(path, record_id="mpo-1")],
        tmp_path / "processed",
    )

    assert len(records) == len(ROBUSTNESS_TRANSFORMATIONS)


def test_cli_verifies_then_preprocesses_images(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    source = image_root / "source.png"
    data = _create_image(source)
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(
        json.dumps(
            [
                {
                    "id": "image-001",
                    "path": "source.png",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "sizeBytes": len(data),
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "preprocessed"

    exit_code = main(
        [
            "preprocess-image",
            str(input_manifest),
            "--image-root",
            str(image_root),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert "Wrote 6 image variant(s)" in capsys.readouterr().out
    assert (output / "manifest.jsonl").is_file()
