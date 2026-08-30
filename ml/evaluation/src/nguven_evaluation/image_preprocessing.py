/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Versioned, deterministic preprocessing and robustness variants for images."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator
from PIL import Image, ImageOps, UnidentifiedImageError

from nguven_evaluation.image_dataset_inputs import VerifiedImageInput


EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PREPROCESSED_SCHEMA_PATH = (
    EVALUATION_ROOT / "image" / "preprocessed" / "schema.json"
)
DEFAULT_IMAGE_PREPROCESSING_VERSION = "image-preprocessing-v1"
SUPPORTED_SOURCE_FORMATS = frozenset({"JPEG", "MPO", "PNG", "WEBP"})
MIN_IMAGE_DIMENSION = 32
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 40_000_000
PNG_COMPRESSION_LEVEL = 1
ROBUSTNESS_TRANSFORMATIONS = (
    "canonical",
    "jpeg-q90",
    "jpeg-q70",
    "resize-75",
    "center-crop-90",
    "screenshot-v1",
)


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be decoded or normalized safely."""


@dataclass(frozen=True, slots=True)
class ImageVariant:
    transformation: str
    data: bytes
    width: int
    height: int
    format: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def build_image_variants(path: Path) -> list[ImageVariant]:
    """Decode once and build deterministic canonical and robustness variants."""
    image = _load_normalized_rgb(path)
    variants = [
        _variant("canonical", image, output_format="png"),
        _variant("jpeg-q90", image, output_format="jpeg", quality=90),
        _variant("jpeg-q70", image, output_format="jpeg", quality=70),
    ]

    resized = image.resize(
        (
            max(MIN_IMAGE_DIMENSION, round(image.width * 0.75)),
            max(MIN_IMAGE_DIMENSION, round(image.height * 0.75)),
        ),
        Image.Resampling.LANCZOS,
    )
    variants.append(_variant("resize-75", resized, output_format="png"))

    crop_width = max(MIN_IMAGE_DIMENSION, round(image.width * 0.9))
    crop_height = max(MIN_IMAGE_DIMENSION, round(image.height * 0.9))
    left = max(0, (image.width - crop_width) // 2)
    top = max(0, (image.height - crop_height) // 2)
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    variants.append(_variant("center-crop-90", cropped, output_format="png"))

    screenshot = Image.new("RGB", image.size, color=(242, 242, 242))
    inner = image.resize(
        (
            max(MIN_IMAGE_DIMENSION, round(image.width * 0.9)),
            max(MIN_IMAGE_DIMENSION, round(image.height * 0.9)),
        ),
        Image.Resampling.LANCZOS,
    )
    screenshot.paste(
        inner,
        ((image.width - inner.width) // 2, (image.height - inner.height) // 2),
    )
    variants.append(_variant("screenshot-v1", screenshot, output_format="png"))
    return variants


def write_preprocessed_image_dataset(
    inputs: Iterable[VerifiedImageInput],
    output_root: Path,
    *,
    schema_path: Path = DEFAULT_IMAGE_PREPROCESSED_SCHEMA_PATH,
) -> list[dict[str, object]]:
    """Atomically write private image variants and their hash-bound JSONL manifest."""
    records = list(inputs)
    if not records:
        raise ImagePreprocessingError("Image preprocessing requires at least one input")
    record_ids = [item.record_id for item in records]
    if len(set(record_ids)) != len(record_ids):
        raise ImagePreprocessingError("Image preprocessing input ids must be unique")
    if output_root.exists() or output_root.is_symlink():
        raise ImagePreprocessingError("Image preprocessing output must not already exist")
    schema = _load_schema(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    os.chmod(temporary_root, 0o700)
    manifest: list[dict[str, object]] = []
    try:
        for item in sorted(records, key=lambda value: value.record_id):
            _verify_input_identity(item)
            record_root = temporary_root / "records" / item.record_id
            record_root.mkdir(parents=True, mode=0o700)
            for variant in build_image_variants(item.path):
                suffix = ".jpg" if variant.format == "jpeg" else ".png"
                relative = Path("records") / item.record_id / f"{variant.transformation}{suffix}"
                destination = temporary_root / relative
                destination.write_bytes(variant.data)
                document: dict[str, object] = {
                    "id": item.record_id,
                    "sourceSha256": item.sha256,
                    "preprocessingVersion": DEFAULT_IMAGE_PREPROCESSING_VERSION,
                    "transformation": variant.transformation,
                    "path": relative.as_posix(),
                    "sha256": variant.sha256,
                    "sizeBytes": len(variant.data),
                    "width": variant.width,
                    "height": variant.height,
                    "format": variant.format,
                }
                if list(validator.iter_errors(document)):
                    raise ImagePreprocessingError(
                        "Generated image variant violates the preprocessing schema"
                    )
                manifest.append(document)
        manifest_path = temporary_root / "manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest),
            encoding="utf-8",
        )
        os.replace(temporary_root, output_root)
        return manifest
    except (OSError, ImagePreprocessingError):
        raise
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _load_normalized_rgb(path: Path) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                source_format = str(source.format or "").upper()
                if source_format not in SUPPORTED_SOURCE_FORMATS:
                    raise ImagePreprocessingError(
                        f"Unsupported image format: {source_format or 'unknown'}"
                    )
                if source_format == "MPO" and int(getattr(source, "n_frames", 1)) != 2:
                    raise ImagePreprocessingError(
                        "MPO input must contain exactly two views"
                    )
                if source_format != "MPO" and bool(getattr(source, "is_animated", False)):
                    raise ImagePreprocessingError("Animated images are not supported")
                width, height = source.size
                _validate_dimensions(width, height)
                source.seek(0)
                source.load()
                transposed = ImageOps.exif_transpose(source)
                _validate_dimensions(*transposed.size)
                return _flatten_to_rgb(transposed)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as error:
        raise ImagePreprocessingError("Unable to decode image safely") from error


def _verify_input_identity(item: VerifiedImageInput) -> None:
    if item.path.is_symlink() or not item.path.is_file():
        raise ImagePreprocessingError(
            f"Verified image is no longer a regular file: {item.record_id}"
        )
    if item.path.stat().st_size != item.size_bytes:
        raise ImagePreprocessingError(
            f"Verified image size changed before preprocessing: {item.record_id}"
        )
    digest = hashlib.sha256()
    with item.path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != item.sha256:
        raise ImagePreprocessingError(
            f"Verified image hash changed before preprocessing: {item.record_id}"
        )


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, color=(255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _validate_dimensions(width: int, height: int) -> None:
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ImagePreprocessingError("Image dimensions are below the minimum")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ImagePreprocessingError("Image dimensions exceed the maximum")
    if width * height > MAX_IMAGE_PIXELS:
        raise ImagePreprocessingError("Image pixel count exceeds the safety limit")


def _variant(
    transformation: str,
    image: Image.Image,
    *,
    output_format: str,
    quality: int | None = None,
) -> ImageVariant:
    buffer = io.BytesIO()
    if output_format == "jpeg":
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=False,
            progressive=False,
            subsampling=2,
            exif=b"",
        )
    else:
        image.save(
            buffer,
            format="PNG",
            optimize=False,
            compress_level=PNG_COMPRESSION_LEVEL,
        )
    return ImageVariant(
        transformation=transformation,
        data=buffer.getvalue(),
        width=image.width,
        height=image.height,
        format=output_format,
    )


def _load_schema(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ImagePreprocessingError("Image preprocessing schema must be a regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImagePreprocessingError(
            "Image preprocessing schema must be valid UTF-8 JSON"
        ) from error
    if not isinstance(document, dict):
        raise ImagePreprocessingError(
            "Image preprocessing schema must be a JSON object"
        )
    return document
