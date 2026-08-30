/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Verified materialization and local-only Transformers image inference."""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from PIL import Image

from nguven_evaluation.image_model_adapters import (
    ImageCandidateDescriptor,
    ImageModelAdapterError,
    ImagePrediction,
)


DownloadFunction = Callable[[str, str, str, Path | None, bool], str]


def verify_image_candidate_artifacts(
    candidate: ImageCandidateDescriptor,
    artifact_root: Path,
) -> None:
    """Require the exact allowlisted files, sizes, and hashes for one candidate."""
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ImageModelAdapterError("Image model artifact root must be a regular directory")
    expected = {path: (size, digest) for path, size, digest in candidate.artifacts}
    actual_paths = {item.name for item in artifact_root.iterdir()}
    if actual_paths != set(expected):
        raise ImageModelAdapterError(
            "Image model artifact directory differs from the reviewed allowlist"
        )
    for name, (expected_size, expected_hash) in expected.items():
        path = artifact_root / name
        if path.is_symlink() or not path.is_file():
            raise ImageModelAdapterError(f"Image model artifact must be a regular file: {name}")
        if path.stat().st_size != expected_size:
            raise ImageModelAdapterError(f"Image model artifact size mismatch: {name}")
        if not hmac.compare_digest(_sha256_file(path), expected_hash):
            raise ImageModelAdapterError(f"Image model artifact hash mismatch: {name}")


def materialize_image_candidate(
    candidate: ImageCandidateDescriptor,
    output_root: Path,
    *,
    allow_network: bool = False,
    cache_dir: Path | None = None,
    downloader: DownloadFunction | None = None,
) -> None:
    """Download a pinned allowlist into an owner-only, verified atomic directory."""
    if output_root.exists() or output_root.is_symlink():
        raise ImageModelAdapterError("Image model output must not already exist")
    if cache_dir is not None and cache_dir.is_symlink():
        raise ImageModelAdapterError("Image model cache directory must not be a symbolic link")
    if downloader is None:
        downloader = _hugging_face_download

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    os.chmod(temporary_root, 0o700)
    try:
        for filename, _, _ in candidate.artifacts:
            try:
                downloaded = Path(
                    downloader(
                        candidate.repository,
                        filename,
                        candidate.revision,
                        cache_dir,
                        not allow_network,
                    )
                )
            except Exception as error:
                raise ImageModelAdapterError(
                    f"Unable to materialize reviewed image artifact: {filename}"
                ) from error
            if downloaded.is_symlink():
                downloaded = downloaded.resolve(strict=True)
            if not downloaded.is_file():
                raise ImageModelAdapterError(
                    f"Downloaded image artifact is not a regular file: {filename}"
                )
            destination = temporary_root / filename
            with downloaded.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            os.chmod(destination, 0o600)
        verify_image_candidate_artifacts(candidate, temporary_root)
        os.replace(temporary_root, output_root)
    except (OSError, ImageModelAdapterError):
        raise
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


class LocalTransformersImageBackend:
    """Load a hash-verified image classifier without network or remote code."""

    def __init__(
        self,
        candidate: ImageCandidateDescriptor,
        artifact_root: Path,
        *,
        device: str = "cpu",
    ) -> None:
        verify_image_candidate_artifacts(candidate, artifact_root)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as error:
            raise ImageModelAdapterError(
                'Local image inference requires: pip install -e ".[inference]"'
            ) from error
        if device not in {"cpu", "mps", "cuda"}:
            raise ImageModelAdapterError("Image inference device must be cpu, mps, or cuda")
        self._torch = torch
        self._candidate = candidate
        self._device = device
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                str(artifact_root),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=False,
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                str(artifact_root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            self._model.to(device)
            self._model.eval()
        except (OSError, RuntimeError, ValueError) as error:
            raise ImageModelAdapterError(
                "Unable to load the verified local image model artifact"
            ) from error

        architectures = tuple(getattr(self._model.config, "architectures", ()) or ())
        if architectures != (candidate.architecture,):
            raise ImageModelAdapterError("Loaded image model architecture differs from registry")
        configured_labels = {
            int(index): str(label)
            for index, label in dict(getattr(self._model.config, "id2label", {})).items()
        }
        if configured_labels != dict(candidate.upstream_labels):
            raise ImageModelAdapterError("Loaded image model labels differ from registry")

    def predict(self, image: Image.Image) -> ImagePrediction:
        try:
            encoded = self._processor(images=image, return_tensors="pt")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits[0]
                probabilities = self._torch.softmax(logits, dim=-1)
                score, label_index = self._torch.max(probabilities, dim=-1)
            index = int(label_index.item())
            return ImagePrediction(
                label=self._candidate.normalized_labels[index],
                score=float(score.item()),
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ImageModelAdapterError("Local image Transformers inference failed") from error


def _hugging_face_download(
    repository: str,
    filename: str,
    revision: str,
    cache_dir: Path | None,
    local_files_only: bool,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImageModelAdapterError(
            'Image model materialization requires: pip install -e ".[inference]"'
        ) from error
    return hf_hub_download(
        repo_id=repository,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
