/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from nguven_evaluation.image_model_adapters import (
    ImageModelAdapterError,
    load_image_candidate_registry,
)
from nguven_evaluation.image_transformers_backend import (
    LocalTransformersImageBackend,
    materialize_image_candidate,
    verify_image_candidate_artifacts,
)


def _small_candidate(files: dict[str, bytes]):
    candidate = load_image_candidate_registry()[0]
    return replace(
        candidate,
        artifacts=tuple(
            (name, len(content), hashlib.sha256(content).hexdigest())
            for name, content in files.items()
        ),
    )


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir()
    for name, content in files.items():
        (root / name).write_bytes(content)


def test_materialization_is_pinned_verified_and_owner_only(tmp_path: Path) -> None:
    files = {
        "config.json": b"config",
        "preprocessor_config.json": b"processor",
        "model.safetensors": b"safe-weights",
    }
    candidate = _small_candidate(files)
    source = tmp_path / "source"
    _write_files(source, files)
    calls: list[tuple[str, str, str, Path | None, bool]] = []

    def downloader(repository, filename, revision, cache_dir, local_files_only):
        calls.append((repository, filename, revision, cache_dir, local_files_only))
        return str(source / filename)

    output = tmp_path / "artifacts"
    materialize_image_candidate(candidate, output, downloader=downloader)

    verify_image_candidate_artifacts(candidate, output)
    assert {item.name for item in output.iterdir()} == set(files)
    assert all(call[2] == candidate.revision for call in calls)
    assert all(call[4] is True for call in calls)
    assert all((output / name).stat().st_mode & 0o777 == 0o600 for name in files)


def test_materialization_network_is_explicit_opt_in(tmp_path: Path) -> None:
    files = {
        "config.json": b"config",
        "preprocessor_config.json": b"processor",
        "model.safetensors": b"safe-weights",
    }
    candidate = _small_candidate(files)
    source = tmp_path / "source"
    _write_files(source, files)
    flags: list[bool] = []

    def downloader(repository, filename, revision, cache_dir, local_files_only):
        flags.append(local_files_only)
        return str(source / filename)

    materialize_image_candidate(
        candidate,
        tmp_path / "artifacts",
        allow_network=True,
        downloader=downloader,
    )

    assert flags == [False, False, False]


def test_verification_rejects_hash_mismatch_and_unexpected_files(tmp_path: Path) -> None:
    files = {
        "config.json": b"config",
        "preprocessor_config.json": b"processor",
        "model.safetensors": b"safe-weights",
    }
    candidate = _small_candidate(files)
    root = tmp_path / "artifacts"
    _write_files(root, files)
    (root / "model.safetensors").write_bytes(b"unsafe-data!")

    with pytest.raises(ImageModelAdapterError, match="hash mismatch"):
        verify_image_candidate_artifacts(candidate, root)

    (root / "model.safetensors").write_bytes(files["model.safetensors"])
    (root / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ImageModelAdapterError, match="allowlist"):
        verify_image_candidate_artifacts(candidate, root)


def test_backend_loads_only_local_safetensors_without_remote_code(
    tmp_path: Path,
    monkeypatch,
) -> None:
    files = {
        "config.json": b"config",
        "preprocessor_config.json": b"processor",
        "model.safetensors": b"safe-weights",
    }
    candidate = _small_candidate(files)
    root = tmp_path / "artifacts"
    _write_files(root, files)
    calls: dict[str, dict[str, object]] = {}

    class AutoImageProcessor:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls["processor"] = {"path": path, **kwargs}
            return object()

    class Model:
        config = SimpleNamespace(
            architectures=[candidate.architecture],
            id2label=dict(candidate.upstream_labels),
        )

        def to(self, device: str) -> None:
            calls["device"] = {"value": device}

        def eval(self) -> None:
            calls["eval"] = {}

    class AutoModelForImageClassification:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls["model"] = {"path": path, **kwargs}
            return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=AutoImageProcessor,
            AutoModelForImageClassification=AutoModelForImageClassification,
        ),
    )

    LocalTransformersImageBackend(candidate, root)

    assert calls["processor"] == {
        "path": str(root),
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": False,
    }
    assert calls["model"] == {
        "path": str(root),
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert calls["device"] == {"value": "cpu"}
    assert "eval" in calls
