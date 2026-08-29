"""Exact training runtime locks for auditable experiment execution."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT_SCHEMA_PATH = EVALUATION_ROOT / "experiments" / "environment.schema.json"
DEFAULT_ENVIRONMENT_LOCK_PATH = EVALUATION_ROOT / "experiments" / "training-environment-v1.json"
DEFAULT_MAX_ENVIRONMENT_BYTES = 1024 * 1024


class TrainingEnvironmentError(ValueError):
    """Raised when the active runtime differs from the reviewed environment."""


def load_environment_lock(
    path: Path = DEFAULT_ENVIRONMENT_LOCK_PATH,
    *,
    schema_path: Path = DEFAULT_ENVIRONMENT_SCHEMA_PATH,
) -> dict[str, Any]:
    lock = _load_json_object(path, "environment lock")
    schema = _load_json_object(schema_path, "environment schema")
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(lock))
    if errors:
        locations = sorted(
            ".".join(str(part) for part in error.absolute_path) or "<document>"
            for error in errors
        )
        raise TrainingEnvironmentError(
            "Training environment validation failed at: " + ", ".join(locations)
        )
    return lock


def capture_training_environment() -> dict[str, Any]:
    """Capture only runtime fields that can invalidate result comparability."""
    try:
        import torch
    except ImportError as error:
        raise TrainingEnvironmentError("PyTorch is not installed") from error
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    packages: dict[str, str] = {}
    for package in (
        "accelerate",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError as error:
            raise TrainingEnvironmentError(
                f"Required training package is not installed: {package}"
            ) from error
    return {
        "pythonVersion": platform.python_version(),
        "platformTag": platform.platform(),
        "packages": packages,
        "device": device,
    }


def verify_training_environment(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed when Python, package versions, platform, or device drift."""
    active = capture_training_environment()
    mismatches: list[str] = []
    for field in ("pythonVersion", "platformTag"):
        if active[field] != lock[field]:
            mismatches.append(f"{field}: expected {lock[field]!r}, got {active[field]!r}")
    for package, expected in lock["packages"].items():
        actual = active["packages"].get(package)
        if actual != expected:
            mismatches.append(f"{package}: expected {expected!r}, got {actual!r}")
    expected_device = lock["execution"]["device"]
    if active["device"] != expected_device:
        mismatches.append(
            f"device: expected {expected_device!r}, got {active['device']!r}"
        )
    if mismatches:
        raise TrainingEnvironmentError(
            "Training environment differs from the reviewed lock:\n- "
            + "\n- ".join(mismatches)
        )
    return active


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrainingEnvironmentError(f"{label.title()} must be a regular file")
    if path.stat().st_size > DEFAULT_MAX_ENVIRONMENT_BYTES:
        raise TrainingEnvironmentError(f"{label.title()} exceeds the size limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingEnvironmentError(f"{label.title()} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise TrainingEnvironmentError(f"{label.title()} must be a JSON object")
    return document
