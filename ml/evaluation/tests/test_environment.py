from __future__ import annotations

import pytest

from nguven_evaluation.environment import (
    TrainingEnvironmentError,
    load_environment_lock,
    verify_training_environment,
)


def test_repository_environment_lock_is_valid() -> None:
    lock = load_environment_lock()

    assert lock["environmentId"] == "m2-pro-mps-v1"
    assert lock["execution"]["statisticalSeeds"] == [17, 42, 71]


def test_environment_verification_rejects_version_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = load_environment_lock()
    active = {
        "pythonVersion": lock["pythonVersion"],
        "platformTag": lock["platformTag"],
        "packages": dict(lock["packages"]),
        "device": lock["execution"]["device"],
    }
    active["packages"]["torch"] = "0.0.0"
    monkeypatch.setattr(
        "nguven_evaluation.environment.capture_training_environment",
        lambda: active,
    )

    with pytest.raises(TrainingEnvironmentError, match="torch"):
        verify_training_environment(lock)
