from __future__ import annotations

import json
from pathlib import Path

import pytest

from nguven_evaluation.calibration import (
    CalibrationError,
    fit_temperature_calibration,
    load_calibration_artifact,
    predicted_synthetic_probability,
    temperature_scale,
)


def validation_data() -> tuple[list[dict], list[dict]]:
    manifest: list[dict] = []
    predictions: list[dict] = []
    for index in range(20):
        label = "human" if index < 10 else "synthetic"
        predicted = label if index not in {1, 3, 12, 14} else (
            "synthetic" if label == "human" else "human"
        )
        manifest.append({"id": f"r-{index}", "label": label, "split": "validation"})
        predictions.append(
            {
                "id": f"r-{index}",
                "predictedLabel": predicted,
                "score": 0.95,
                "inferenceMs": 4.0,
            }
        )
    return manifest, predictions


def test_temperature_fit_is_validation_only_and_never_worsens_nll() -> None:
    manifest, predictions = validation_data()

    artifact = fit_temperature_calibration(
        manifest,
        predictions,
        model_name="berturk",
        model_version="berturk-text-origin-v1",
        manifest_sha256="a" * 64,
        predictions_sha256="b" * 64,
    )

    assert artifact["fittedSplit"] == "validation"
    assert artifact["temperature"] > 1
    assert artifact["metrics"]["negativeLogLikelihoodAfter"] <= artifact["metrics"]["negativeLogLikelihoodBefore"]


def test_temperature_fit_rejects_test_records() -> None:
    manifest, predictions = validation_data()
    manifest[0]["split"] = "test"

    with pytest.raises(CalibrationError, match="only the validation split"):
        fit_temperature_calibration(
            manifest,
            predictions,
            model_name="berturk",
            model_version="v1",
            manifest_sha256="a" * 64,
            predictions_sha256="b" * 64,
        )


def test_binary_probability_uses_predicted_label_confidence() -> None:
    assert predicted_synthetic_probability(
        {"predictedLabel": "synthetic", "score": 0.9}
    ) == pytest.approx(0.9)
    assert predicted_synthetic_probability(
        {"predictedLabel": "human", "score": 0.9}
    ) == pytest.approx(0.1)
    assert temperature_scale(0.5, 2.0) == pytest.approx(0.5)


def test_calibration_loader_rejects_symbolic_link(tmp_path: Path) -> None:
    manifest, predictions = validation_data()
    artifact = fit_temperature_calibration(
        manifest,
        predictions,
        model_name="berturk",
        model_version="v1",
        manifest_sha256="a" * 64,
        predictions_sha256="b" * 64,
    )
    target = tmp_path / "calibration.json"
    target.write_text(json.dumps(artifact), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(CalibrationError, match="regular file"):
        load_calibration_artifact(link)
