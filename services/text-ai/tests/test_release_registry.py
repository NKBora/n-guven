import json
from pathlib import Path

from jsonschema import Draft202012Validator


SERVICE_ROOT = Path(__file__).resolve().parents[1]
RELEASES_ROOT = SERVICE_ROOT / "releases"


def test_berturk_release_matches_strict_schema() -> None:
    schema = json.loads((RELEASES_ROOT / "schema.json").read_text(encoding="utf-8"))
    release = json.loads(
        (RELEASES_ROOT / "berturk-text-origin-v1.json").read_text(encoding="utf-8")
    )

    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(release))

    assert errors == []


def test_release_is_validation_selected_and_hash_pinned() -> None:
    release = json.loads(
        (RELEASES_ROOT / "berturk-text-origin-v1.json").read_text(encoding="utf-8")
    )

    assert release["model"]["adapterId"] == "berturk"
    assert release["model"]["seed"] == 17
    assert release["selection"]["validationMacroF1"] == 1.0
    assert release["selection"]["testSplitUsed"] is False
    assert release["calibration"]["fittedSplit"] == "validation"
    assert release["thresholds"] == {
        "version": "text-origin-thresholds-v1",
        "lowMaximum": 0.2,
        "highMinimum": 0.8,
    }
