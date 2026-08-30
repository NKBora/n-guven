from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from nguven_evaluation.image_model_adapters import (
    CandidateImageModelAdapter,
    ImageModelAdapterError,
    ImagePrediction,
    load_image_candidate_registry,
)


class FakeBackend:
    def __init__(self, prediction: ImagePrediction) -> None:
        self.prediction = prediction
        self.inputs: list[Image.Image] = []

    def predict(self, image: Image.Image) -> ImagePrediction:
        self.inputs.append(image)
        return self.prediction


def test_registry_pins_complete_reviewed_candidate_pair() -> None:
    candidates = load_image_candidate_registry()

    assert [candidate.adapter_id for candidate in candidates] == [
        "siglip2-aiornot",
        "vit-cifake",
    ]
    assert all(len(candidate.revision) == 40 for candidate in candidates)
    assert all(candidate.weights_license == "Apache-2.0" for candidate in candidates)
    assert all(candidate.normalized_labels == {0: "human", 1: "synthetic"} for candidate in candidates)
    assert all(
        {path for path, _, _ in candidate.artifacts}
        == {"config.json", "preprocessor_config.json", "model.safetensors"}
        for candidate in candidates
    )


@pytest.mark.parametrize("candidate_index", [0, 1])
def test_candidates_share_normalized_prediction_contract(candidate_index: int) -> None:
    candidate = load_image_candidate_registry()[candidate_index]
    backend = FakeBackend(ImagePrediction("synthetic", 0.91))
    adapter = CandidateImageModelAdapter(candidate, backend)
    image = Image.new("RGB", (224, 224), "white")

    assert adapter.predict(image) == ImagePrediction("synthetic", 0.91)
    assert backend.inputs == [image]
    assert adapter.preprocessing_version == "image-preprocessing-v1"


def test_registry_rejects_changed_revision(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "image" / "models" / "candidates.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["candidates"][0]["upstream"]["revision"] = "0" * 40
    changed = tmp_path / "candidates.json"
    changed.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ImageModelAdapterError, match="differs from reviewed pin"):
        load_image_candidate_registry(changed)


def test_registry_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    changed = tmp_path / "candidates.json"
    changed.write_text('{"schemaVersion":"one","schemaVersion":"two"}', encoding="utf-8")

    with pytest.raises(ImageModelAdapterError, match="Duplicate JSON key"):
        load_image_candidate_registry(changed)


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (ImagePrediction("unknown", 0.8), "outside the registry"),
        (ImagePrediction("human", float("nan")), "finite"),
        (ImagePrediction("human", 1.01), "within"),
    ],
)
def test_adapter_rejects_invalid_predictions(
    prediction: ImagePrediction,
    message: str,
) -> None:
    candidate = load_image_candidate_registry()[0]
    adapter = CandidateImageModelAdapter(candidate, FakeBackend(prediction))

    with pytest.raises(ImageModelAdapterError, match=message):
        adapter.predict(Image.new("RGB", (224, 224)))


def test_adapter_requires_normalized_rgb_input() -> None:
    candidate = load_image_candidate_registry()[0]
    adapter = CandidateImageModelAdapter(
        candidate,
        FakeBackend(ImagePrediction("human", 0.9)),
    )

    with pytest.raises(ImageModelAdapterError, match="normalized RGB"):
        adapter.predict(Image.new("RGBA", (224, 224)))
