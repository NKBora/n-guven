from __future__ import annotations

from typing import Any

import pytest

from nguven_evaluation.model_adapters import (
    BERTURK,
    MODERNBERT_TR,
    AdapterPrediction,
    CandidateTextModelAdapter,
    ModelAdapterError,
)


class FakeBackend:
    def __init__(self, prediction: AdapterPrediction) -> None:
        self.prediction = prediction
        self.inputs: list[str] = []

    def predict(self, text: str) -> AdapterPrediction:
        self.inputs.append(text)
        return self.prediction


def manifest(adapter_id: str = "berturk") -> dict[str, Any]:
    candidate = BERTURK if adapter_id == "berturk" else MODERNBERT_TR
    return {
        "modelId": f"synthetic-{adapter_id}-v1",
        "adapterId": adapter_id,
        "preprocessingVersion": "tr-text-v1",
        "upstream": {
            "repository": candidate.repository,
            "revision": candidate.revision,
        },
        "labels": {"0": "human", "1": "synthetic"},
    }


@pytest.mark.parametrize("adapter_id", ["berturk", "modernbert-tr"])
def test_candidate_adapters_share_one_prediction_contract(adapter_id: str) -> None:
    backend = FakeBackend(AdapterPrediction(label="synthetic", score=0.91))
    adapter = CandidateTextModelAdapter(manifest(adapter_id), backend)

    assert adapter.adapter_id == adapter_id
    assert adapter.preprocessing_version == "tr-text-v1"
    assert adapter.predict("Türkçe örnek") == AdapterPrediction("synthetic", 0.91)
    assert backend.inputs == ["Türkçe örnek"]


def test_adapter_rejects_unreviewed_upstream_revision() -> None:
    data = manifest()
    data["upstream"]["revision"] = "0" * 40

    with pytest.raises(ModelAdapterError, match="revision"):
        CandidateTextModelAdapter(data, FakeBackend(AdapterPrediction("human", 0.8)))


def test_adapter_rejects_wrong_preprocessing_version() -> None:
    data = manifest()
    data["preprocessingVersion"] = "tr-text-v2"

    with pytest.raises(ModelAdapterError, match="tr-text-v1"):
        CandidateTextModelAdapter(data, FakeBackend(AdapterPrediction("human", 0.8)))


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (AdapterPrediction("unknown", 0.8), "outside the manifest"),
        (AdapterPrediction("human", float("nan")), "finite"),
        (AdapterPrediction("human", 1.01), "within"),
    ],
)
def test_adapter_rejects_invalid_backend_predictions(
    prediction: AdapterPrediction,
    message: str,
) -> None:
    adapter = CandidateTextModelAdapter(manifest(), FakeBackend(prediction))

    with pytest.raises(ModelAdapterError, match=message):
        adapter.predict("güvenli örnek")
