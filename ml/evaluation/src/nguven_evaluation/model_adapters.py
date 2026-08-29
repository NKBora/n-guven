"""Model-independent adapter boundary for approved Turkish encoder candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from nguven_evaluation.preprocessing import DEFAULT_PREPROCESSING_VERSION


class ModelAdapterError(ValueError):
    """Raised when an adapter or inference backend violates its contract."""


@dataclass(frozen=True)
class CandidateDescriptor:
    """Reviewed upstream identity; this is not a production model selection."""

    adapter_id: str
    display_name: str
    provider: str
    repository: str
    revision: str
    weights_license: str
    source_url: str


@dataclass(frozen=True)
class AdapterPrediction:
    """Normalized classification response returned by every adapter."""

    label: str
    score: float | None


@runtime_checkable
class PredictionBackend(Protocol):
    """Locally supplied inference backend; adapters never download artifacts."""

    def predict(self, text: str) -> AdapterPrediction:
        """Classify one already-preprocessed text value."""


@runtime_checkable
class TextModelAdapter(Protocol):
    """Stable model boundary consumed by offline artifact generation."""

    @property
    def model_id(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def preprocessing_version(self) -> str: ...

    def predict(self, text: str) -> AdapterPrediction: ...


BERTURK = CandidateDescriptor(
    adapter_id="berturk",
    display_name="BERTurk",
    provider="dbmdz, Bavarian State Library",
    repository="dbmdz/bert-base-turkish-cased",
    revision="b6e1de16c983e0f2c70664591ea3f22810072608",
    weights_license="MIT",
    source_url="https://huggingface.co/dbmdz/bert-base-turkish-cased",
)

MODERNBERT_TR = CandidateDescriptor(
    adapter_id="modernbert-tr",
    display_name="ModernBERT-TR",
    provider="YTU CE COSMOS Research Group",
    repository="ytu-ce-cosmos/modernbert-tr-base",
    revision="d16b9da8f7dc0eadf8946255ee5f38042bdf59c2",
    weights_license="Apache-2.0",
    source_url="https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base",
)

CANDIDATES: Mapping[str, CandidateDescriptor] = {
    candidate.adapter_id: candidate for candidate in (BERTURK, MODERNBERT_TR)
}


class CandidateTextModelAdapter:
    """Fail-closed wrapper around a reviewed local model and injected backend."""

    def __init__(self, manifest: Mapping[str, Any], backend: PredictionBackend) -> None:
        adapter_id = str(manifest.get("adapterId", ""))
        try:
            candidate = CANDIDATES[adapter_id]
        except KeyError as error:
            raise ModelAdapterError(f"Unsupported text model adapter: {adapter_id}") from error
        upstream = manifest.get("upstream")
        if not isinstance(upstream, Mapping):
            raise ModelAdapterError("Model manifest is missing upstream provenance")
        if upstream.get("repository") != candidate.repository:
            raise ModelAdapterError("Model manifest repository does not match the adapter candidate")
        if upstream.get("revision") != candidate.revision:
            raise ModelAdapterError("Model manifest revision does not match the reviewed candidate")
        preprocessing_version = str(manifest.get("preprocessingVersion", ""))
        if preprocessing_version != DEFAULT_PREPROCESSING_VERSION:
            raise ModelAdapterError(
                f"Adapter requires preprocessing version {DEFAULT_PREPROCESSING_VERSION}"
            )
        if not isinstance(backend, PredictionBackend):
            raise ModelAdapterError("Inference backend does not implement the prediction contract")

        self._candidate = candidate
        self._model_id = str(manifest.get("modelId", ""))
        self._preprocessing_version = preprocessing_version
        labels = manifest.get("labels")
        if not isinstance(labels, Mapping) or not labels:
            raise ModelAdapterError("Model manifest is missing its label mapping")
        self._labels = frozenset(str(label) for label in labels.values())
        self._backend = backend

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def adapter_id(self) -> str:
        return self._candidate.adapter_id

    @property
    def preprocessing_version(self) -> str:
        return self._preprocessing_version

    def predict(self, text: str) -> AdapterPrediction:
        if not isinstance(text, str) or not text:
            raise ModelAdapterError("Adapter input must be non-empty preprocessed text")
        prediction = self._backend.predict(text)
        if not isinstance(prediction, AdapterPrediction):
            raise ModelAdapterError("Inference backend returned an invalid prediction type")
        if not prediction.label:
            raise ModelAdapterError("Inference backend returned an empty label")
        if prediction.label not in self._labels:
            raise ModelAdapterError("Inference backend returned a label outside the manifest")
        if prediction.score is not None and (
            not math.isfinite(prediction.score) or not 0 <= prediction.score <= 1
        ):
            raise ModelAdapterError("Inference backend score must be finite and within [0, 1]")
        return prediction
