"""Local-only Hugging Face Transformers inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from nguven_evaluation.model_adapters import AdapterPrediction, ModelAdapterError


class LocalTransformersBackend:
    """Load a verified sequence classifier without network or remote code execution."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        labels: Mapping[str, str],
        max_sequence_length: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ModelAdapterError(
                'Local Transformers inference requires: pip install -e ".[inference]"'
            ) from error

        if max_sequence_length <= 0:
            raise ModelAdapterError("Maximum sequence length must be positive")
        self._torch = torch
        self._labels = {int(index): label for index, label in labels.items()}
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(artifact_root),
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(artifact_root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
        except (OSError, ValueError) as error:
            raise ModelAdapterError(
                "Unable to load the verified local Transformers model artifact"
            ) from error
        if int(self._model.config.num_labels) != len(self._labels):
            raise ModelAdapterError("Model output count does not match manifest labels")
        if set(self._labels) != set(range(len(self._labels))):
            raise ModelAdapterError("Manifest label indexes must be contiguous from zero")

        self._max_sequence_length = max_sequence_length
        self._model.eval()

    def predict(self, text: str) -> AdapterPrediction:
        try:
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_sequence_length,
            )
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits[0]
                probabilities = self._torch.softmax(logits, dim=-1)
                score, label_index = self._torch.max(probabilities, dim=-1)
            index = int(label_index.item())
            return AdapterPrediction(label=self._labels[index], score=float(score.item()))
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ModelAdapterError("Local Transformers inference failed") from error
