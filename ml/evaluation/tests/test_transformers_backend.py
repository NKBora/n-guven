from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from nguven_evaluation.transformers_backend import LocalTransformersBackend


def test_backend_loads_only_local_safetensors_without_remote_code(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls["tokenizer"] = {"path": path, **kwargs}
            return object()

    class Model:
        config = SimpleNamespace(num_labels=2)

        def eval(self) -> None:
            calls["eval"] = {}

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls["model"] = {"path": path, **kwargs}
            return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=AutoModelForSequenceClassification,
            AutoTokenizer=AutoTokenizer,
        ),
    )

    LocalTransformersBackend(
        tmp_path,
        labels={"0": "human", "1": "synthetic"},
        max_sequence_length=512,
    )

    assert calls["tokenizer"] == {
        "path": str(tmp_path),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert calls["model"] == {
        "path": str(tmp_path),
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert "eval" in calls
