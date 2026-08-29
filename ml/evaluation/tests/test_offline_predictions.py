from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Iterator

import pytest

from nguven_evaluation.integrity import compute_text_content_hash
from nguven_evaluation.model_adapters import AdapterPrediction, ModelAdapterError
from nguven_evaluation.offline_predictions import (
    OfflinePredictionError,
    build_prediction_records,
    load_preprocessed_records,
    write_private_predictions,
)


class FakeAdapter:
    model_id = "synthetic-model"
    adapter_id = "berturk"
    preprocessing_version = "tr-text-v1"

    def predict(self, text: str) -> AdapterPrediction:
        return AdapterPrediction("synthetic" if "AI" in text else "human", 0.75)


class FailingAdapter(FakeAdapter):
    def predict(self, text: str) -> AdapterPrediction:
        raise ModelAdapterError(f"backend leaked: {text}")


def record(record_id: str, text: str, version: str = "tr-text-v1") -> dict[str, object]:
    content_hash = compute_text_content_hash(text)
    return {
        "id": record_id,
        "text": text,
        "preprocessingVersion": version,
        "inputContentHash": content_hash,
        "outputContentHash": content_hash,
        "inputCharacters": len(text),
        "outputCharacters": len(text),
    }


def clock(values: list[int]) -> Iterator[int]:
    yield from values


def test_load_preprocessed_records_rechecks_output_hash(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    item = record("a", "güvenli metin")
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")

    assert load_preprocessed_records(path) == [item]

    item["text"] = "değiştirildi"
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(OfflinePredictionError, match="hash mismatch"):
        load_preprocessed_records(path)


def test_build_predictions_is_sorted_and_records_inference_time() -> None:
    timer = clock([10_000_000, 12_000_000, 20_000_000, 25_500_000])
    records = [record("b", "AI üretimi"), record("a", "insan yazısı")]

    output = build_prediction_records(records, adapter=FakeAdapter(), clock_ns=lambda: next(timer))

    assert [item["id"] for item in output] == ["a", "b"]
    assert output[0] == {
        "id": "a",
        "predictedLabel": "human",
        "score": 0.75,
        "inferenceMs": 2.0,
    }
    assert output[1]["inferenceMs"] == 5.5


def test_build_predictions_rejects_preprocessing_mismatch() -> None:
    with pytest.raises(OfflinePredictionError, match="Preprocessing version mismatch"):
        build_prediction_records([record("a", "örnek", "tr-text-v2")], adapter=FakeAdapter())


def test_inference_error_redacts_input_text() -> None:
    private_text = "kişisel ve hassas metin"

    with pytest.raises(OfflinePredictionError, match="record id: safe-id") as error:
        build_prediction_records([record("safe-id", private_text)], adapter=FailingAdapter())

    assert private_text not in str(error.value)


def test_write_predictions_is_owner_only_atomic_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    records = [
        {"id": "a", "predictedLabel": "human", "score": 0.75, "inferenceMs": 2.0}
    ]

    digest = write_private_predictions(records, path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text(encoding="utf-8")) == records[0]
    with pytest.raises(OfflinePredictionError, match="--force"):
        write_private_predictions(records, path)


def test_write_predictions_rejects_symbolic_link_output(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("preserve", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    output.symlink_to(target)

    with pytest.raises(OfflinePredictionError, match="symbolic link"):
        write_private_predictions([], output, force=True)

    assert target.read_text(encoding="utf-8") == "preserve"
