"""Reproducible Transformers training for reviewed Turkish encoder candidates."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator

from nguven_evaluation.finetuning import (
    DEFAULT_FINETUNING_RECORD_SCHEMA_PATH,
    FineTuningReadinessError,
    load_finetuning_plan,
    sha256_regular_file,
)
from nguven_evaluation.model_adapters import CANDIDATES, CandidateDescriptor

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTION_SCHEMA_PATH = EVALUATION_ROOT / "finetuning" / "execution.schema.json"
DEFAULT_MAX_TRAINING_FILE_BYTES = 512 * 1024 * 1024
TRAINING_STAGES = ("linear-probe", "fine-tune")


class TrainingExecutionError(ValueError):
    """Raised when a fine-tuning execution cannot preserve the frozen protocol."""


@dataclass(frozen=True)
class TrainingStageRequest:
    adapter_id: str
    stage: str
    seed: int
    output_directory: Path


@dataclass(frozen=True)
class TrainingStageResult:
    stage: str
    seed: int
    validation_macro_f1: float
    best_epoch: float
    training_seconds: float
    artifact_directory: str


class TrainingBackend(Protocol):
    """Injected backend so orchestration can be tested without model downloads."""

    def train(
        self,
        request: TrainingStageRequest,
        *,
        candidate: CandidateDescriptor,
        protocol: Mapping[str, Any],
        train_records: Sequence[Mapping[str, Any]],
        validation_records: Sequence[Mapping[str, Any]],
    ) -> TrainingStageResult: ...


def load_training_splits(
    training_root: Path,
    *,
    record_schema_path: Path = DEFAULT_FINETUNING_RECORD_SCHEMA_PATH,
    max_file_bytes: int = DEFAULT_MAX_TRAINING_FILE_BYTES,
) -> dict[str, list[dict[str, Any]]]:
    """Load train/validation only and reject any test material in the training root."""
    if training_root.is_symlink() or not training_root.is_dir():
        raise TrainingExecutionError("Training root must be a regular directory")
    forbidden = sorted(
        path.name
        for path in training_root.iterdir()
        if "test" in path.name.lower() or path.name.lower().startswith("evaluation")
    )
    if forbidden:
        raise TrainingExecutionError(
            "Training root contains forbidden test material: " + ", ".join(forbidden)
        )
    unexpected = sorted(
        path.name
        for path in training_root.iterdir()
        if path.name not in {"train.jsonl", "validation.jsonl"}
    )
    if unexpected:
        raise TrainingExecutionError(
            "Training root contains unexpected entries: " + ", ".join(unexpected)
        )

    schema = _load_json_object(record_schema_path, "fine-tuning record schema")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    loaded: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        path = training_root / f"{split}.jsonl"
        records = _load_jsonl(path, max_file_bytes=max_file_bytes)
        for number, record in enumerate(records, start=1):
            errors = list(validator.iter_errors(record))
            if errors:
                raise TrainingExecutionError(
                    f"Invalid {split} training record at line {number}"
                )
            if record["split"] != split:
                raise TrainingExecutionError(
                    f"Training record split mismatch at {split} line {number}"
                )
        if not records:
            raise TrainingExecutionError(f"Training split is empty: {split}")
        loaded[split] = records
    return loaded


def build_training_requests(
    plan: Mapping[str, Any],
    *,
    adapter_id: str,
    output_root: Path,
) -> list[TrainingStageRequest]:
    """Build deterministic linear-probe then fine-tune requests for every seed."""
    if adapter_id not in CANDIDATES:
        raise TrainingExecutionError(f"Unsupported training adapter: {adapter_id}")
    plan_candidates = {str(item["adapterId"]) for item in plan["candidates"]}
    if adapter_id not in plan_candidates:
        raise TrainingExecutionError("Training adapter is absent from the fine-tuning plan")
    requests: list[TrainingStageRequest] = []
    for seed in plan["protocol"]["seeds"]:
        for stage in TRAINING_STAGES:
            requests.append(
                TrainingStageRequest(
                    adapter_id=adapter_id,
                    stage=stage,
                    seed=int(seed),
                    output_directory=output_root / f"seed-{seed}" / stage,
                )
            )
    return requests


def execute_candidate_training(
    *,
    plan_path: Path,
    training_root: Path,
    adapter_id: str,
    run_id: str,
    git_commit: str,
    output_root: Path,
    backend: TrainingBackend,
    execution_schema_path: Path = DEFAULT_EXECUTION_SCHEMA_PATH,
) -> dict[str, Any]:
    """Execute both reviewed training stages atomically without access to test text."""
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in run_id):
        raise TrainingExecutionError("Run id must use lowercase letters, digits, dot, dash, or underscore")
    if not 7 <= len(git_commit) <= 40 or any(character not in "0123456789abcdef" for character in git_commit):
        raise TrainingExecutionError("Git commit must be a 7-40 character lowercase SHA")
    if output_root.exists() or output_root.is_symlink():
        raise TrainingExecutionError("Training output directory must not already exist")

    plan = load_finetuning_plan(plan_path)
    splits = load_training_splits(training_root)
    candidate = CANDIDATES.get(adapter_id)
    if candidate is None:
        raise TrainingExecutionError(f"Unsupported training adapter: {adapter_id}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    os.chmod(temporary_root, 0o700)
    stage_results: list[TrainingStageResult] = []
    try:
        requests = build_training_requests(
            plan, adapter_id=adapter_id, output_root=temporary_root
        )
        for request in requests:
            request.output_directory.mkdir(parents=True, mode=0o700)
            result = backend.train(
                request,
                candidate=candidate,
                protocol=plan["protocol"],
                train_records=splits["train"],
                validation_records=splits["validation"],
            )
            _validate_stage_result(result, request=request, temporary_root=temporary_root)
            stage_results.append(result)

        selections = _select_stages(stage_results, seeds=plan["protocol"]["seeds"])
        document = {
            "schemaVersion": "1.0",
            "runId": run_id,
            "planId": plan["planId"],
            "planSha256": sha256_regular_file(plan_path),
            "adapterId": adapter_id,
            "upstream": {
                "repository": candidate.repository,
                "revision": candidate.revision,
            },
            "dataset": {
                "version": plan["dataset"]["version"],
                "manifestSha256": plan["dataset"]["manifestSha256"],
                "preprocessedSha256": plan["dataset"]["preprocessedSha256"],
            },
            "gitCommit": git_commit,
            "stages": [
                {
                    "stage": result.stage,
                    "seed": result.seed,
                    "validationMacroF1": result.validation_macro_f1,
                    "bestEpoch": result.best_epoch,
                    "trainingSeconds": result.training_seconds,
                    "artifactDirectory": result.artifact_directory,
                }
                for result in stage_results
            ],
            "selection": selections,
        }
        _validate_execution(document, schema_path=execution_schema_path)
        _write_private_json(temporary_root / "execution.json", document)
        os.replace(temporary_root, output_root)
        return document
    except (OSError, FineTuningReadinessError) as error:
        raise TrainingExecutionError("Fine-tuning execution failed safely") from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


class TransformersTrainerBackend:
    """Lazy local-or-pinned Transformers backend with safe serialization."""

    def __init__(self, *, allow_network: bool = False, cache_dir: Path | None = None) -> None:
        self._allow_network = allow_network
        self._cache_dir = cache_dir

    def train(
        self,
        request: TrainingStageRequest,
        *,
        candidate: CandidateDescriptor,
        protocol: Mapping[str, Any],
        train_records: Sequence[Mapping[str, Any]],
        validation_records: Sequence[Mapping[str, Any]],
    ) -> TrainingStageResult:
        try:
            import numpy as np
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                DataCollatorWithPadding,
                EarlyStoppingCallback,
                Trainer,
                TrainingArguments,
                set_seed,
            )
        except ImportError as error:
            raise TrainingExecutionError(
                "Install the training extra before running Transformers fine-tuning"
            ) from error

        set_seed(request.seed)
        load_options = {
            "revision": candidate.revision,
            "cache_dir": str(self._cache_dir) if self._cache_dir else None,
            "local_files_only": not self._allow_network,
            "trust_remote_code": False,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate.repository, **load_options)
            model = AutoModelForSequenceClassification.from_pretrained(
                candidate.repository,
                num_labels=2,
                id2label={0: "human", 1: "synthetic"},
                label2id={"human": 0, "synthetic": 1},
                use_safetensors=True,
                ignore_mismatched_sizes=True,
                **load_options,
            )
        except (OSError, ValueError) as error:
            raise TrainingExecutionError(
                "Unable to load the pinned candidate revision safely"
            ) from error

        if request.stage == "linear-probe":
            for parameter in model.base_model.parameters():
                parameter.requires_grad = False

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
                self._records = list(records)

            def __len__(self) -> int:
                return len(self._records)

            def __getitem__(self, index: int) -> dict[str, Any]:
                record = self._records[index]
                encoded = tokenizer(
                    str(record["text"]),
                    truncation=True,
                    max_length=int(protocol["maxSequenceLength"]),
                )
                encoded["labels"] = int(record["labelId"])
                return encoded

        def compute_metrics(prediction: Any) -> dict[str, float]:
            predicted = np.argmax(prediction.predictions, axis=-1).tolist()
            actual = prediction.label_ids.tolist()
            return {"macroF1": _macro_f1(actual, predicted)}

        arguments = TrainingArguments(
            output_dir=str(request.output_directory / "checkpoints"),
            overwrite_output_dir=False,
            num_train_epochs=int(protocol["epochs"]),
            per_device_train_batch_size=int(protocol["trainBatchSize"]),
            per_device_eval_batch_size=int(protocol["evaluationBatchSize"]),
            learning_rate=float(protocol["learningRate"]),
            weight_decay=float(protocol["weightDecay"]),
            warmup_ratio=float(protocol["warmupRatio"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macroF1",
            greater_is_better=True,
            save_safetensors=True,
            save_total_limit=2,
            seed=request.seed,
            data_seed=request.seed,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=TextDataset(train_records),
            eval_dataset=TextDataset(validation_records),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=int(protocol["earlyStoppingPatience"])
                )
            ],
        )
        started = time.monotonic()
        train_output = trainer.train()
        metrics = trainer.evaluate()
        elapsed = time.monotonic() - started
        export = request.output_directory / "export"
        export.mkdir(mode=0o700)
        trainer.save_model(str(export))
        tokenizer.save_pretrained(str(export))
        for path in export.iterdir():
            if path.is_file():
                os.chmod(path, 0o600)
        best_epoch = float(train_output.metrics.get("epoch", 0.0))
        validation_macro_f1 = float(metrics["eval_macroF1"])
        return TrainingStageResult(
            stage=request.stage,
            seed=request.seed,
            validation_macro_f1=validation_macro_f1,
            best_epoch=best_epoch,
            training_seconds=elapsed,
            artifact_directory=str(request.output_directory.relative_to(request.output_directory.parents[1]) / "export"),
        )


def _select_stages(
    results: Sequence[TrainingStageResult], *, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for seed in seeds:
        candidates = [result for result in results if result.seed == int(seed)]
        if {result.stage for result in candidates} != set(TRAINING_STAGES):
            raise TrainingExecutionError(f"Missing training stage for seed {seed}")
        winner = max(
            candidates,
            key=lambda result: (
                result.validation_macro_f1,
                1 if result.stage == "linear-probe" else 0,
            ),
        )
        selected.append(
            {
                "seed": int(seed),
                "selectedStage": winner.stage,
                "validationMacroF1": winner.validation_macro_f1,
            }
        )
    return selected


def _validate_stage_result(
    result: TrainingStageResult,
    *,
    request: TrainingStageRequest,
    temporary_root: Path,
) -> None:
    if result.stage != request.stage or result.seed != request.seed:
        raise TrainingExecutionError("Training backend returned mismatched stage metadata")
    for value in (
        result.validation_macro_f1,
        result.best_epoch,
        result.training_seconds,
    ):
        if not math.isfinite(value) or value < 0:
            raise TrainingExecutionError("Training backend returned invalid metrics")
    if result.validation_macro_f1 > 1:
        raise TrainingExecutionError("Validation Macro F1 must be within [0, 1]")
    artifact_path = temporary_root / result.artifact_directory
    if artifact_path.resolve(strict=False).is_relative_to(temporary_root.resolve(strict=False)) is False:
        raise TrainingExecutionError("Training artifact escaped the output root")
    if artifact_path.is_symlink() or not artifact_path.is_dir():
        raise TrainingExecutionError("Training backend did not produce an artifact directory")


def _macro_f1(actual: Sequence[int], predicted: Sequence[int]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise TrainingExecutionError("Metric inputs must be non-empty and aligned")
    scores: list[float] = []
    for label in (0, 1):
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _validate_execution(document: Mapping[str, Any], *, schema_path: Path) -> None:
    schema = _load_json_object(schema_path, "execution schema")
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        raise TrainingExecutionError("Generated training execution violates its schema")


def _load_jsonl(path: Path, *, max_file_bytes: int) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise TrainingExecutionError(f"Training split must be a regular file: {path.name}")
    if path.stat().st_size > max_file_bytes:
        raise TrainingExecutionError(f"Training split exceeds the size limit: {path.name}")
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise TrainingExecutionError(f"Blank training record at line {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TrainingExecutionError(f"Training record must be an object at line {number}")
                records.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingExecutionError(f"Training split must be valid UTF-8 JSON Lines: {path.name}") from error
    return records


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrainingExecutionError(f"{label.title()} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingExecutionError(f"{label.title()} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TrainingExecutionError(f"{label.title()} must be a JSON object")
    return value


def _write_private_json(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        json.dump(document, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
