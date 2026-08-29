"""Model-independent readiness controls for fair Turkish text fine-tuning."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from nguven_evaluation.model_adapters import CANDIDATES
from nguven_evaluation.preprocessing import DEFAULT_PREPROCESSING_VERSION
from nguven_evaluation.splitting import SPLIT_NAMES, audit_manifest

EVALUATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY_PATH = EVALUATION_ROOT / "labels" / "text-origin-v1.json"
DEFAULT_FINETUNING_PLAN_SCHEMA_PATH = EVALUATION_ROOT / "finetuning" / "plan.schema.json"
DEFAULT_FINETUNING_RECORD_SCHEMA_PATH = EVALUATION_ROOT / "finetuning" / "record.schema.json"
DEFAULT_ONTOLOGY_VERSION = "text-origin-v1"
DEFAULT_LABELS: Mapping[str, str] = {"0": "human", "1": "synthetic"}
DEFAULT_MAX_CONTROL_FILE_BYTES = 1024 * 1024


class FineTuningReadinessError(ValueError):
    """Raised when fine-tuning inputs do not satisfy the shared protocol."""


@dataclass(frozen=True)
class PreparedFineTuningPackage:
    """Private split artifacts plus a non-textual audit summary."""

    training_records: Mapping[str, list[dict[str, Any]]]
    test_records: list[dict[str, Any]]
    test_manifest_records: list[dict[str, Any]]
    summary: dict[str, Any]
    plan: dict[str, Any]


def build_finetuning_plan(
    *,
    plan_id: str,
    dataset_version: str,
    manifest_path: Path,
    preprocessed_path: Path,
    seeds: Sequence[int],
    epochs: int,
    train_batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_sequence_length: int,
    early_stopping_patience: int,
    schema_path: Path = DEFAULT_FINETUNING_PLAN_SCHEMA_PATH,
) -> dict[str, Any]:
    """Create one immutable, shared plan for both reviewed model candidates."""
    plan = {
        "schemaVersion": "1.0",
        "planId": plan_id,
        "dataset": {
            "version": dataset_version,
            "manifestSha256": sha256_regular_file(manifest_path),
            "preprocessedSha256": sha256_regular_file(preprocessed_path),
            "ontologyVersion": DEFAULT_ONTOLOGY_VERSION,
            "preprocessingVersion": DEFAULT_PREPROCESSING_VERSION,
        },
        "candidates": [
            {
                "adapterId": candidate.adapter_id,
                "repository": candidate.repository,
                "revision": candidate.revision,
            }
            for candidate in CANDIDATES.values()
        ],
        "protocol": {
            "seeds": list(seeds),
            "epochs": epochs,
            "trainBatchSize": train_batch_size,
            "evaluationBatchSize": evaluation_batch_size,
            "learningRate": learning_rate,
            "weightDecay": weight_decay,
            "warmupRatio": warmup_ratio,
            "maxSequenceLength": max_sequence_length,
            "earlyStoppingPatience": early_stopping_patience,
            "metricForBestModel": "macroF1",
            "greaterIsBetter": True,
            "loadBestModelAtEnd": True,
            "saveFormat": "safetensors",
            "testSplitPolicy": "untouched-until-final-evaluation",
        },
    }
    validate_finetuning_plan(plan, schema_path=schema_path)
    return plan


def load_finetuning_plan(
    plan_path: Path,
    *,
    manifest_path: Path | None = None,
    preprocessed_path: Path | None = None,
    schema_path: Path = DEFAULT_FINETUNING_PLAN_SCHEMA_PATH,
) -> dict[str, Any]:
    """Load a plan and optionally bind it to exact local dataset artifacts."""
    plan = _load_json_object(plan_path, "fine-tuning plan")
    validate_finetuning_plan(plan, schema_path=schema_path)
    dataset = plan["dataset"]
    if manifest_path is not None and not hmac.compare_digest(
        str(dataset["manifestSha256"]), sha256_regular_file(manifest_path)
    ):
        raise FineTuningReadinessError("Fine-tuning plan manifest hash mismatch")
    if preprocessed_path is not None and not hmac.compare_digest(
        str(dataset["preprocessedSha256"]), sha256_regular_file(preprocessed_path)
    ):
        raise FineTuningReadinessError("Fine-tuning plan preprocessing hash mismatch")
    return plan


def validate_finetuning_plan(
    plan: Mapping[str, Any],
    *,
    schema_path: Path = DEFAULT_FINETUNING_PLAN_SCHEMA_PATH,
) -> None:
    """Validate schema and exact candidate provenance for a shared plan."""
    _validate_document(plan, schema_path=schema_path, label="fine-tuning plan")
    candidates = plan["candidates"]
    candidate_ids = [str(item["adapterId"]) for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise FineTuningReadinessError("Fine-tuning plan contains duplicate candidates")
    if set(candidate_ids) != set(CANDIDATES):
        raise FineTuningReadinessError(
            "Fine-tuning plan must contain exactly BERTurk and ModernBERT-TR"
        )
    for item in candidates:
        candidate = CANDIDATES[str(item["adapterId"])]
        if item["repository"] != candidate.repository or item["revision"] != candidate.revision:
            raise FineTuningReadinessError(
                f"Fine-tuning plan provenance mismatch for candidate: {candidate.adapter_id}"
            )


def load_label_ontology(
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
) -> dict[str, Any]:
    """Load the fixed binary text-origin ontology used by all candidates."""
    ontology = _load_json_object(ontology_path, "label ontology")
    if ontology != {
        "schemaVersion": "1.0",
        "ontologyVersion": DEFAULT_ONTOLOGY_VERSION,
        "task": "text-classification",
        "labels": dict(DEFAULT_LABELS),
    }:
        raise FineTuningReadinessError("Label ontology does not match text-origin-v1")
    return ontology


def prepare_finetuning_package(
    manifest_records: Sequence[dict[str, Any]],
    preprocessed_records: Sequence[dict[str, Any]],
    *,
    plan: Mapping[str, Any],
    ontology: Mapping[str, Any],
    record_schema_path: Path = DEFAULT_FINETUNING_RECORD_SCHEMA_PATH,
) -> PreparedFineTuningPackage:
    """Join verified text to labels and physically isolate the untouched test split."""
    validate_finetuning_plan(plan)
    if ontology.get("ontologyVersion") != plan["dataset"]["ontologyVersion"]:
        raise FineTuningReadinessError("Fine-tuning plan and ontology versions differ")
    labels = ontology.get("labels")
    if labels != dict(DEFAULT_LABELS):
        raise FineTuningReadinessError("Unsupported fine-tuning label ontology")
    label_to_id = {label: int(index) for index, label in labels.items()}

    audit_manifest(manifest_records)
    manifest_by_id = _index_unique(manifest_records, "manifest")
    preprocessed_by_id = _index_unique(preprocessed_records, "preprocessed input")
    missing = sorted(set(manifest_by_id) - set(preprocessed_by_id))
    unexpected = sorted(set(preprocessed_by_id) - set(manifest_by_id))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing preprocessed id(s): " + ", ".join(missing))
        if unexpected:
            details.append("unexpected preprocessed id(s): " + ", ".join(unexpected))
        raise FineTuningReadinessError("Fine-tuning coverage mismatch; " + "; ".join(details))

    split_records: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    split_counts: dict[str, dict[str, int]] = {
        name: {label: 0 for label in label_to_id} for name in SPLIT_NAMES
    }
    expected_preprocessing = str(plan["dataset"]["preprocessingVersion"])
    for record_id in sorted(manifest_by_id):
        manifest = manifest_by_id[record_id]
        preprocessed = preprocessed_by_id[record_id]
        label = manifest["label"]
        if not isinstance(label, str) or label not in label_to_id:
            raise FineTuningReadinessError(f"Unsupported label for record id: {record_id}")
        if not str(manifest["language"]).lower().startswith("tr"):
            raise FineTuningReadinessError(f"Non-Turkish record in fine-tuning data: {record_id}")
        if not str(manifest["contentType"]).lower().startswith("text/"):
            raise FineTuningReadinessError(f"Non-text record in fine-tuning data: {record_id}")
        if preprocessed["preprocessingVersion"] != expected_preprocessing:
            raise FineTuningReadinessError(
                f"Preprocessing version mismatch for record id: {record_id}"
            )
        if not hmac.compare_digest(
            str(manifest["contentHash"]).lower(), str(preprocessed["inputContentHash"])
        ):
            raise FineTuningReadinessError(
                f"Manifest/preprocessing content hash mismatch for record id: {record_id}"
            )
        split = str(manifest["split"])
        prepared = {
            "id": record_id,
            "text": str(preprocessed["text"]),
            "label": label,
            "labelId": label_to_id[label],
            "split": split,
            "preprocessingVersion": expected_preprocessing,
            "inputContentHash": str(preprocessed["inputContentHash"]),
            "outputContentHash": str(preprocessed["outputContentHash"]),
        }
        split_records[split].append(prepared)
        split_counts[split][label] += 1

    missing_classes = [
        f"{split}:{label}"
        for split in SPLIT_NAMES
        for label, count in split_counts[split].items()
        if count == 0
    ]
    if missing_classes:
        raise FineTuningReadinessError(
            "Every split must contain every label; missing: " + ", ".join(missing_classes)
        )
    for records in split_records.values():
        _validate_records(records, schema_path=record_schema_path)

    summary = {
        "schemaVersion": "1.0",
        "planId": plan["planId"],
        "dataset": dict(plan["dataset"]),
        "recordCount": sum(len(records) for records in split_records.values()),
        "splitCounts": split_counts,
        "trainingSplits": ["train", "validation"],
        "isolatedEvaluationSplit": "test",
    }
    return PreparedFineTuningPackage(
        training_records={
            "train": split_records["train"],
            "validation": split_records["validation"],
        },
        test_records=split_records["test"],
        test_manifest_records=[
            manifest_by_id[str(record["id"])] for record in split_records["test"]
        ],
        summary=summary,
        plan=dict(plan),
    )


def write_private_finetuning_package(
    package: PreparedFineTuningPackage,
    output_root: Path,
) -> None:
    """Atomically publish one owner-only package with test data isolated by directory."""
    if output_root.exists() or output_root.is_symlink():
        raise FineTuningReadinessError("Fine-tuning output directory must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.parent.is_symlink():
        raise FineTuningReadinessError("Fine-tuning output parent must not be a symbolic link")
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        os.chmod(temporary_root, 0o700)
        training_root = temporary_root / "training"
        evaluation_root = temporary_root / "evaluation"
        training_root.mkdir(mode=0o700)
        evaluation_root.mkdir(mode=0o700)
        _write_jsonl(training_root / "train.jsonl", package.training_records["train"])
        _write_jsonl(
            training_root / "validation.jsonl",
            package.training_records["validation"],
        )
        _write_jsonl(evaluation_root / "test.jsonl", package.test_records)
        _write_jsonl(
            evaluation_root / "test-manifest.jsonl",
            package.test_manifest_records,
        )
        _write_json(temporary_root / "dataset-summary.json", package.summary)
        _write_json(temporary_root / "plan.json", package.plan)
        os.replace(temporary_root, output_root)
    except OSError as error:
        raise FineTuningReadinessError("Unable to write fine-tuning readiness package") from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def write_private_json(document: Mapping[str, Any], output_path: Path) -> None:
    """Write a bounded control artifact atomically with owner-only permissions."""
    if output_path.exists() or output_path.is_symlink():
        raise FineTuningReadinessError("Control output must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            os.chmod(temporary_path, 0o600)
            json.dump(document, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as error:
        raise FineTuningReadinessError("Unable to write private control artifact") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sha256_regular_file(path: Path) -> str:
    """Hash one explicit regular file while rejecting symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise FineTuningReadinessError("Artifact to hash must be a regular non-symbolic-link file")
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_unique(
    records: Sequence[dict[str, Any]], artifact_name: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for record in records:
        record_id = str(record["id"])
        if record_id in indexed:
            duplicates.add(record_id)
        indexed[record_id] = record
    if duplicates:
        raise FineTuningReadinessError(
            f"Duplicate id(s) in {artifact_name}: " + ", ".join(sorted(duplicates))
        )
    return indexed


def _validate_records(records: Sequence[Mapping[str, Any]], *, schema_path: Path) -> None:
    schema = _load_json_object(schema_path, "fine-tuning record schema")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    invalid = [
        number
        for number, record in enumerate(records, start=1)
        if any(validator.iter_errors(record))
    ]
    if invalid:
        raise FineTuningReadinessError(
            "Generated fine-tuning records violate the contract at record(s): "
            + ", ".join(str(number) for number in invalid)
        )


def _validate_document(
    document: Mapping[str, Any], *, schema_path: Path, label: str
) -> None:
    schema = _load_json_object(schema_path, f"{label} schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda item: list(item.path),
    )
    if errors:
        fields = [
            ".".join(str(part) for part in error.absolute_path) or "<document>"
            for error in errors
        ]
        raise FineTuningReadinessError(
            f"{label.title()} validation failed at: " + ", ".join(fields)
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FineTuningReadinessError(f"{label.title()} must be a regular file")
    if path.stat().st_size > DEFAULT_MAX_CONTROL_FILE_BYTES:
        raise FineTuningReadinessError(f"{label.title()} exceeds the size limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FineTuningReadinessError(f"{label.title()} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise FineTuningReadinessError(f"{label.title()} must be a JSON object")
    return document


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        json.dump(document, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
