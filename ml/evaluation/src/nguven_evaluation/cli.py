"""Command-line interface for N-Güven evaluation tooling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from nguven_evaluation.benchmark import (
    DEFAULT_BENCHMARK_SCHEMA_PATH,
    BenchmarkContractError,
    benchmark_evidence_allowed,
    load_benchmark_lock,
)
from nguven_evaluation.calibration import (
    DEFAULT_CALIBRATION_SCHEMA_PATH,
    CalibrationError,
    fit_temperature_calibration,
    load_calibration_artifact,
)

from nguven_evaluation.dataset_inputs import (
    DEFAULT_INPUT_SCHEMA_PATH,
    DEFAULT_MAX_INPUT_BYTES,
    DatasetInputError,
    load_dataset_input,
)
from nguven_evaluation.comparison import (
    ModelComparisonError,
    compare_result_files,
    write_comparison_report,
)
from nguven_evaluation.evaluation import (
    EvaluationInputError,
    EvaluationMetadata,
    evaluate_predictions,
    load_predictions,
    sha256_file,
)
from nguven_evaluation.environment import (
    DEFAULT_ENVIRONMENT_LOCK_PATH,
    DEFAULT_ENVIRONMENT_SCHEMA_PATH,
    TrainingEnvironmentError,
    load_environment_lock,
    verify_training_environment,
)
from nguven_evaluation.experiments import (
    DEFAULT_EXPERIMENT_SCHEMA_PATH,
    ExperimentContractError,
    load_experiment_spec,
    validate_comparable_experiments,
    validate_experiment_inputs,
)
from nguven_evaluation.integrity import (
    DatasetIntegrityError,
    verify_dataset_content_hashes,
)
from nguven_evaluation.inference_data import (
    InferenceDataError,
    prepare_inference_split,
    sha256_file as inference_sha256,
    write_private_inference_split,
)
from nguven_evaluation.finetuning import (
    DEFAULT_FINETUNING_PLAN_SCHEMA_PATH,
    DEFAULT_FINETUNING_RECORD_SCHEMA_PATH,
    DEFAULT_ONTOLOGY_PATH,
    FineTuningReadinessError,
    build_finetuning_plan,
    load_finetuning_plan,
    load_label_ontology,
    prepare_finetuning_package,
    write_private_finetuning_package,
    write_private_json,
)
from nguven_evaluation.manifests import (
    DEFAULT_SCHEMA_PATH,
    ManifestValidationError,
    load_manifest,
)
from nguven_evaluation.materialization import (
    BenchmarkMaterializationError,
    fetch_locked_sources,
    materialize_benchmark,
    sha256_regular_file as materialized_sha256,
    write_materialized_benchmark,
)
from nguven_evaluation.model_adapters import CandidateTextModelAdapter, ModelAdapterError
from nguven_evaluation.model_artifacts import (
    DEFAULT_MODEL_SCHEMA_PATH,
    ModelArtifactError,
    load_model_artifact_manifest,
    verify_model_artifacts,
)
from nguven_evaluation.model_packaging import (
    ModelPackagingError,
    package_finetuned_model,
    write_model_manifest,
)
from nguven_evaluation.offline_predictions import (
    DEFAULT_MAX_PREPROCESSED_BYTES,
    OfflinePredictionError,
    build_prediction_records,
    ensure_prediction_path_is_distinct,
    load_preprocessed_records,
    write_private_predictions,
)
from nguven_evaluation.offline_preprocessing import (
    DEFAULT_PREPROCESSED_SCHEMA_PATH,
    OfflinePreprocessingError,
    build_preprocessed_records,
    ensure_distinct_artifact_paths,
    write_private_jsonl,
)
from nguven_evaluation.preprocessing import DEFAULT_PREPROCESSING_VERSION
from nguven_evaluation.splitting import (
    DEFAULT_GROUP_DIMENSIONS,
    DatasetLeakageError,
    SplitRatios,
    SUPPORTED_GROUP_DIMENSIONS,
    assign_splits,
    audit_manifest,
)
from nguven_evaluation.transformers_backend import LocalTransformersBackend
from nguven_evaluation.training import (
    DEFAULT_EXECUTION_SCHEMA_PATH,
    TrainingExecutionError,
    TransformersTrainerBackend,
    execute_candidate_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nguven-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="validate JSON or JSON Lines dataset manifests",
    )
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)

    benchmark_parser = subparsers.add_parser(
        "validate-benchmark",
        help="validate a versioned Turkish text benchmark source lock",
    )
    benchmark_parser.add_argument("benchmark", type=Path)
    benchmark_parser.add_argument(
        "--schema", type=Path, default=DEFAULT_BENCHMARK_SCHEMA_PATH
    )

    materialize_parser = subparsers.add_parser(
        "materialize-benchmark",
        help="download, verify, sample, and preprocess the private Turkish benchmark",
    )
    materialize_parser.add_argument("benchmark", type=Path)
    materialize_parser.add_argument("--source-cache", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--accessed-at", required=True)
    materialize_parser.add_argument("--allow-network", action="store_true")
    materialize_parser.add_argument(
        "--schema", type=Path, default=DEFAULT_BENCHMARK_SCHEMA_PATH
    )

    experiment_parser = subparsers.add_parser(
        "validate-experiment",
        help="validate one pinned candidate fine-tuning experiment",
    )
    experiment_parser.add_argument("experiment", type=Path)
    experiment_parser.add_argument(
        "--schema", type=Path, default=DEFAULT_EXPERIMENT_SCHEMA_PATH
    )

    experiment_pair_parser = subparsers.add_parser(
        "validate-experiment-pair",
        help="verify BERTurk and ModernBERT-TR use an identical comparison protocol",
    )
    experiment_pair_parser.add_argument(
        "--experiment", type=Path, action="append", required=True
    )

    environment_parser = subparsers.add_parser(
        "validate-training-environment",
        help="verify the active runtime against the reviewed experiment lock",
    )
    environment_parser.add_argument(
        "environment", type=Path, default=DEFAULT_ENVIRONMENT_LOCK_PATH, nargs="?"
    )
    environment_parser.add_argument(
        "--schema", type=Path, default=DEFAULT_ENVIRONMENT_SCHEMA_PATH
    )
    experiment_pair_parser.add_argument(
        "--schema", type=Path, default=DEFAULT_EXPERIMENT_SCHEMA_PATH
    )

    input_parser = subparsers.add_parser(
        "validate-input",
        help="validate a local-only JSON or JSON Lines text dataset input",
    )
    input_parser.add_argument("input", type=Path)
    input_parser.add_argument("--schema", type=Path, default=DEFAULT_INPUT_SCHEMA_PATH)
    input_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_INPUT_BYTES,
    )

    integrity_parser = subparsers.add_parser(
        "verify-content-hashes",
        help="verify local text content against manifest SHA-256 hashes",
    )
    integrity_parser.add_argument("manifest", type=Path)
    integrity_parser.add_argument("--input", type=Path, required=True)
    integrity_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    integrity_parser.add_argument(
        "--input-schema",
        type=Path,
        default=DEFAULT_INPUT_SCHEMA_PATH,
    )
    integrity_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_INPUT_BYTES,
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess-text",
        help="verify and preprocess local text into a private JSON Lines artifact",
    )
    preprocess_parser.add_argument("manifest", type=Path)
    preprocess_parser.add_argument("--input", type=Path, required=True)
    preprocess_parser.add_argument("--output", type=Path, required=True)
    preprocess_parser.add_argument("--version", default=DEFAULT_PREPROCESSING_VERSION)
    preprocess_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    preprocess_parser.add_argument(
        "--input-schema",
        type=Path,
        default=DEFAULT_INPUT_SCHEMA_PATH,
    )
    preprocess_parser.add_argument(
        "--output-schema",
        type=Path,
        default=DEFAULT_PREPROCESSED_SCHEMA_PATH,
    )
    preprocess_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_INPUT_BYTES,
    )
    preprocess_parser.add_argument("--force", action="store_true")

    plan_parser = subparsers.add_parser(
        "create-finetuning-plan",
        help="create one immutable protocol for both fine-tuning candidates",
    )
    plan_parser.add_argument("manifest", type=Path)
    plan_parser.add_argument("--preprocessed", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--plan-id", required=True)
    plan_parser.add_argument("--dataset-version", required=True)
    plan_parser.add_argument("--seed", type=int, action="append", required=True)
    plan_parser.add_argument("--epochs", type=int, default=3)
    plan_parser.add_argument("--train-batch-size", type=int, default=16)
    plan_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    plan_parser.add_argument("--evaluation-batch-size", type=int, default=32)
    plan_parser.add_argument("--learning-rate", type=float, default=0.00002)
    plan_parser.add_argument("--weight-decay", type=float, default=0.01)
    plan_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    plan_parser.add_argument("--max-sequence-length", type=int, default=512)
    plan_parser.add_argument("--early-stopping-patience", type=int, default=2)
    plan_parser.add_argument(
        "--plan-schema",
        type=Path,
        default=DEFAULT_FINETUNING_PLAN_SCHEMA_PATH,
    )

    readiness_parser = subparsers.add_parser(
        "prepare-finetuning-data",
        help="materialize private train/validation data and an isolated test package",
    )
    readiness_parser.add_argument("manifest", type=Path)
    readiness_parser.add_argument("--preprocessed", type=Path, required=True)
    readiness_parser.add_argument("--plan", type=Path, required=True)
    readiness_parser.add_argument("--output", type=Path, required=True)
    readiness_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    readiness_parser.add_argument(
        "--preprocessed-schema",
        type=Path,
        default=DEFAULT_PREPROCESSED_SCHEMA_PATH,
    )
    readiness_parser.add_argument(
        "--record-schema",
        type=Path,
        default=DEFAULT_FINETUNING_RECORD_SCHEMA_PATH,
    )
    readiness_parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)

    inference_data_parser = subparsers.add_parser(
        "prepare-inference-split",
        help="materialize an aligned validation or completion-gated frozen test split",
    )
    inference_data_parser.add_argument("manifest", type=Path)
    inference_data_parser.add_argument("--preprocessed", type=Path, required=True)
    inference_data_parser.add_argument(
        "--split", required=True, choices=("validation", "test")
    )
    inference_data_parser.add_argument("--experiment", type=Path, action="append")
    inference_data_parser.add_argument("--output", type=Path, required=True)
    inference_data_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    inference_data_parser.add_argument(
        "--preprocessed-schema",
        type=Path,
        default=DEFAULT_PREPROCESSED_SCHEMA_PATH,
    )

    train_parser = subparsers.add_parser(
        "train-text-model",
        help="run linear-probe and fine-tuning stages without access to test data",
    )
    train_parser.add_argument("training_root", type=Path)
    train_parser.add_argument("--plan", type=Path, required=True)
    train_parser.add_argument("--experiment", type=Path, required=True)
    train_parser.add_argument("--benchmark", type=Path, required=True)
    train_parser.add_argument(
        "--environment-lock",
        type=Path,
        default=DEFAULT_ENVIRONMENT_LOCK_PATH,
    )
    train_parser.add_argument(
        "--adapter-id", required=True, choices=("berturk", "modernbert-tr")
    )
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--git-commit", required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--cache-dir", type=Path)
    train_parser.add_argument("--allow-network", action="store_true")
    train_parser.add_argument(
        "--execution-schema", type=Path, default=DEFAULT_EXECUTION_SCHEMA_PATH
    )

    package_parser = subparsers.add_parser(
        "package-finetuned-model",
        help="create a verified manifest for one local fine-tuned safetensors bundle",
    )
    package_parser.add_argument("artifact_root", type=Path)
    package_parser.add_argument("--output", type=Path, required=True)
    package_parser.add_argument("--model-id", required=True)
    package_parser.add_argument(
        "--adapter-id",
        required=True,
        choices=("berturk", "modernbert-tr"),
    )
    package_parser.add_argument("--framework-version", required=True)
    package_parser.add_argument("--max-sequence-length", type=int, required=True)
    package_parser.add_argument("--plan", type=Path, required=True)
    package_parser.add_argument("--seed", type=int, required=True)
    package_parser.add_argument("--git-commit", required=True)
    package_parser.add_argument("--intended-use", required=True)
    package_parser.add_argument("--limitations", required=True)
    package_parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    package_parser.add_argument("--model-schema", type=Path, default=DEFAULT_MODEL_SCHEMA_PATH)

    compare_parser = subparsers.add_parser(
        "compare-models",
        help="aggregate same-seed evaluation evidence for the reviewed candidates",
    )
    compare_parser.add_argument("--plan", type=Path, required=True)
    compare_parser.add_argument("--result", type=Path, action="append", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    predict_parser = subparsers.add_parser(
        "predict-text",
        help="generate private predictions from verified local model files",
    )
    predict_parser.add_argument("preprocessed", type=Path)
    predict_parser.add_argument("--model-manifest", type=Path, required=True)
    predict_parser.add_argument("--artifact-root", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-schema", type=Path, default=DEFAULT_MODEL_SCHEMA_PATH)
    predict_parser.add_argument(
        "--preprocessed-schema",
        type=Path,
        default=DEFAULT_PREPROCESSED_SCHEMA_PATH,
    )
    predict_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_PREPROCESSED_BYTES,
    )
    predict_parser.add_argument("--force", action="store_true")

    audit_parser = subparsers.add_parser(
        "check-leakage",
        help="check duplicate records and cross-split source/generator groups",
    )
    audit_parser.add_argument("manifest", type=Path)
    audit_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    _add_grouping_arguments(audit_parser)

    split_parser = subparsers.add_parser(
        "split-manifest",
        help="assign deterministic leakage-safe splits and write a JSON manifest",
    )
    split_parser.add_argument("manifest", type=Path)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, required=True)
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--validation-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    split_parser.add_argument("--force", action="store_true")
    _add_grouping_arguments(split_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate an offline prediction artifact against a manifest",
    )
    evaluate_parser.add_argument("manifest", type=Path)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--git-commit", required=True)
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--dataset-version", required=True)
    evaluate_parser.add_argument("--model-name", required=True)
    evaluate_parser.add_argument("--model-version", required=True)
    evaluate_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    evaluate_parser.add_argument("--calibration", type=Path)
    evaluate_parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    evaluate_parser.add_argument("--force", action="store_true")

    calibration_parser = subparsers.add_parser(
        "fit-temperature-calibration",
        help="fit validation-only temperature scaling for one candidate run",
    )
    calibration_parser.add_argument("manifest", type=Path)
    calibration_parser.add_argument("--predictions", type=Path, required=True)
    calibration_parser.add_argument("--output", type=Path, required=True)
    calibration_parser.add_argument(
        "--model-name", required=True, choices=("berturk", "modernbert-tr")
    )
    calibration_parser.add_argument("--model-version", required=True)
    calibration_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    calibration_parser.add_argument(
        "--calibration-schema", type=Path, default=DEFAULT_CALIBRATION_SCHEMA_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "validate-benchmark":
            benchmark_lock = load_benchmark_lock(args.benchmark, schema_path=args.schema)
        elif args.command == "materialize-benchmark":
            benchmark_lock = load_benchmark_lock(args.benchmark, schema_path=args.schema)
            source_paths = fetch_locked_sources(
                benchmark_lock,
                args.source_cache,
                allow_network=args.allow_network,
            )
            materialized_benchmark = materialize_benchmark(
                benchmark_lock,
                source_paths,
                accessed_at=args.accessed_at,
            )
            write_materialized_benchmark(materialized_benchmark, args.output)
            materialized_hashes = {
                "manifestSha256": materialized_sha256(args.output / "manifest.jsonl"),
                "preprocessedSha256": materialized_sha256(
                    args.output / "preprocessed.jsonl"
                ),
                "recordCount": materialized_benchmark.summary["recordCount"],
            }
        elif args.command == "validate-experiment":
            experiment_specification = load_experiment_spec(
                args.experiment, schema_path=args.schema
            )
        elif args.command == "validate-experiment-pair":
            experiment_specifications = [
                load_experiment_spec(path, schema_path=args.schema)
                for path in args.experiment
            ]
            validate_comparable_experiments(experiment_specifications)
        elif args.command == "validate-training-environment":
            environment_lock = load_environment_lock(
                args.environment,
                schema_path=args.schema,
            )
            active_environment = verify_training_environment(environment_lock)
        elif args.command == "fit-temperature-calibration":
            records = load_manifest(args.manifest, schema_path=args.schema)
            audit_manifest(records)
            predictions = load_predictions(args.predictions)
            calibration_artifact = fit_temperature_calibration(
                records,
                predictions,
                model_name=args.model_name,
                model_version=args.model_version,
                manifest_sha256=sha256_file(args.manifest),
                predictions_sha256=sha256_file(args.predictions),
                schema_path=args.calibration_schema,
            )
            write_private_json(calibration_artifact, args.output)
        elif args.command == "create-finetuning-plan":
            load_manifest(args.manifest)
            load_preprocessed_records(args.preprocessed)
            plan = build_finetuning_plan(
                plan_id=args.plan_id,
                dataset_version=args.dataset_version,
                manifest_path=args.manifest,
                preprocessed_path=args.preprocessed,
                seeds=args.seed,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                evaluation_batch_size=args.evaluation_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                max_sequence_length=args.max_sequence_length,
                early_stopping_patience=args.early_stopping_patience,
                schema_path=args.plan_schema,
            )
            write_private_json(plan, args.output)
        elif args.command == "prepare-finetuning-data":
            records = load_manifest(args.manifest, schema_path=args.schema)
            preprocessed_records = load_preprocessed_records(
                args.preprocessed,
                schema_path=args.preprocessed_schema,
            )
            plan = load_finetuning_plan(
                args.plan,
                manifest_path=args.manifest,
                preprocessed_path=args.preprocessed,
            )
            ontology = load_label_ontology(args.ontology)
            readiness_package = prepare_finetuning_package(
                records,
                preprocessed_records,
                plan=plan,
                ontology=ontology,
                record_schema_path=args.record_schema,
            )
            write_private_finetuning_package(readiness_package, args.output)
        elif args.command == "prepare-inference-split":
            records = load_manifest(args.manifest, schema_path=args.schema)
            preprocessed_records = load_preprocessed_records(
                args.preprocessed,
                schema_path=args.preprocessed_schema,
            )
            experiments = [
                load_experiment_spec(path) for path in (args.experiment or [])
            ]
            inference_package = prepare_inference_split(
                records,
                preprocessed_records,
                split=args.split,
                experiments=experiments,
            )
            inference_provenance = write_private_inference_split(
                inference_package,
                args.output,
                source_manifest_sha256=inference_sha256(args.manifest),
                source_preprocessed_sha256=inference_sha256(args.preprocessed),
            )
        elif args.command == "train-text-model":
            training_plan = load_finetuning_plan(args.plan)
            experiment_specification = load_experiment_spec(args.experiment)
            benchmark_lock = load_benchmark_lock(args.benchmark)
            validate_experiment_inputs(
                experiment_specification,
                plan=training_plan,
                benchmark=benchmark_lock,
                require_execution_ready=True,
            )
            if experiment_specification["adapterId"] != args.adapter_id:
                raise ExperimentContractError(
                    "Training adapter differs from the reviewed experiment"
                )
            environment_lock = load_environment_lock(args.environment_lock)
            verify_training_environment(environment_lock)
            training_execution = execute_candidate_training(
                plan_path=args.plan,
                training_root=args.training_root,
                adapter_id=args.adapter_id,
                run_id=args.run_id,
                git_commit=args.git_commit,
                output_root=args.output,
                backend=TransformersTrainerBackend(
                    allow_network=args.allow_network,
                    cache_dir=args.cache_dir,
                ),
                experiment=experiment_specification,
                experiment_sha256=materialized_sha256(args.experiment),
                benchmark=benchmark_lock,
                environment_lock=environment_lock,
                environment_sha256=materialized_sha256(args.environment_lock),
                execution_schema_path=args.execution_schema,
            )
        elif args.command == "package-finetuned-model":
            if args.output.resolve(strict=False).is_relative_to(
                args.artifact_root.resolve(strict=False)
            ):
                raise ModelPackagingError(
                    "Model manifest output must be outside the artifact directory"
                )
            model_manifest = package_finetuned_model(
                args.artifact_root,
                model_id=args.model_id,
                adapter_id=args.adapter_id,
                framework_version=args.framework_version,
                max_sequence_length=args.max_sequence_length,
                plan_path=args.plan,
                seed=args.seed,
                git_commit=args.git_commit,
                intended_use=args.intended_use,
                limitations=args.limitations,
                ontology_path=args.ontology,
                schema_path=args.model_schema,
            )
            write_model_manifest(model_manifest, args.output)
        elif args.command == "compare-models":
            comparison_report = compare_result_files(args.result, plan_path=args.plan)
            write_comparison_report(comparison_report, args.output)
        elif args.command == "predict-text":
            ensure_prediction_path_is_distinct(
                args.output,
                [args.preprocessed, args.model_manifest, args.model_schema, args.preprocessed_schema],
            )
            model_manifest = load_model_artifact_manifest(
                args.model_manifest,
                schema_path=args.model_schema,
            )
            verify_model_artifacts(model_manifest, artifact_root=args.artifact_root)
            runtime = model_manifest["runtime"]
            if runtime["framework"] != "transformers" or runtime["artifactFormat"] != "safetensors":
                raise OfflinePredictionError(
                    "predict-text currently requires a Transformers safetensors artifact"
                )
            backend = LocalTransformersBackend(
                args.artifact_root,
                labels=model_manifest["labels"],
                max_sequence_length=int(runtime["maxSequenceLength"]),
            )
            adapter = CandidateTextModelAdapter(model_manifest, backend)
            records = load_preprocessed_records(
                args.preprocessed,
                schema_path=args.preprocessed_schema,
                max_file_bytes=args.max_file_bytes,
            )
            records = build_prediction_records(records, adapter=adapter)
            prediction_sha256 = write_private_predictions(
                records,
                args.output,
                force=args.force,
            )
        elif args.command == "validate-input":
            records = load_dataset_input(
                args.input,
                schema_path=args.schema,
                max_file_bytes=args.max_file_bytes,
            )
        else:
            records = load_manifest(args.manifest, schema_path=args.schema)
        if args.command == "preprocess-text":
            ensure_distinct_artifact_paths(
                args.output,
                [args.manifest, args.input, args.schema, args.input_schema, args.output_schema],
            )
            input_records = load_dataset_input(
                args.input,
                schema_path=args.input_schema,
                max_file_bytes=args.max_file_bytes,
            )
            records = build_preprocessed_records(
                records,
                input_records,
                version=args.version,
                schema_path=args.output_schema,
            )
            write_private_jsonl(records, args.output, force=args.force)
        elif args.command == "verify-content-hashes":
            input_records = load_dataset_input(
                args.input,
                schema_path=args.input_schema,
                max_file_bytes=args.max_file_bytes,
            )
            integrity_report = verify_dataset_content_hashes(records, input_records)
        elif args.command == "check-leakage":
            audit_manifest(
                records,
                group_dimensions=args.group_by or DEFAULT_GROUP_DIMENSIONS,
            )
        elif args.command == "split-manifest":
            if args.output.exists() and not args.force:
                raise ValueError(f"Output already exists; use --force to replace it: {args.output}")
            ratios = SplitRatios(args.train_ratio, args.validation_ratio, args.test_ratio)
            records = assign_splits(
                records,
                seed=args.seed,
                ratios=ratios,
                group_dimensions=args.group_by or DEFAULT_GROUP_DIMENSIONS,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        elif args.command == "evaluate":
            if args.output.exists() and not args.force:
                raise ValueError(f"Output already exists; use --force to replace it: {args.output}")
            audit_manifest(records)
            predictions = load_predictions(args.predictions)
            metadata = EvaluationMetadata(
                run_id=args.run_id,
                git_commit=args.git_commit,
                seed=args.seed,
                dataset_version=args.dataset_version,
                model_name=args.model_name,
                model_version=args.model_version,
            )
            result = evaluate_predictions(
                records,
                predictions,
                metadata=metadata,
                manifest_sha256=sha256_file(args.manifest),
                predictions_sha256=sha256_file(args.predictions),
                calibration_artifact=(
                    load_calibration_artifact(args.calibration)
                    if args.calibration is not None
                    else None
                ),
                high_confidence_threshold=args.high_confidence_threshold,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (
        BenchmarkContractError,
        BenchmarkMaterializationError,
        CalibrationError,
        DatasetLeakageError,
        DatasetInputError,
        DatasetIntegrityError,
        EvaluationInputError,
        ExperimentContractError,
        FineTuningReadinessError,
        InferenceDataError,
        ManifestValidationError,
        ModelAdapterError,
        ModelArtifactError,
        ModelComparisonError,
        ModelPackagingError,
        OfflinePredictionError,
        OfflinePreprocessingError,
        TrainingExecutionError,
        TrainingEnvironmentError,
        ValueError,
    ) as error:
        print(error)
        return 1

    if args.command == "validate-benchmark":
        evidence = "enabled" if benchmark_evidence_allowed(benchmark_lock) else "disabled"
        print(
            f"Validated benchmark {benchmark_lock['benchmarkId']}:{benchmark_lock['version']} "
            f"with result evidence {evidence}"
        )
    elif args.command == "materialize-benchmark":
        print(json.dumps(materialized_hashes, sort_keys=True))
    elif args.command == "validate-experiment":
        print(
            f"Validated {experiment_specification['adapterId']} experiment "
            f"{experiment_specification['experimentId']} with status "
            f"{experiment_specification['execution']['status']}"
        )
    elif args.command == "validate-experiment-pair":
        print("Validated identical BERTurk and ModernBERT-TR experiment protocols")
    elif args.command == "validate-training-environment":
        print(
            f"Validated {environment_lock['environmentId']} on "
            f"{active_environment['device']}"
        )
    elif args.command == "fit-temperature-calibration":
        print(
            f"Wrote validation-only temperature calibration for "
            f"{calibration_artifact['model']['name']} to {args.output}"
        )
    elif args.command == "create-finetuning-plan":
        print(f"Wrote fine-tuning plan {plan['planId']} to {args.output}")
    elif args.command == "prepare-finetuning-data":
        print(
            f"Prepared {readiness_package.summary['recordCount']} fine-tuning record(s) "
            f"with an isolated test split at {args.output}"
        )
    elif args.command == "prepare-inference-split":
        print(
            f"Prepared {inference_provenance['recordCount']} {args.split} inference "
            f"record(s) at {args.output}"
        )
    elif args.command == "train-text-model":
        print(
            f"Completed {len(training_execution['stages'])} training stage(s) for "
            f"{training_execution['adapterId']} at {args.output}"
        )
    elif args.command == "package-finetuned-model":
        print(
            f"Wrote manifest for {model_manifest['modelId']} with "
            f"{len(model_manifest['artifacts'])} verified artifact(s) to {args.output}"
        )
    elif args.command == "compare-models":
        leaders = ", ".join(comparison_report["selection"]["leaders"])
        print(
            f"Wrote {comparison_report['selection']['status']} comparison result "
            f"for {leaders} to {args.output}"
        )
    elif args.command == "predict-text":
        print(
            f"Wrote {len(records)} prediction record(s) to {args.output} "
            f"(sha256:{prediction_sha256})"
        )
    elif args.command == "validate-manifest":
        print(f"Validated {len(records)} record(s) from {args.manifest}")
    elif args.command == "validate-input":
        print(f"Validated {len(records)} local input record(s) from {args.input}")
    elif args.command == "verify-content-hashes":
        print(
            f"Verified {integrity_report.verified_record_count} content hash(es) "
            f"from {args.manifest}"
        )
    elif args.command == "preprocess-text":
        print(
            f"Wrote {len(records)} preprocessed record(s) with {args.version} "
            f"to {args.output}"
        )
    elif args.command == "check-leakage":
        print(f"Leakage checks passed for {len(records)} record(s) from {args.manifest}")
    elif args.command == "split-manifest":
        print(f"Wrote {len(records)} record(s) to {args.output}")
    else:
        print(f"Wrote evaluation result for {len(records)} record(s) to {args.output}")
    return 0


def _add_grouping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group-by",
        action="append",
        choices=SUPPORTED_GROUP_DIMENSIONS,
        default=None,
        help="grouping dimension; repeat to select multiple (default: source and generator)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
