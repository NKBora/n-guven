"""Reproducible evaluation tooling for N-Güven ML services."""

from nguven_evaluation.benchmark import (
    BenchmarkContractError,
    benchmark_evidence_allowed,
    load_benchmark_lock,
)

from nguven_evaluation.comparison import (
    ModelComparisonError,
    compare_evaluation_results,
    load_evaluation_result,
    write_comparison_report,
)
from nguven_evaluation.dataset_inputs import DatasetInputError, load_dataset_input
from nguven_evaluation.evaluation import (
    EvaluationInputError,
    EvaluationMetadata,
    evaluate_predictions,
    load_predictions,
)
from nguven_evaluation.experiments import (
    ExperimentContractError,
    experiment_execution_allowed,
    load_experiment_spec,
    validate_experiment_inputs,
)
from nguven_evaluation.integrity import (
    DatasetIntegrityError,
    IntegrityReport,
    compute_text_content_hash,
    verify_dataset_content_hashes,
)
from nguven_evaluation.finetuning import (
    FineTuningReadinessError,
    PreparedFineTuningPackage,
    build_finetuning_plan,
    load_finetuning_plan,
    load_label_ontology,
    prepare_finetuning_package,
    write_private_finetuning_package,
)
from nguven_evaluation.manifests import ManifestValidationError, load_manifest
from nguven_evaluation.model_adapters import (
    BERTURK,
    MODERNBERT_TR,
    AdapterPrediction,
    CandidateTextModelAdapter,
    ModelAdapterError,
    TextModelAdapter,
)
from nguven_evaluation.model_artifacts import (
    ModelArtifactError,
    VerifiedModelArtifacts,
    load_model_artifact_manifest,
    verify_model_artifacts,
)
from nguven_evaluation.model_packaging import (
    ModelPackagingError,
    package_finetuned_model,
    write_model_manifest,
)
from nguven_evaluation.offline_preprocessing import (
    OfflinePreprocessingError,
    build_preprocessed_records,
    write_private_jsonl,
)
from nguven_evaluation.offline_predictions import (
    OfflinePredictionError,
    build_prediction_records,
    load_preprocessed_records,
    write_private_predictions,
)
from nguven_evaluation.preprocessing import (
    DEFAULT_PREPROCESSING_VERSION,
    SUPPORTED_PREPROCESSING_VERSIONS,
    PreprocessedText,
    TextPreprocessingError,
    preprocess_turkish_text,
)
from nguven_evaluation.splitting import (
    DatasetLeakageError,
    SplitRatios,
    assign_splits,
    audit_manifest,
)
from nguven_evaluation.training import (
    TrainingExecutionError,
    TrainingStageRequest,
    TrainingStageResult,
    TransformersTrainerBackend,
    build_training_requests,
    execute_candidate_training,
    load_training_splits,
)

__all__ = [
    "AdapterPrediction",
    "BERTURK",
    "BenchmarkContractError",
    "CandidateTextModelAdapter",
    "DatasetLeakageError",
    "DatasetInputError",
    "DatasetIntegrityError",
    "DEFAULT_PREPROCESSING_VERSION",
    "EvaluationInputError",
    "EvaluationMetadata",
    "ExperimentContractError",
    "FineTuningReadinessError",
    "IntegrityReport",
    "ManifestValidationError",
    "MODERNBERT_TR",
    "ModelAdapterError",
    "ModelArtifactError",
    "ModelComparisonError",
    "ModelPackagingError",
    "OfflinePreprocessingError",
    "OfflinePredictionError",
    "PreprocessedText",
    "PreparedFineTuningPackage",
    "SUPPORTED_PREPROCESSING_VERSIONS",
    "SplitRatios",
    "TextPreprocessingError",
    "TextModelAdapter",
    "TrainingExecutionError",
    "TrainingStageRequest",
    "TrainingStageResult",
    "TransformersTrainerBackend",
    "VerifiedModelArtifacts",
    "assign_splits",
    "audit_manifest",
    "benchmark_evidence_allowed",
    "build_preprocessed_records",
    "build_prediction_records",
    "build_finetuning_plan",
    "build_training_requests",
    "compare_evaluation_results",
    "compute_text_content_hash",
    "evaluate_predictions",
    "experiment_execution_allowed",
    "execute_candidate_training",
    "load_manifest",
    "load_benchmark_lock",
    "load_evaluation_result",
    "load_experiment_spec",
    "load_finetuning_plan",
    "load_label_ontology",
    "load_model_artifact_manifest",
    "load_dataset_input",
    "load_predictions",
    "load_training_splits",
    "load_preprocessed_records",
    "preprocess_turkish_text",
    "prepare_finetuning_package",
    "package_finetuned_model",
    "verify_dataset_content_hashes",
    "verify_model_artifacts",
    "validate_experiment_inputs",
    "write_private_jsonl",
    "write_private_finetuning_package",
    "write_private_predictions",
    "write_comparison_report",
    "write_model_manifest",
]
