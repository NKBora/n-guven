"""Reproducible evaluation tooling for N-Güven ML services."""

from nguven_evaluation.benchmark import (
    BenchmarkContractError,
    benchmark_evidence_allowed,
    load_benchmark_lock,
)
from nguven_evaluation.calibration import (
    CalibrationError,
    brier_score,
    expected_calibration_error,
    fit_temperature_calibration,
    load_calibration_artifact,
    predicted_synthetic_probability,
    temperature_scale,
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
    validate_comparable_experiments,
    validate_experiment_inputs,
)
from nguven_evaluation.integrity import (
    DatasetIntegrityError,
    IntegrityReport,
    compute_text_content_hash,
    verify_dataset_content_hashes,
)
from nguven_evaluation.inference_data import (
    InferenceDataError,
    PreparedInferenceSplit,
    prepare_inference_split,
    write_private_inference_split,
)
from nguven_evaluation.image_dataset_inputs import (
    ImageDatasetInputError,
    VerifiedImageInput,
    load_image_dataset_inputs,
)
from nguven_evaluation.image_preprocessing import (
    DEFAULT_IMAGE_PREPROCESSING_VERSION,
    ImagePreprocessingError,
    ImageVariant,
    build_image_variants,
    write_preprocessed_image_dataset,
)
from nguven_evaluation.image_model_adapters import (
    CandidateImageModelAdapter,
    ImageCandidateDescriptor,
    ImageModelAdapterError,
    ImagePrediction,
    load_image_candidate_registry,
)
from nguven_evaluation.image_benchmark import (
    ImageBenchmarkContractError,
    image_benchmark_evidence_allowed,
    load_image_benchmark_lock,
)
from nguven_evaluation.image_transformers_backend import (
    LocalTransformersImageBackend,
    materialize_image_candidate,
    verify_image_candidate_artifacts,
)
from nguven_evaluation.image_benchmark_materialization import (
    ImageBenchmarkMaterializationError,
    fetch_locked_image_sources,
    materialize_image_benchmark,
)
from nguven_evaluation.image_offline_predictions import (
    ImageOfflinePredictionError,
    VerifiedImageVariant,
    build_image_benchmark_predictions,
    load_image_benchmark_labels,
    load_preprocessed_image_variants,
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
    "CandidateImageModelAdapter",
    "CalibrationError",
    "DatasetLeakageError",
    "DatasetInputError",
    "DatasetIntegrityError",
    "DEFAULT_IMAGE_PREPROCESSING_VERSION",
    "DEFAULT_PREPROCESSING_VERSION",
    "EvaluationInputError",
    "EvaluationMetadata",
    "ExperimentContractError",
    "FineTuningReadinessError",
    "IntegrityReport",
    "ImageDatasetInputError",
    "ImageCandidateDescriptor",
    "ImageBenchmarkContractError",
    "ImageBenchmarkMaterializationError",
    "ImageModelAdapterError",
    "ImageOfflinePredictionError",
    "ImagePrediction",
    "LocalTransformersImageBackend",
    "ImagePreprocessingError",
    "ImageVariant",
    "InferenceDataError",
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
    "PreparedInferenceSplit",
    "SUPPORTED_PREPROCESSING_VERSIONS",
    "SplitRatios",
    "TextPreprocessingError",
    "TextModelAdapter",
    "TrainingExecutionError",
    "TrainingStageRequest",
    "TrainingStageResult",
    "TransformersTrainerBackend",
    "VerifiedModelArtifacts",
    "VerifiedImageInput",
    "VerifiedImageVariant",
    "assign_splits",
    "audit_manifest",
    "benchmark_evidence_allowed",
    "brier_score",
    "build_preprocessed_records",
    "build_image_variants",
    "build_image_benchmark_predictions",
    "build_prediction_records",
    "build_finetuning_plan",
    "build_training_requests",
    "compare_evaluation_results",
    "compute_text_content_hash",
    "evaluate_predictions",
    "experiment_execution_allowed",
    "expected_calibration_error",
    "fit_temperature_calibration",
    "execute_candidate_training",
    "load_manifest",
    "load_benchmark_lock",
    "load_calibration_artifact",
    "load_evaluation_result",
    "load_experiment_spec",
    "load_finetuning_plan",
    "load_label_ontology",
    "load_model_artifact_manifest",
    "load_dataset_input",
    "load_image_dataset_inputs",
    "load_image_candidate_registry",
    "load_image_benchmark_lock",
    "load_image_benchmark_labels",
    "load_preprocessed_image_variants",
    "materialize_image_candidate",
    "materialize_image_benchmark",
    "load_predictions",
    "image_benchmark_evidence_allowed",
    "fetch_locked_image_sources",
    "load_training_splits",
    "load_preprocessed_records",
    "preprocess_turkish_text",
    "prepare_finetuning_package",
    "prepare_inference_split",
    "predicted_synthetic_probability",
    "package_finetuned_model",
    "verify_dataset_content_hashes",
    "verify_model_artifacts",
    "verify_image_candidate_artifacts",
    "validate_experiment_inputs",
    "validate_comparable_experiments",
    "temperature_scale",
    "write_private_jsonl",
    "write_private_finetuning_package",
    "write_private_inference_split",
    "write_private_predictions",
    "write_preprocessed_image_dataset",
    "write_comparison_report",
    "write_model_manifest",
]
