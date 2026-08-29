"""Reproducible evaluation tooling for N-Güven ML services."""

from nguven_evaluation.dataset_inputs import DatasetInputError, load_dataset_input
from nguven_evaluation.evaluation import (
    EvaluationInputError,
    EvaluationMetadata,
    evaluate_predictions,
    load_predictions,
)
from nguven_evaluation.integrity import (
    DatasetIntegrityError,
    IntegrityReport,
    compute_text_content_hash,
    verify_dataset_content_hashes,
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

__all__ = [
    "AdapterPrediction",
    "BERTURK",
    "CandidateTextModelAdapter",
    "DatasetLeakageError",
    "DatasetInputError",
    "DatasetIntegrityError",
    "DEFAULT_PREPROCESSING_VERSION",
    "EvaluationInputError",
    "EvaluationMetadata",
    "IntegrityReport",
    "ManifestValidationError",
    "MODERNBERT_TR",
    "ModelAdapterError",
    "ModelArtifactError",
    "OfflinePreprocessingError",
    "OfflinePredictionError",
    "PreprocessedText",
    "SUPPORTED_PREPROCESSING_VERSIONS",
    "SplitRatios",
    "TextPreprocessingError",
    "TextModelAdapter",
    "VerifiedModelArtifacts",
    "assign_splits",
    "audit_manifest",
    "build_preprocessed_records",
    "build_prediction_records",
    "compute_text_content_hash",
    "evaluate_predictions",
    "load_manifest",
    "load_model_artifact_manifest",
    "load_dataset_input",
    "load_predictions",
    "load_preprocessed_records",
    "preprocess_turkish_text",
    "verify_dataset_content_hashes",
    "verify_model_artifacts",
    "write_private_jsonl",
    "write_private_predictions",
]
