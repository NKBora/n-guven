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
    "DatasetLeakageError",
    "DatasetInputError",
    "DatasetIntegrityError",
    "DEFAULT_PREPROCESSING_VERSION",
    "EvaluationInputError",
    "EvaluationMetadata",
    "IntegrityReport",
    "ManifestValidationError",
    "PreprocessedText",
    "SUPPORTED_PREPROCESSING_VERSIONS",
    "SplitRatios",
    "TextPreprocessingError",
    "assign_splits",
    "audit_manifest",
    "compute_text_content_hash",
    "evaluate_predictions",
    "load_manifest",
    "load_dataset_input",
    "load_predictions",
    "preprocess_turkish_text",
    "verify_dataset_content_hashes",
]
