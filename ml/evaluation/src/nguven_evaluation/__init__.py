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
    "EvaluationInputError",
    "EvaluationMetadata",
    "IntegrityReport",
    "ManifestValidationError",
    "SplitRatios",
    "assign_splits",
    "audit_manifest",
    "compute_text_content_hash",
    "evaluate_predictions",
    "load_manifest",
    "load_dataset_input",
    "load_predictions",
    "verify_dataset_content_hashes",
]
