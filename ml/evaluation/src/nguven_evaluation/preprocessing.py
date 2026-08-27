"""Versioned, deterministic preprocessing for Turkish text evaluation."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from nguven_evaluation.integrity import compute_text_content_hash

DEFAULT_PREPROCESSING_VERSION = "tr-text-v1"
SUPPORTED_PREPROCESSING_VERSIONS = frozenset({DEFAULT_PREPROCESSING_VERSION})


class TextPreprocessingError(ValueError):
    """Raised when text cannot be processed by the requested version."""


@dataclass(frozen=True, slots=True)
class PreprocessedText:
    """Processed text plus non-sensitive reproducibility metadata."""

    text: str
    preprocessing_version: str
    input_content_hash: str
    output_content_hash: str
    input_characters: int
    output_characters: int


def preprocess_turkish_text(
    text: str,
    *,
    version: str = DEFAULT_PREPROCESSING_VERSION,
) -> PreprocessedText:
    """Apply the selected Turkish text preprocessing contract."""
    if version not in SUPPORTED_PREPROCESSING_VERSIONS:
        raise TextPreprocessingError(f"Unsupported preprocessing version: {version}")
    if not isinstance(text, str):
        raise TextPreprocessingError("Text input must be a string")
    if "\x00" in text:
        raise TextPreprocessingError("Text input must not contain NUL characters")

    normalized = text.removeprefix("\ufeff")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized)
    if not normalized:
        raise TextPreprocessingError("Text is empty after preprocessing")

    return PreprocessedText(
        text=normalized,
        preprocessing_version=version,
        input_content_hash=compute_text_content_hash(text),
        output_content_hash=compute_text_content_hash(normalized),
        input_characters=len(text),
        output_characters=len(normalized),
    )
