from __future__ import annotations

import unicodedata

import pytest

from nguven_evaluation.integrity import compute_text_content_hash
from nguven_evaluation.preprocessing import (
    DEFAULT_PREPROCESSING_VERSION,
    TextPreprocessingError,
    preprocess_turkish_text,
)


def test_preprocessing_normalizes_unicode_to_nfc() -> None:
    decomposed = "Tu\u0308rkc\u0327e"

    result = preprocess_turkish_text(decomposed)

    assert result.text == unicodedata.normalize("NFC", decomposed)
    assert unicodedata.is_normalized("NFC", result.text)


def test_preprocessing_normalizes_line_endings_and_removes_leading_bom() -> None:
    result = preprocess_turkish_text("\ufeffBirinci\r\nİkinci\rÜçüncü")

    assert result.text == "Birinci\nİkinci\nÜçüncü"


def test_preprocessing_preserves_case_punctuation_and_spacing() -> None:
    text = "  I İ ı i; Türkçe!  "

    result = preprocess_turkish_text(text)

    assert result.text == text


def test_preprocessing_is_idempotent() -> None:
    first = preprocess_turkish_text("\ufeffİlk\r\nsatır")
    second = preprocess_turkish_text(first.text)

    assert second.text == first.text
    assert second.output_content_hash == first.output_content_hash


def test_preprocessing_records_version_hashes_and_character_counts() -> None:
    text = "\ufeffTürkçe\r\nmetin"

    result = preprocess_turkish_text(text)

    assert result.preprocessing_version == DEFAULT_PREPROCESSING_VERSION
    assert result.input_content_hash == compute_text_content_hash(text)
    assert result.output_content_hash == compute_text_content_hash(result.text)
    assert result.input_characters == len(text)
    assert result.output_characters == len(result.text)


def test_preprocessing_rejects_unsupported_version() -> None:
    with pytest.raises(TextPreprocessingError, match="Unsupported preprocessing version"):
        preprocess_turkish_text("Test metni", version="tr-text-v999")


def test_preprocessing_rejects_nul_character() -> None:
    with pytest.raises(TextPreprocessingError, match="NUL"):
        preprocess_turkish_text("Güvenli olmayan\x00metin")


def test_preprocessing_rejects_text_that_becomes_empty() -> None:
    with pytest.raises(TextPreprocessingError, match="empty after preprocessing"):
        preprocess_turkish_text("\ufeff")
