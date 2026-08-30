from dataclasses import dataclass
import os
from pathlib import Path


def _positive_int_from_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_path_from_environment(name: str) -> Path | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    return Path(raw_value).expanduser()


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "text-ai"
    max_text_length: int = _positive_int_from_environment(
        "TEXT_AI_MAX_TEXT_LENGTH",
        10_000,
    )
    max_analysis_id_length: int = 128
    model_root: Path | None = _optional_path_from_environment("TEXT_AI_MODEL_ROOT")


settings = Settings()
