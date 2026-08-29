from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from nguven_evaluation.cli import build_parser
from nguven_evaluation.splitting import (
    DatasetLeakageError,
    SplitRatios,
    assign_splits,
    audit_manifest,
)


def record(
    record_id: str,
    *,
    source: str,
    generator: str | None,
    split: str = "train",
) -> dict[str, object]:
    return {
        "id": record_id,
        "source": source,
        "sourceUrl": f"https://example.invalid/{record_id}",
        "accessedAt": "2026-08-27T12:00:00Z",
        "license": "test-only",
        "contentHash": f"sha256:{record_id.encode().hex().ljust(64, '0')[:64]}",
        "language": "tr",
        "contentType": "text/plain",
        "label": "synthetic" if generator else "human",
        "labelSource": "test-fixture",
        "generatorModel": generator,
        "transformation": "none",
        "intendedUse": "Automated testing only",
        "split": split,
    }


def test_audit_rejects_duplicate_content() -> None:
    records = [
        record("sample-a", source="source-a", generator=None),
        record("sample-b", source="source-b", generator=None),
    ]
    records[1]["contentHash"] = records[0]["contentHash"]

    with pytest.raises(DatasetLeakageError, match="duplicate contentHash"):
        audit_manifest(records)


def test_audit_rejects_source_group_across_splits() -> None:
    records = [
        record("sample-a", source="shared-source", generator=None, split="train"),
        record("sample-b", source="shared-source", generator=None, split="test"),
    ]

    with pytest.raises(DatasetLeakageError, match="source:shared-source"):
        audit_manifest(records)


def test_audit_rejects_generator_family_across_splits() -> None:
    first = record("sample-a", source="source-a", generator="model-v1", split="train")
    second = record("sample-b", source="source-b", generator="model-v2", split="test")
    first["generatorFamily"] = "model-family"
    second["generatorFamily"] = "model-family"

    with pytest.raises(DatasetLeakageError, match="generator:model-family"):
        audit_manifest([first, second], group_dimensions=("source", "generator"))


def test_assign_splits_is_repeatable_and_does_not_mutate_input() -> None:
    records = [
        record("sample-a", source="source-a", generator=None),
        record("sample-b", source="source-b", generator="model-b"),
        record("sample-c", source="source-c", generator="model-c"),
    ]
    original = deepcopy(records)

    first = assign_splits(records, seed=42)
    second = assign_splits(records, seed=42)

    assert first == second
    assert records == original


def test_assign_splits_keeps_connected_groups_together() -> None:
    records = [
        record("sample-a", source="shared-source", generator="model-a"),
        record("sample-b", source="shared-source", generator="model-b"),
        record("sample-c", source="source-c", generator="model-b"),
    ]

    assigned = assign_splits(
        records,
        seed=7,
        group_dimensions=("source", "generator"),
    )

    assert len({item["split"] for item in assigned}) == 1


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRatios(0.8, 0.2, 0.1)


def test_cli_can_select_one_grouping_dimension() -> None:
    args = build_parser().parse_args(
        ["check-leakage", str(Path("manifest.json")), "--group-by", "source"]
    )

    assert args.group_by == ["source"]
