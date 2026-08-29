"""Deterministic dataset splitting and leakage controls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SPLIT_NAMES = ("train", "validation", "test")
# Generator family is a reporting slice, not a default leakage group. Treating an
# entire generator family as one component makes a three-way stratified split
# impossible when the benchmark intentionally contains only a few families.
DEFAULT_GROUP_DIMENSIONS = ("source",)
SUPPORTED_GROUP_DIMENSIONS = ("source", "generator")


class DatasetLeakageError(ValueError):
    """Raised when duplicate or cross-split records are detected."""


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Split ratios must be between 0 and 1")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1")


def audit_manifest(
    records: Sequence[dict[str, Any]],
    *,
    group_dimensions: Sequence[str] = DEFAULT_GROUP_DIMENSIONS,
) -> None:
    """Reject duplicate identities/content and grouping keys spanning splits."""
    _validate_group_dimensions(group_dimensions)
    issues = _duplicate_issues(records)

    groups: dict[str, set[str]] = {}
    for record in records:
        split = str(record["split"])
        for token in _group_tokens(record, group_dimensions):
            groups.setdefault(token, set()).add(split)

    for token, splits in sorted(groups.items()):
        if len(splits) > 1:
            issues.append(f"group {token!r} spans splits: {', '.join(sorted(splits))}")

    if issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise DatasetLeakageError(f"Dataset leakage checks failed:\n{details}")


def assign_splits(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    ratios: SplitRatios = SplitRatios(),
    group_dimensions: Sequence[str] = DEFAULT_GROUP_DIMENSIONS,
) -> list[dict[str, Any]]:
    """Assign connected source/generator groups to stable hash-based splits."""
    _validate_group_dimensions(group_dimensions)
    duplicate_issues = _duplicate_issues(records)
    if duplicate_issues:
        details = "\n".join(f"- {issue}" for issue in duplicate_issues)
        raise DatasetLeakageError(f"Dataset leakage checks failed:\n{details}")

    components = _connected_components(records, group_dimensions)
    split_by_index: dict[int, str] = {}
    for indices, tokens in components:
        component_key = "|".join(sorted(tokens))
        if not component_key:
            component_key = str(records[indices[0]]["contentHash"])
        split = _select_split(component_key, seed=seed, ratios=ratios)
        split_by_index.update((index, split) for index in indices)

    assigned = [dict(record, split=split_by_index[index]) for index, record in enumerate(records)]
    audit_manifest(assigned, group_dimensions=group_dimensions)
    return assigned


def _duplicate_issues(records: Sequence[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for field in ("id", "contentHash"):
        seen: dict[Any, int] = {}
        for record_number, record in enumerate(records, start=1):
            value = record[field]
            if value in seen:
                issues.append(
                    f"duplicate {field} {value!r} in records {seen[value]} and {record_number}"
                )
            else:
                seen[value] = record_number
    return issues


def _connected_components(
    records: Sequence[dict[str, Any]],
    group_dimensions: Sequence[str],
) -> list[tuple[list[int], set[str]]]:
    parent = list(range(len(records)))
    token_owner: dict[str, int] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    record_tokens: list[set[str]] = []
    for index, record in enumerate(records):
        tokens = set(_group_tokens(record, group_dimensions))
        record_tokens.append(tokens)
        for token in tokens:
            if token in token_owner:
                union(index, token_owner[token])
            else:
                token_owner[token] = index

    components: dict[int, tuple[list[int], set[str]]] = {}
    for index, tokens in enumerate(record_tokens):
        root = find(index)
        indices, component_tokens = components.setdefault(root, ([], set()))
        indices.append(index)
        component_tokens.update(tokens)
    return list(components.values())


def _group_tokens(
    record: dict[str, Any],
    group_dimensions: Iterable[str],
) -> Iterable[str]:
    if "source" in group_dimensions:
        yield f"source:{record.get('sourceGroup') or record['source']}"

    if "generator" in group_dimensions:
        generator = record.get("generatorFamily") or record.get("generatorModel")
        if generator is not None:
            yield f"generator:{generator}"


def _select_split(component_key: str, *, seed: int, ratios: SplitRatios) -> str:
    digest = hashlib.sha256(f"{seed}:{component_key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < ratios.train:
        return "train"
    if fraction < ratios.train + ratios.validation:
        return "validation"
    return "test"


def _validate_group_dimensions(group_dimensions: Sequence[str]) -> None:
    unknown = sorted(set(group_dimensions) - set(SUPPORTED_GROUP_DIMENSIONS))
    if unknown:
        raise ValueError(f"Unknown grouping dimension(s): {', '.join(unknown)}")
