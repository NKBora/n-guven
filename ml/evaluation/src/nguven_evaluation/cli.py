"""Command-line interface for N-Güven evaluation tooling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from nguven_evaluation.evaluation import (
    EvaluationInputError,
    EvaluationMetadata,
    evaluate_predictions,
    load_predictions,
    sha256_file,
)
from nguven_evaluation.manifests import (
    DEFAULT_SCHEMA_PATH,
    ManifestValidationError,
    load_manifest,
)
from nguven_evaluation.splitting import (
    DEFAULT_GROUP_DIMENSIONS,
    DatasetLeakageError,
    SplitRatios,
    assign_splits,
    audit_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nguven-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="validate JSON or JSON Lines dataset manifests",
    )
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)

    audit_parser = subparsers.add_parser(
        "check-leakage",
        help="check duplicate records and cross-split source/generator groups",
    )
    audit_parser.add_argument("manifest", type=Path)
    audit_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    _add_grouping_arguments(audit_parser)

    split_parser = subparsers.add_parser(
        "split-manifest",
        help="assign deterministic leakage-safe splits and write a JSON manifest",
    )
    split_parser.add_argument("manifest", type=Path)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, required=True)
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--validation-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    split_parser.add_argument("--force", action="store_true")
    _add_grouping_arguments(split_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate an offline prediction artifact against a manifest",
    )
    evaluate_parser.add_argument("manifest", type=Path)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--git-commit", required=True)
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--dataset-version", required=True)
    evaluate_parser.add_argument("--model-name", required=True)
    evaluate_parser.add_argument("--model-version", required=True)
    evaluate_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    evaluate_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        records = load_manifest(args.manifest, schema_path=args.schema)
        if args.command == "check-leakage":
            audit_manifest(
                records,
                group_dimensions=args.group_by or DEFAULT_GROUP_DIMENSIONS,
            )
        elif args.command == "split-manifest":
            if args.output.exists() and not args.force:
                raise ValueError(f"Output already exists; use --force to replace it: {args.output}")
            ratios = SplitRatios(args.train_ratio, args.validation_ratio, args.test_ratio)
            records = assign_splits(
                records,
                seed=args.seed,
                ratios=ratios,
                group_dimensions=args.group_by or DEFAULT_GROUP_DIMENSIONS,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        elif args.command == "evaluate":
            if args.output.exists() and not args.force:
                raise ValueError(f"Output already exists; use --force to replace it: {args.output}")
            audit_manifest(records)
            predictions = load_predictions(args.predictions)
            metadata = EvaluationMetadata(
                run_id=args.run_id,
                git_commit=args.git_commit,
                seed=args.seed,
                dataset_version=args.dataset_version,
                model_name=args.model_name,
                model_version=args.model_version,
            )
            result = evaluate_predictions(
                records,
                predictions,
                metadata=metadata,
                manifest_sha256=sha256_file(args.manifest),
                predictions_sha256=sha256_file(args.predictions),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (
        DatasetLeakageError,
        EvaluationInputError,
        ManifestValidationError,
        ValueError,
    ) as error:
        print(error)
        return 1

    if args.command == "validate-manifest":
        print(f"Validated {len(records)} record(s) from {args.manifest}")
    elif args.command == "check-leakage":
        print(f"Leakage checks passed for {len(records)} record(s) from {args.manifest}")
    elif args.command == "split-manifest":
        print(f"Wrote {len(records)} record(s) to {args.output}")
    else:
        print(f"Wrote evaluation result for {len(records)} record(s) to {args.output}")
    return 0


def _add_grouping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group-by",
        action="append",
        choices=DEFAULT_GROUP_DIMENSIONS,
        default=None,
        help="grouping dimension; repeat to select multiple (default: source and generator)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
