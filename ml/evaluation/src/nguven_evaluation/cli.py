"""Command-line interface for N-Güven evaluation tooling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from nguven_evaluation.dataset_inputs import (
    DEFAULT_INPUT_SCHEMA_PATH,
    DEFAULT_MAX_INPUT_BYTES,
    DatasetInputError,
    load_dataset_input,
)
from nguven_evaluation.evaluation import (
    EvaluationInputError,
    EvaluationMetadata,
    evaluate_predictions,
    load_predictions,
    sha256_file,
)
from nguven_evaluation.integrity import (
    DatasetIntegrityError,
    verify_dataset_content_hashes,
)
from nguven_evaluation.manifests import (
    DEFAULT_SCHEMA_PATH,
    ManifestValidationError,
    load_manifest,
)
from nguven_evaluation.offline_preprocessing import (
    DEFAULT_PREPROCESSED_SCHEMA_PATH,
    OfflinePreprocessingError,
    build_preprocessed_records,
    ensure_distinct_artifact_paths,
    write_private_jsonl,
)
from nguven_evaluation.preprocessing import DEFAULT_PREPROCESSING_VERSION
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

    input_parser = subparsers.add_parser(
        "validate-input",
        help="validate a local-only JSON or JSON Lines text dataset input",
    )
    input_parser.add_argument("input", type=Path)
    input_parser.add_argument("--schema", type=Path, default=DEFAULT_INPUT_SCHEMA_PATH)
    input_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_INPUT_BYTES,
    )

    integrity_parser = subparsers.add_parser(
        "verify-content-hashes",
        help="verify local text content against manifest SHA-256 hashes",
    )
    integrity_parser.add_argument("manifest", type=Path)
    integrity_parser.add_argument("--input", type=Path, required=True)
    integrity_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    integrity_parser.add_argument(
        "--input-schema",
        type=Path,
        default=DEFAULT_INPUT_SCHEMA_PATH,
    )
    integrity_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_INPUT_BYTES,
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess-text",
        help="verify and preprocess local text into a private JSON Lines artifact",
    )
    preprocess_parser.add_argument("manifest", type=Path)
    preprocess_parser.add_argument("--input", type=Path, required=True)
    preprocess_parser.add_argument("--output", type=Path, required=True)
    preprocess_parser.add_argument("--version", default=DEFAULT_PREPROCESSING_VERSION)
    preprocess_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    preprocess_parser.add_argument(
        "--input-schema",
        type=Path,
        default=DEFAULT_INPUT_SCHEMA_PATH,
    )
    preprocess_parser.add_argument(
        "--output-schema",
        type=Path,
        default=DEFAULT_PREPROCESSED_SCHEMA_PATH,
    )
    preprocess_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_INPUT_BYTES,
    )
    preprocess_parser.add_argument("--force", action="store_true")

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
        if args.command == "validate-input":
            records = load_dataset_input(
                args.input,
                schema_path=args.schema,
                max_file_bytes=args.max_file_bytes,
            )
        else:
            records = load_manifest(args.manifest, schema_path=args.schema)
        if args.command == "preprocess-text":
            ensure_distinct_artifact_paths(
                args.output,
                [args.manifest, args.input, args.schema, args.input_schema, args.output_schema],
            )
            input_records = load_dataset_input(
                args.input,
                schema_path=args.input_schema,
                max_file_bytes=args.max_file_bytes,
            )
            records = build_preprocessed_records(
                records,
                input_records,
                version=args.version,
                schema_path=args.output_schema,
            )
            write_private_jsonl(records, args.output, force=args.force)
        elif args.command == "verify-content-hashes":
            input_records = load_dataset_input(
                args.input,
                schema_path=args.input_schema,
                max_file_bytes=args.max_file_bytes,
            )
            integrity_report = verify_dataset_content_hashes(records, input_records)
        elif args.command == "check-leakage":
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
        DatasetInputError,
        DatasetIntegrityError,
        EvaluationInputError,
        ManifestValidationError,
        OfflinePreprocessingError,
        ValueError,
    ) as error:
        print(error)
        return 1

    if args.command == "validate-manifest":
        print(f"Validated {len(records)} record(s) from {args.manifest}")
    elif args.command == "validate-input":
        print(f"Validated {len(records)} local input record(s) from {args.input}")
    elif args.command == "verify-content-hashes":
        print(
            f"Verified {integrity_report.verified_record_count} content hash(es) "
            f"from {args.manifest}"
        )
    elif args.command == "preprocess-text":
        print(
            f"Wrote {len(records)} preprocessed record(s) with {args.version} "
            f"to {args.output}"
        )
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
