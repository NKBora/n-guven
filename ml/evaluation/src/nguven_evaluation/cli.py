"""Command-line interface for N-Güven evaluation tooling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from nguven_evaluation.manifests import (
    DEFAULT_SCHEMA_PATH,
    ManifestValidationError,
    load_manifest,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        records = load_manifest(args.manifest, schema_path=args.schema)
    except ManifestValidationError as error:
        print(error)
        return 1

    print(f"Validated {len(records)} record(s) from {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
