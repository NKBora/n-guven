# N-Güven Evaluation Pipeline

This package provides model-independent, reproducible controls for N-Güven dataset manifests and offline prediction artifacts.

**No benchmark has been performed yet.**

No dataset, metric, threshold, license conclusion, or model comparison is recorded as an N-Güven result. ModernBERT-TR and BERTurk remain candidates; neither has been integrated or selected.

## Implemented scope

- JSON and JSON Lines dataset manifest validation against JSON Schema Draft 2020-12.
- Strict, local-only JSON/JSONL text input validation without logging raw text.
- One-to-one manifest/input coverage and SHA-256 verification over exact UTF-8 text bytes.
- Versioned Turkish text preprocessing with NFC Unicode and LF newline normalization.
- Duplicate record ID and SHA-256 content-hash rejection.
- Cross-split source and generator-family leakage checks.
- Seeded, order-independent split assignment for connected source/generator groups.
- Offline prediction validation and complete manifest-to-prediction coverage checks.
- Accuracy, macro precision, macro recall, macro F1, and mean inference-time calculation.
- Traceable result artifacts containing the dataset/model versions, Git commit, seed, creation time, and input hashes.
- Strict model artifact provenance and local SHA-256 verification without network downloads.
- A shared adapter boundary for pinned BERTurk and ModernBERT-TR upstream candidates.

The package does not download data or model weights, execute inference, choose a model, or establish production thresholds.

## Layout

```text
ml/evaluation/
├── inputs/
│   └── schema.json
├── manifests/
│   └── schema.json
├── models/
│   └── schema.json
├── preprocessed/
│   └── schema.json
├── predictions/
│   └── schema.json
├── results/
│   ├── .gitkeep
│   └── schema.json
├── scripts/
│   └── .gitkeep
├── src/nguven_evaluation/
├── tests/
└── pyproject.toml
```

Generated result files belong under `results/` but must not be described as project evidence until their dataset and protocol have been reviewed.

## Model artifact and adapter boundary

`models/schema.json` records immutable upstream model and tokenizer revisions, separate weight/code licenses, the required preprocessing version, runtime details, label mapping, artifact sizes, and SHA-256 hashes. Local weights and caches under `models/` are ignored by Git.

The initial adapter registry pins the reviewed upstream identities of `dbmdz/bert-base-turkish-cased` and `ytu-ce-cosmos/modernbert-tr-base`. Their presence is a candidate declaration, not an integration, benchmark result, production approval, or final model selection. Adapters accept only a locally supplied inference backend and never download code or weights.

## Local setup

Python 3.12 is required.

```bash
cd ml/evaluation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## CLI workflow

Validate a JSON or JSON Lines manifest:

```bash
nguven-eval validate-manifest path/to/manifest.jsonl
```

Validate a local text input artifact without printing raw text:

```bash
nguven-eval validate-input path/to/local-input.jsonl
```

Files placed under `ml/evaluation/inputs/` are ignored by Git except for the schema. Prefer an access-controlled path outside the repository for real evaluation content.

Verify that every local text matches its manifest `contentHash`:

```bash
nguven-eval verify-content-hashes path/to/manifest.jsonl \
  --input path/to/local-input.jsonl
```

For text records, `contentHash` is `sha256:` followed by the lowercase SHA-256 digest of the exact decoded text encoded as UTF-8. Normalization is intentionally not applied at this integrity stage; versioned preprocessing is a separate step.

## Turkish text preprocessing

The initial `tr-text-v1` contract removes one leading Unicode BOM, normalizes CRLF/CR line endings to LF, and applies NFC Unicode normalization. It rejects NUL characters and text that becomes empty.

Case, Turkish dotted/dotless I, punctuation, internal spacing, and leading/trailing whitespace are preserved because these may carry useful generation signals. The pipeline deliberately does not lowercase, collapse whitespace, tokenize, truncate, or apply model-specific transformations.

Create a deterministic local-only JSON Lines artifact after manifest and hash verification:

```bash
nguven-eval preprocess-text path/to/manifest.jsonl \
  --input path/to/local-input.jsonl \
  --output ml/evaluation/preprocessed/run-id.jsonl \
  --version tr-text-v1
```

The command refuses source/output path collisions, symbolic-link outputs, non-JSONL output names, and overwrites unless `--force` is explicit. Output is written atomically with owner-only read/write permissions. Files under `ml/evaluation/preprocessed/` are ignored by Git except for the schema because they contain local text.

Reject duplicate identities/content and source or generator groups spanning multiple splits:

```bash
nguven-eval check-leakage path/to/manifest.jsonl
```

Generate deterministic split assignments. Existing output is never replaced without `--force`:

```bash
nguven-eval split-manifest path/to/manifest.jsonl \
  --seed 42 \
  --output path/to/split-manifest.json
```

Evaluate an offline prediction artifact:

```bash
nguven-eval evaluate path/to/split-manifest.json \
  --predictions path/to/predictions.jsonl \
  --output results/run-id.json \
  --run-id run-id \
  --git-commit "$(git rev-parse HEAD)" \
  --seed 42 \
  --dataset-version reviewed-dataset-version \
  --model-name reviewed-model-name \
  --model-version reviewed-model-version
```

The evaluation command first revalidates the manifest and leakage controls. It rejects missing, extra, or duplicate prediction IDs and validates the generated result against `results/schema.json`.

## Data governance

Manifest records must contain verified source, access date, license, content hash, label provenance, intended use, and split metadata. `sourceGroup` and `generatorFamily` should be supplied when the raw source/model names are too granular to enforce family-level isolation.

Only synthetic fixtures are stored in tests. Real evaluation content, private URLs, credentials, personal data, or unreviewed licensed datasets must not be committed.
