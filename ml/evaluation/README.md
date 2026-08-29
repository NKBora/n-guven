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
- Local-only Transformers sequence-classification inference and private prediction JSONL output.
- A fixed `text-origin-v1` ontology and one shared multi-seed fine-tuning plan for both candidates.
- Leakage-safe train/validation materialization with the final test split isolated by directory.
- Fail-closed safetensors packaging and variance-preserving candidate comparison reports.
- A versioned Turkish text benchmark source lock with pinned revisions, licenses,
  generator provenance, balanced sampling targets, and a fail-closed evidence gate.

The package does not download data or model weights, fine-tune models, claim benchmark evidence, or establish production thresholds. It can execute verified local inference after approved fine-tuned artifacts are supplied.

## Turkish text benchmark source lock

`benchmarks/text-origin-tr-v1.json` pins reviewed candidate sources rather than
committing third-party text. It requires a balanced 12,000-record target, at least two
synthetic generator families, deterministic 80/10/10 grouping, and explicit license and
revision metadata. The committed release is intentionally `source-locked`: benchmark
claims remain disabled until a separately reviewed private materialization binds exact
manifest and preprocessing hashes.

```bash
nguven-eval validate-benchmark benchmarks/text-origin-tr-v1.json
```

This source lock is a reproducibility and governance artifact, not a measured dataset
release or a performance result.

## Layout

```text
ml/evaluation/
├── comparisons/
│   └── schema.json
├── finetuning/
│   ├── plan.schema.json
│   └── record.schema.json
├── inputs/
│   └── schema.json
├── labels/
│   └── text-origin-v1.json
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

After an approved candidate has been fine-tuned for text classification, place its complete local safetensors/tokenizer bundle in an ignored artifact directory and prepare a reviewed manifest. Install the optional runtime only on the offline inference host:

```bash
python -m pip install -e ".[inference]"
```

Generate predictions without network access or remote model code:

```bash
nguven-eval predict-text ml/evaluation/preprocessed/run-id.jsonl \
  --model-manifest path/to/reviewed-model-manifest.json \
  --artifact-root ml/evaluation/models/artifacts/reviewed-model \
  --output ml/evaluation/predictions/run-id.jsonl
```

Before loading Transformers, the command verifies every declared local artifact size and SHA-256 hash. It uses `local_files_only=True`, disables remote code, accepts safetensors rather than pickle-based weights, validates preprocessing compatibility and labels, and writes predictions atomically with owner-only permissions.

## Fine-tuning and comparison readiness

This repository does not pretrain either model. Both pretrained candidates must be fine-tuned under one immutable plan and evaluated on the same untouched test records. Create that plan only after the reviewed manifest and `tr-text-v1` artifact exist:

```bash
nguven-eval create-finetuning-plan path/to/split-manifest.jsonl \
  --preprocessed path/to/preprocessed.jsonl \
  --output path/to/plan.json \
  --plan-id text-origin-comparison-v1 \
  --dataset-version reviewed-dataset-v1 \
  --seed 17 --seed 42 --seed 71
```

Materialize the exact fine-tuning package:

```bash
nguven-eval prepare-finetuning-data path/to/split-manifest.jsonl \
  --preprocessed path/to/preprocessed.jsonl \
  --plan path/to/plan.json \
  --output ml/evaluation/finetuning/runs/comparison-v1
```

The output contains `training/train.jsonl` and `training/validation.jsonl`. Test text and its test-only manifest are placed under `evaluation/`, so a fine-tuning command can receive only the `training/` directory. Every split must contain both `human` and `synthetic` labels; coverage, content hashes, Turkish text constraints, preprocessing version, duplicate content, source groups, and generator families are checked again before output.

After fine-tuning, export a clean directory containing `model.safetensors`, `config.json`, tokenizer files, and no checkpoints, optimizer state, pickle files, executable code, subdirectories, or symbolic links. Package it for the existing offline prediction command:

```bash
nguven-eval package-finetuned-model path/to/clean-export \
  --output path/to/berturk-manifest.json \
  --model-id berturk-text-origin-v1 \
  --adapter-id berturk \
  --framework-version 4.48.0 \
  --max-sequence-length 512 \
  --plan path/to/plan.json \
  --seed 17 \
  --git-commit abcdef1234567 \
  --intended-use "Offline Turkish human and synthetic text comparison." \
  --limitations "Requires independent validation before any production decision."
```

Run `predict-text` and `evaluate` once per candidate and seed against only `evaluation/test.jsonl` and `evaluation/test-manifest.jsonl`. Then aggregate all result files without discarding seed variance:

```bash
nguven-eval compare-models \
  --plan path/to/plan.json \
  --result path/to/berturk-seed-17.json \
  --result path/to/modernbert-tr-seed-17.json \
  --output ml/evaluation/comparisons/reports/comparison-v1.json
```

The comparison requires identical dataset identity and seed coverage. It ranks mean macro F1 first, mean accuracy second, and mean inference time third while retaining population standard deviation and every prediction artifact hash.

## Reproducible candidate training

Install the isolated training runtime, then pass only the package's `training/`
directory. The command fails if that directory contains test or unexpected material.

```bash
python -m pip install -e ".[training]"
nguven-eval train-text-model path/to/package/training \
  --plan path/to/plan.json \
  --adapter-id berturk \
  --run-id berturk-text-origin-v1 \
  --git-commit "$(git rev-parse HEAD)" \
  --output path/to/private-runs/berturk-v1
```

Every seed first runs a frozen-encoder linear probe and then low-learning-rate
fine-tuning under the same immutable protocol. Selection uses validation Macro F1;
ties prefer the simpler linear probe. Pinned upstream revisions are loaded with remote
code disabled and safetensors required. Network access is opt-in only for the initial
reviewed download; subsequent runs can use an explicit local cache.

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
