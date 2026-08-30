/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# N-Güven Evaluation Pipeline

This package provides model-independent, reproducible controls for N-Güven dataset manifests and offline prediction artifacts.

**The reviewed `text-origin-tr-v1` benchmark is complete.**

BERTurk and ModernBERT-TR were fine-tuned under the same immutable three-seed
protocol, calibrated on validation only, and evaluated once on the frozen 1,200-record
test split. Both passed every acceptance target. Their mean Macro F1 and accuracy were
equal, so the predeclared mean-inference-latency tie-breaker selected BERTurk for the
next prototype integration step. This is not production approval or evidence of
unseen-domain generalization. The canonical report is
`comparisons/text-origin-tr-v1.json` and the scoped decision is recorded in
`../../docs/adr/0001-select-berturk-for-text-origin-prototype.md`.

## Implemented scope

- JSON and JSON Lines dataset manifest validation against JSON Schema Draft 2020-12.
- Strict, local-only JSON/JSONL text input validation without logging raw text.
- One-to-one manifest/input coverage and SHA-256 verification over exact UTF-8 text bytes.
- Versioned Turkish text preprocessing with NFC Unicode and LF newline normalization.
- Duplicate record ID and SHA-256 content-hash rejection.
- Cross-split origin-group leakage checks with optional generator-family holdout audits.
- Seeded, order-independent split assignment for connected origin groups.
- Offline prediction validation and complete manifest-to-prediction coverage checks.
- Accuracy, macro precision, macro recall, macro F1, and mean inference-time calculation.
- Traceable result artifacts containing the dataset/model versions, Git commit, seed, creation time, and input hashes.
- Strict model artifact provenance and local SHA-256 verification without network downloads.
- A shared adapter boundary for pinned BERTurk and ModernBERT-TR upstream candidates.
- Local-only Transformers sequence-classification inference and private prediction JSONL output.
- A fixed `text-origin-v1` ontology and one shared multi-seed fine-tuning plan for both candidates.
- Leakage-safe train/validation materialization with the final test split isolated by directory.
- Completion-gated validation/test inference materialization that keeps the frozen test
  inaccessible until both comparable candidate experiments are closed.
- Fail-closed safetensors packaging and variance-preserving candidate comparison reports.
- A versioned Turkish text benchmark source lock with pinned revisions, licenses,
  generator provenance, balanced sampling targets, and a fail-closed evidence gate.
- Validation-only temperature scaling plus PR-AUC, Brier, ECE, high-confidence false
  positive rate, P95 latency, source/generator/transformation slices, and 95% confidence
  intervals across seeds.
- Local-only image input verification with bounded manifests, safe relative paths,
  symbolic-link isolation, byte-size limits, and SHA-256 integrity checks before decode.
- An immutable Apache-2.0 image candidate pair with reviewed Hugging Face revisions,
  normalized labels, local-only safetensors policy, and remote code disabled.

The package downloads data only through the explicit hash-verified materialization
command and model weights only through the opt-in training command. It does not make
benchmark claims until reviewed private artifacts bind the exact hashes.

Private image inputs can be verified without logging their contents:

```bash
nguven-eval validate-image-input path/to/private-images.jsonl \
  --image-root path/to/private-images
```

The `image-preprocessing-v1` command strips metadata, applies EXIF orientation, converts
to deterministic RGB, and writes canonical plus JPEG, resize, crop, and screenshot
robustness variants into a new private directory:

```bash
nguven-eval preprocess-image path/to/private-images.jsonl \
  --image-root path/to/private-images \
  --output image/preprocessed/private-run
```

The first benchmark pair compares SigLIP2 and ViT checkpoints under one adapter
contract. Their declared training sources are AI-or-Not and CIFAKE respectively; both
are excluded from the external test source. Neither model may be selected from its
model-card score. Selection requires the frozen, source-separated external benchmark
and per-transformation robustness results.

```bash
nguven-eval validate-image-candidates
```

The registry pins full 40-character upstream revisions, Apache-2.0 weight licenses,
and the exact size and SHA-256 of every allowed config, processor, and safetensors
artifact. Runtime materialization must use `local_files_only=True` and
`trust_remote_code=False`; pickle-based `.bin` weights are not accepted.

The frozen external benchmark is source-locked to the Apache-2.0 SynCred-Bench
revision recorded in `image/benchmarks/image-origin-robustness-v1.json`. Its 450 real
negative and 600 synthetic misinformation images are outside the candidates' declared
AI-or-Not training source. Each of the 1,050 originals must be evaluated in all six
preprocessing/robustness forms (6,300 predictions per candidate). Selection prioritizes
the worst transformation Macro F1 before overall Macro F1, false positives, and latency.

```bash
nguven-eval validate-image-benchmark
```

The committed lock is intentionally `source-locked` with result evidence disabled.
Downloading and hashing the roughly 1.46 GB pinned source, materializing the private
images, and completing both model runs are required before that gate may be reviewed.

Model weights are materialized only through the reviewed three-file allowlist. Network
access is opt-in for the initial pinned download; every later verification and inference
step is local-only. The destination is published atomically only after all declared
sizes and SHA-256 hashes match. Unexpected files, symlinks, pickle weights, architecture
drift, and label-map drift fail closed.

Materialize the frozen source only into ignored private storage. The command downloads
the four revision-pinned Parquet shards when explicitly allowed, verifies their exact
source identity, accepts embedded image bytes only, and atomically emits `images/`,
`inputs.jsonl`, `labels.jsonl`, and a non-image summary:

```bash
python -m pip install -e ".[materialization]"
nguven-eval materialize-image-benchmark \
  --source-cache image/benchmarks/cache/syncred-v1 \
  --output image/benchmarks/private/syncred-v1 \
  --allow-network
```

Offline prediction preparation re-verifies every preprocessed variant, requires all six
transformations for every source image, joins private labels with exact coverage, and
produces stable variant IDs plus transformation-aware evaluation records. Inference
errors expose only the variant ID, never image bytes or filenames.

Run each reviewed detector with verified local weights and no network access. The runner
performs identical warm-up, requires the full frozen protocol, records per-image latency,
and atomically writes owner-only manifests, predictions, metrics, and provenance:

```bash
nguven-eval run-image-benchmark \
  --adapter-id siglip2-aiornot \
  --preprocessed-root image/preprocessed/syncred-v1 \
  --labels image/benchmarks/private/syncred-v1/labels.jsonl \
  --artifact-root image/models/artifacts/siglip2-aiornot \
  --output image/predictions/siglip2-syncred-v1 \
  --run-id siglip2-syncred-v1 \
  --git-commit "$(git rev-parse HEAD)" \
  --device mps
```

Until the benchmark materialization is reviewed, generated run metadata is explicitly
marked `experimental-unreviewed`; the runner does not convert it into a public claim.

Compare both complete run directories only after they use byte-identical labels,
preprocessed variants, candidate registry, and benchmark lock:

```bash
nguven-eval compare-image-models \
  --run image/predictions/siglip2-syncred-v1 \
  --run image/predictions/vit-syncred-v1 \
  --output image/comparisons/private/image-origin-robustness-v1.json
```

The comparison checks all transformation slices and applies the frozen order: worst
transformation Macro F1, overall Macro F1, high-confidence false-positive rate, then
P95 latency. A candidate that fails any acceptance target cannot be selected, and a
source-locked benchmark can report only an `experimental-leader`.

## Turkish text benchmark source lock

`benchmarks/text-origin-tr-v1.json` pins reviewed candidate sources rather than
committing third-party text. It requires a balanced 12,000-record target, at least two
synthetic generator families, deterministic 80/10/10 grouping, and explicit license and
revision metadata. The committed release records the reviewed private materialization
hashes and non-textual sampling summary. Result evidence is enabled only for artifacts
that bind those exact manifest and preprocessing hashes.

```bash
nguven-eval validate-benchmark benchmarks/text-origin-tr-v1.json
```

This source lock is a reproducibility and governance artifact, not a measured dataset
release or a performance result.

The materializer downloads only the declared revisions, verifies every file SHA-256,
removes hidden reasoning from the MiniMax source, excludes Qwen source passages,
deduplicates exact content, and creates balanced per-source splits. Human excerpts are
length-matched to the synthetic distribution to reduce a trivial length cue. Raw text
stays local and is ignored by Git.

```bash
python -m pip install -e ".[materialization]"
nguven-eval materialize-benchmark benchmarks/text-origin-tr-v1.json \
  --source-cache benchmarks/cache/text-origin-tr-v1 \
  --output benchmarks/private/text-origin-tr-v1 \
  --accessed-at 2026-08-29T12:00:00+03:00 \
  --allow-network
```

Only the non-textual summary, exact artifact hashes, and reviewed source lock may be
published. The manifest, raw input, and preprocessed JSONL files remain private.

## Layout

```text
ml/evaluation/
├── comparisons/
│   ├── schema.json
│   └── text-origin-tr-v1.json
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

Private per-seed results belong under ignored storage. Only the reviewed, text-free
comparison and evidence hashes are published in the repository.

## Model artifact and adapter boundary

`models/schema.json` records immutable upstream model and tokenizer revisions, separate weight/code licenses, the required preprocessing version, runtime details, label mapping, artifact sizes, and SHA-256 hashes. Local weights and caches under `models/` are ignored by Git.

The adapter registry pins the reviewed upstream identities of
`dbmdz/bert-base-turkish-cased` and `ytu-ce-cosmos/modernbert-tr-base`. The completed
comparison selects BERTurk for prototype integration only; neither local weight bundle
is committed or approved for production. Adapters accept only a locally supplied
inference backend and never download code or weights.

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

This repository does not pretrain either model. The two pretrained candidates were
fine-tuned under one immutable plan and evaluated on the same untouched test records.
The commands below document how the reviewed run is reproduced.

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
  --max-sequence-length 128 \
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
nguven-eval validate-training-environment experiments/training-environment-v1.json
nguven-eval train-text-model path/to/package/training \
  --plan finetuning/plans/text-origin-tr-v1.json \
  --experiment experiments/berturk-v1.json \
  --benchmark benchmarks/text-origin-tr-v1.json \
  --environment-lock experiments/training-environment-v1.json \
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

The BERTurk baseline is frozen in `experiments/berturk-v1.json`: upstream revision,
three seeds, both adaptation stages, sequence length, and the report's acceptance
targets are explicit. Its status is `completed`; its validation-only execution evidence
is hash-bound under `experiments/evidence/berturk-v1.json`.

```bash
nguven-eval validate-experiment experiments/berturk-v1.json
```

ModernBERT-TR is also `completed` and independently hash-bound. The validation command
continues to enforce byte-for-byte comparability on benchmark, seeds, training stages,
sequence length, and acceptance thresholds:

```bash
nguven-eval validate-experiment-pair \
  --experiment experiments/berturk-v1.json \
  --experiment experiments/modernbert-tr-v1.json
```

`experiments/training-environment-v1.json` freezes Python, PyTorch, Transformers,
Accelerate, tokenizer, operating-system, MPS device, and M2 Pro hardware metadata.
Every execution manifest binds the hashes of the experiment, benchmark, plan, and
environment lock. The reviewed MPS compute budget uses one complete epoch, a 128-token
input ceiling, and an on-device batch of 32 on the 16 GB host. The plan explicitly uses
Transformers `eager` attention because the
reviewed PyTorch MPS runtime does not implement training dropout in its SDPA path.

## Calibration and final evidence

Temperature scaling is fitted on validation predictions only. The resulting private
artifact is bound to the validation manifest and prediction hashes:

```bash
nguven-eval fit-temperature-calibration path/to/validation-manifest.jsonl \
  --predictions path/to/validation-predictions.jsonl \
  --model-name berturk \
  --model-version berturk-text-origin-v1 \
  --output ml/evaluation/calibration/runs/berturk-seed-17.json
```

Pass that artifact only when evaluating the untouched test split. The result adds
PR-AUC, Brier score, ECE, the frozen 0.80-confidence false-positive rate, P95 latency,
and source, generator, and transformation slices. `compare-models` preserves every
seed, reports population standard deviation and a two-sided 95% t interval, and checks
the report targets (Macro F1 >= 0.80, high-confidence false-positive rate <= 0.05,
P95 text latency <= 3000 ms). Missing advanced metrics produce
`insufficient-evidence`, never an implied pass.

The completed comparison reports equal mean Macro F1 (`0.9997222220`) and accuracy
(`0.9997222222`). BERTurk wins the declared third tie-breaker with `38.4120 ms` mean
inference time versus `55.2437 ms`; its mean p95 is `45.8162 ms` versus `66.2645 ms`.
Both candidates pass all frozen targets. BERTurk also records lower mean Brier score,
ECE, and high-confidence false-positive rate. These results are limited to the reviewed
source families; near-perfect performance is treated as a shortcut-learning warning,
not a broad detector claim.

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
