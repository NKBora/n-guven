# N-Güven Text AI Service

FastAPI service for the calibrated Turkish text AI-generation prototype signal. The HTTP
contract keeps a safe fallback when no private model bundle is mounted.

**Current status: calibrated BERTurk prototype runtime implemented.**

BERTurk was selected over ModernBERT-TR for prototype integration by the frozen
three-seed comparison. The service release registry pins the validation-selected
seed-17 model manifest, weights, calibration, and threshold identities by SHA-256.
The 442 MB weights remain outside Git and must be supplied through the approved runtime
artifact channel. The service verifies the registered model manifest, calibration, and
every declared artifact before loading local safetensors without remote code execution.

The registered release is not production approval. Its near-perfect validation result
may contain source/style shortcuts and still requires external-domain evidence.

## Contract

- `GET /health` reports local service health.
- `POST /v1/analyze/text` validates an analysis identifier and Turkish text, then returns the stable analysis response contract.
- Without `TEXT_AI_MODEL_ROOT`, the safe fallback returns `score = null`,
  `confidenceLevel = "UNAVAILABLE"`, and unconfigured version identifiers.
- With a verified artifact root, `score` is the validation-calibrated probability of the
  `synthetic` class. Scores at or below `0.2` are `LOW`, scores at or above `0.8` are
  `HIGH`, and the middle band is `UNCERTAIN`.
- `inferenceMs` measures the local model call in the configured path; the safe fallback
  retains the explicit value `0`.

The service exposes only possible AI/synthetic-generation signals. It does not assess claims and does not make moderation decisions.

## Local setup

Python 3.12 is required.

```bash
cd services/text-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Run the API locally:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

The maximum accepted text length defaults to 10,000 Unicode characters. Override it before process startup when a different local contract limit is required:

```bash
TEXT_AI_MAX_TEXT_LENGTH=12000 uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The registered runtime truncates tokenized input to the frozen 128-token model limit.
Raw request text is still bounded by the API character limit before tokenization.

## Local model runtime

Install the optional inference dependencies and mount the private release bundle:

```bash
python -m pip install -e ".[inference]"
TEXT_AI_MODEL_ROOT=/secure/path/berturk-text-origin-v1 \
  uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The directory must contain `model-manifest.json`, `calibration.json`, and every file
declared by the manifest. Any missing file, symbolic link, size mismatch, hash mismatch,
model identity mismatch, or calibration mismatch fails closed before model use.

## Docker

From the repository root:

```bash
docker build -t nguven-text-ai services/text-ai
docker run --rm -p 8000:8000 nguven-text-ai
```

Then:

```bash
curl http://localhost:8000/health
```

The image uses Python 3.12 slim, installs runtime dependencies only, and runs the application as a non-root user.
