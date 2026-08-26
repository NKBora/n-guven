# N-Güven Text AI Service

FastAPI foundation for the Turkish text AI-generation signal service. The HTTP contract, validation, and inference boundary are implemented; the inference adapter is currently a deterministic, model-independent stub.

**Current status: service foundation only.**

**No ML model is currently loaded.**

**No measured model accuracy exists yet.**

ModernBERT-TR and BERTurk are future model candidates. Neither is implemented, selected, or benchmarked by this service.

## Contract

- `GET /health` reports local service health.
- `POST /v1/analyze/text` validates an analysis identifier and Turkish text, then returns the stable analysis response contract.
- The current stub always returns `score = null`, `confidenceLevel = "UNAVAILABLE"`, and unconfigured version identifiers.
- `inferenceMs = 0` is a stub contract value, not a measured model inference latency.

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

No tokenizer limit is applied because no model or tokenizer is loaded.

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
