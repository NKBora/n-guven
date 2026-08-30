FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    TOKENIZERS_PARALLELISM=false

ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --home-dir /app --shell /usr/sbin/nologin app

COPY pyproject.toml README.md ./
COPY app ./app
COPY releases ./releases

RUN python -m pip install --index-url "$PYTORCH_CPU_INDEX_URL" "torch>=2.4,<3" \
    && python -m pip install ".[inference]"

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
