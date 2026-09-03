# ---------------------------------------------------------------------------
# NovaPay Fraud Detection API — production image
# Multi-stage build: compile deps in a builder layer, ship a slim runtime
# image with no build toolchain, no source of the training pipeline's dev
# dependencies (pytest, jupyter, etc.), and a non-root user.
# ---------------------------------------------------------------------------

# ---- builder -----------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime -------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 is required at runtime by xgboost/lightgbm (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY api/ ./api/
COPY pyproject.toml .
COPY artifacts/ ./artifacts/

RUN pip install --no-cache-dir --no-deps -e . \
    && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
