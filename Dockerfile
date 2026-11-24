FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY gateway ./gateway
COPY tests ./tests
COPY scripts/download_model.py ./scripts/download_model.py
ENV PYTHONUNBUFFERED=1 MEM0_TELEMETRY=false FASTEMBED_CACHE_PATH=/data/embeddings
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
