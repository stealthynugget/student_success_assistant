# syntax=docker/dockerfile:1.6
# Stage 1: build the React console
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Python service (FastAPI) that also serves the console at /
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 ssa
WORKDIR /app
COPY requirements-core.txt requirements.txt requirements.lock ./
ARG INSTALL_EXTRAS=true
# Exact versions from requirements.lock (same as the development venv, so pickled models load cleanly). The PyTorch CPU
# index supplies torch==x+cpu: the default Linux wheel pulls in ~6 GB of CUDA libraries the service never uses.
RUN if [ "$INSTALL_EXTRAS" = "true" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock; \
    else pip install -r requirements-core.txt -c requirements.lock; fi
# Bake the sentence-embedding model into the image: otherwise the first dense search in a new container downloads it
# (about 45 s cold start). Change EMBEDDING_MODEL to match EMBEDDING_MODEL_NAME if you override it.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN if [ "$INSTALL_EXTRAS" = "true" ]; then \
      python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" "$EMBEDDING_MODEL" \
      && chown -R ssa:ssa /opt/hf; fi
# The model is in the image: do not contact the Hugging Face Hub at run time (override with HF_HUB_OFFLINE=0).
ENV HF_HUB_OFFLINE=1
COPY --chown=ssa:ssa ssa ./ssa
COPY --chown=ssa:ssa scripts ./scripts
COPY --chown=ssa:ssa pyproject.toml VERSION ./
COPY --from=web --chown=ssa:ssa /web/dist ./frontend/dist
RUN mkdir -p /app/data /app/artifacts && chown -R ssa:ssa /app/data /app/artifacts
USER ssa
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1
# Workers: WEB_CONCURRENCY (uvicorn reads it; default 1). With more than one, set SHARED_STATE=sqlite so job records,
# rate limits and recent analyses are shared through artifacts/app (see docs/DEPLOYMENT.md).
CMD ["uvicorn", "ssa.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
