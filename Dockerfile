# Pipeline image: runs the DVC pipeline (CI checks and containerized training), the
# MLflow tracking server and the model servers in one environment.
FROM python:3.11.17-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_FORMAT=json \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONPATH=/opt/pypirisk

# git: the train stage logs the commit hash. The repo is bind-mounted with a different
# owner than the container user, so git must be told to trust it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*' \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /mlflow /workspace \
    && chown app:app /mlflow /workspace

WORKDIR /opt/pypirisk
# dev.txt = pipeline + api + frontend + ruff and pytest: `dvc repro` runs the CI stages
# (lint, test) in this image too, so development, CI and training share one environment.
COPY requirements/ requirements/
RUN pip install -r requirements/dev.txt
# LightGBM needs the OpenMP runtime, which the slim base image does not ship.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pypirisk ./pypirisk

USER app
WORKDIR /workspace
CMD ["dvc", "repro"]
