FROM python:3.11.17-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_FORMAT=json \
    PYTHONPATH=/opt/pypirisk:/opt/pypirisk/services/api

RUN useradd --create-home --uid 1000 app

WORKDIR /opt/pypirisk
COPY requirements/api.txt requirements/
RUN pip install -r requirements/api.txt
COPY pypirisk ./pypirisk
COPY services/api ./services/api

USER app
EXPOSE 8000
CMD ["uvicorn", "--factory", "pypirisk_api.main:create_app_from_env", "--host", "0.0.0.0", "--port", "8000"]
