FROM python:3.11.17-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_FORMAT=json \
    PYTHONPATH=/opt/pypirisk:/opt/pypirisk/services/frontend

RUN useradd --create-home --uid 1000 app

WORKDIR /opt/pypirisk
COPY requirements/frontend.txt requirements/
RUN pip install -r requirements/frontend.txt
COPY pypirisk ./pypirisk
COPY services/frontend ./services/frontend

USER app
EXPOSE 8501 9101
CMD ["streamlit", "run", "services/frontend/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
