# Low-level design

The request and response schemas are the Pydantic models in `services/api/pypirisk_api/schemas.py`. They are also published at `http://localhost:8000/docs`.

## API (`api`, port 8000)

| Endpoint | Input | Output | Errors |
| --- | --- | --- | --- |
| `GET /health` | - | `status`, `model_server`, `secondary_model_server`, `packages_scored`, `model_version` | - |
| `GET /metrics` | - | Prometheus text | - |
| `GET /packages/{name}/risk` | `live` (bool), `backend` (`mlflow` / `ray`) | `package`, `risk` (0-1), `grade` (A-F), `top_factors`, `source`, `backend`, `model_version` | 404 not scored, 400 Ray not configured, 502 model server failed, 503 no scores yet |
| `GET /packages/{name}` | - | risk fields + `repo_url`, activity features, `dependencies`, `dependents_in_universe` | 404, 503 |
| `POST /projects/scan` | `requirements` (text, <= 100,000 chars) | `grade`, `score`, `expected_abandoned`, `contributions` (paths), `unscored`, `edges` | 422, 503 |
| `GET /ecosystem/watch` | `limit` (1-500), `min_dependents` | list of `package`, `risk`, `grade`, `dependents`, `priority` | 422, 503 |
| `GET /pipeline/dag` | - | `stages` (name, cmd, outputs, last run), `edges` | - |
| `GET /pipeline/runs` | `limit`, `status` | stage runs, newest first, with `error` and `traceback` on failure | 422 |
| `GET /pipeline/reports` | - | ingest, validation, metrics and registry reports | - |
| `GET /models` | - | registry `versions` (version, run, aliases, tags) and the batch model | 502 |

## Model servers

| Service | Endpoint | Input | Output |
| --- | --- | --- | --- |
| `model-server` (5001) and `ray-serve` (8002) | `POST /invocations` | `{"dataframe_records": [{feature: value}]}`, 11 optional doubles | `{"predictions": [{"risk", "top_factors"}]}` |
| both | `GET /health`, `GET /metrics` | - | status, Prometheus text |

The frontend (8501) has `/health` and `/metrics` on port 9101.

## Pipeline stages

| Stage | Reads | Writes |
| --- | --- | --- |
| lint, test | code, tests | `reports/lint.json`, `reports/tests.json` |
| ingest | test report, cache, public APIs | `data/raw/*.parquet`, `reports/ingest.json` |
| validate | `data/raw` | `data/validated`, `reports/validation.json` |
| graph | `data/validated` | `data/graph` |
| features | `data/validated`, `data/graph` | `data/features/features.parquet` |
| label | `data/validated` | `data/labels/labels.parquet` |
| train | features, labels | MLflow run and model, `models/train_run.json` |
| evaluate | model, features, labels | `reports/metrics.json`, `reports/plots/`, `models/reference.json` |
| register | run, metrics | registry version and alias, `reports/registry.json` |
| score | `@champion`, features, graph | `data/serving/` |
