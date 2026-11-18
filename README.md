# AI Operations (AIOps): End-semester project, pypirisk

pypirisk predicts the probability that an actively maintained PyPI package is abandoned within the next 18 months and explains the score. It then pushes that risk through the dependency graph to grade a whole project from its `requirements.txt`.

Demo video: <https://drive.google.com/file/d/1sy-6wYKyeqN-yousTmwWelIz8BVllXlY/view?usp=sharing>

Report: [report.pdf](report.pdf)

Documents in `docs/`:

- [HLD.md](docs/HLD.md): architecture diagram with its blocks, and the design choices with their reasons.
- [LLD.md](docs/LLD.md): every API endpoint with its input and output, and each pipeline stage's files.
- [test_plan.md](docs/test_plan.md): test levels, acceptance criteria, test cases and the test report.
- [user_manual.md](docs/user_manual.md): how a non-technical user uses the four pages.

Structure of the repository:

```
pypirisk/                     pipeline code, one folder per DVC stage
  ingest/                     Ray crawler: rate-limit actor, response cache, PyPI / ecosyste.ms / git sources
  validate/                   Ray Data validation
  graph/ features/ labels/    dependency graph per snapshot, 11 features, abandonment labels
  models/                     train, evaluate, register (promotion gate + rollback), batch score
  serving/                    MLflow model server with /metrics, Ray Serve deployment
  monitoring/                 feature drift
  ops/                        CI checks (lint, test) and the drift / broken-model demos
services/api/                 FastAPI backend (Dockerfile + pypirisk_api/)
services/frontend/            Streamlit frontend: 4 pages, /metrics on :9101
dvc.yaml, dvc.lock            the pipeline DAG (also the CI) and its hashes
params.yaml                   every tunable: data profile, snapshots, thresholds, hyperparameters
MLproject                     MLflow entry points, run in the pipeline image
Dockerfile                    pipeline image (also runs MLflow, model-server, ray-serve)
docker-compose.yml            all services
monitoring/                   prometheus.yml, alert_rules.yml, alertmanager.yml, grafana/dashboard.json
data/fixture/cache/           50 packages of real responses, replayed offline by CI
reports/                      DVC metrics and plots written by the stages
tests/                        unit, leakage, integration, e2e
notebooks/eda.ipynb           exploratory data analysis
examples/requirements-demo.txt  sample project for Project Scan
docs/                         HLD, LLD, test plan and report, user manual
```

| Service | Port |
|---|---|
| Frontend (Streamlit) | 8501 |
| Frontend metrics (`/metrics`, `/health`) | 9101 |
| API (FastAPI, docs at `/docs`) | 8000 |
| Model server (MLflow scoring) | 5001 |
| Ray Serve | 8002 |
| MLflow | 5000 |
| SeaweedFS (DVC remote, S3) | 8333 |
| Prometheus | 9090 |
| Alertmanager | 9093 |
| Grafana | 3000 |
| node_exporter | 9100 |

## Environment

You need Docker with Compose. For running DVC on the host, you also need:

```bash
conda create -n pypirisk python=3.11 -y
conda activate pypirisk
pip install -r requirements/dev.txt
```

Then copy the env file and set the passwords and keys:

```bash
cp .env.example .env
```

`.env` is git-ignored. Run every command from the repository root.

## Running

1. Start all services:

   ```bash
   docker compose up -d --build
   ```

   `docker compose ps` shows them healthy after about a minute. The model server and Ray Serve wait until a model is registered.

2. Run the pipeline (CI checks, then data, training, registration and scoring):

   ```bash
   docker compose run --rm pipeline
   ```

   The committed profile is `fixture` (50 packages, offline, a few minutes). For real data, set `profile: pilot` (500 packages plus dependencies) or `profile: full` (5,000) in `params.yaml`. A full crawl takes hours because of the ecosyste.ms rate limit. It can resume, because every response is cached in `data/cache`.

3. Restart the model server, so it loads the new `@champion`:

   ```bash
   docker compose restart model-server ray-serve
   ```

4. Open <http://localhost:8501>.

5. Set up Grafana at <http://localhost:3000> (user `admin`, password `GRAFANA_ADMIN_PASSWORD` from `.env`):
   - Add a Prometheus data source with URL `http://prometheus:9090`.
   - Go to Dashboards → New → Import and choose `monitoring/grafana/dashboard.json`.

Check: <http://localhost:9090/targets> shows all ten targets as UP. The frontend's target comes up after the first page load.

## CI and versioning (DVC)

```bash
dvc repro                                  # lint -> test -> ingest -> ... -> score, only what changed
dvc dag                                    # the pipeline as a graph
dvc metrics show / dvc metrics diff        # metrics, and the change since the last commit
dvc params diff / dvc plots diff           # changed parameters, PR and calibration curves
dvc exp run -S profile=pilot               # an experiment on real data
dvc push / dvc pull                        # data and models to / from SeaweedFS
```

If `lint` or `test` fails, `dvc repro` stops before `ingest`. The failure shows in the Pipeline Console and in `reports/lint.json` / `reports/tests.json`.

For `dvc push`, put the S3 keys in the git-ignored `.dvc/config.local`:

```bash
dvc remote modify --local storage access_key_id "$S3_ACCESS_KEY"
dvc remote modify --local storage secret_access_key "$S3_SECRET_KEY"
```

## Retraining, drift and rollback

```bash
dvc repro                                                                   # retrain
docker compose run --rm pipeline dvc repro --single-item --force score      # re-score
docker compose run --rm pipeline python -m pypirisk.ops.inject_drift        # drift demo
docker compose run --rm pipeline python -m pypirisk.ops.broken_model_demo   # rollback demo
```

- Drift demo: within a minute the largest feature drift in Grafana goes above 1. `FeatureDriftDetected` then fires in Prometheus and shows in Alertmanager. Re-scoring clears it.
- Rollback demo: it promotes a model that returns NaN. The post-deploy check fails and moves `@champion` back, and the broken version is tagged `rolled_back`.
- Manual rollback: in MLflow, go to Models → `pypirisk-abandonment` and move the `champion` alias to the previous version. Then restart `model-server`.

## Metrics

| Type | Metric | What it tracks |
|---|---|---|
| Counter | `http_requests_total` | requests per service, route and status (api, model-server, ray-serve) |
| Histogram | `http_request_duration_seconds` | request latency per service and route |
| Gauge | `http_requests_in_progress` | requests in flight |
| Histogram | `api_risk_score` | risk scores served, by source (batch / live) |
| Summary | `api_scan_dependencies` | dependencies per project scan |
| Counter | `api_unscored_lookups_total` | lookups of packages outside the scored set |
| Histogram | `api_model_server_request_duration_seconds` | API wait on each serving backend |
| Gauge | `pypirisk_feature_drift` | drift per feature, in standard deviations of the reference |
| Gauge | `api_scored_packages`, `api_batch_model_version` | size and model version of the batch scores |
| Info | `model_server_model_info` | model URI and run served by the model server |
| Counter | `frontend_api_calls_total` | frontend calls to the API, by endpoint and outcome |
| Histogram | `frontend_api_call_duration_seconds` | duration of those calls |

MLflow, SeaweedFS, Prometheus, Alertmanager, Grafana and node_exporter export their own metrics.

## Alerts

| Alert | Severity | Fires when |
|---|---|---|
| `ServiceDown` | critical | a scrape target is unreachable for 1 min |
| `ApiHighErrorRate` | warning | more than 5% of API requests return 5xx for 5 min |
| `ApiSlowRequests` | warning | API p95 latency above 1 s for 5 min |
| `FeatureDriftDetected` | warning | the largest feature drift is above 1.0 for 1 min |
| `HostLowDiskSpace` | warning | less than 10% of `/` is free for 5 min |

## Tests

```bash
pytest                    # unit, leakage and integration (also the DVC test stage)
pytest -m e2e             # against the running stack
```

AI use is described in [AI_DISCLOSURE.md](AI_DISCLOSURE.md).
