"""Run every DVC stage, in DAG order, on the offline fixture; then serve its output."""

import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from pypirisk_api.external import RegistryClient
from pypirisk_api.main import Services, create_app
from pypirisk_api.settings import Settings
from pypirisk_api.store import ServingStore

STAGES = [
    "pypirisk.ingest",
    "pypirisk.validate",
    "pypirisk.graph",
    "pypirisk.features",
    "pypirisk.labels",
    "pypirisk.models.train",
    "pypirisk.models.evaluate",
    "pypirisk.models.register",
    "pypirisk.models.score",
]


@pytest.fixture(scope="module")
def pipeline_dir(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("pipeline")
    # Always the offline fixture, whatever profile the workspace is set to.
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    (workdir / "params.yaml").write_text(yaml.safe_dump({**params, "profile": "fixture"}))
    shutil.copytree(REPO_ROOT / "data" / "fixture", workdir / "data" / "fixture")
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": f"sqlite:///{workdir / 'mlflow.db'}",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        # Stages run from a copy of the data, so point Python at the repository code.
        "PYTHONPATH": str(REPO_ROOT),
        "PIPELINE_LOG_DIR": str(workdir / "logs"),
    }
    for stage in STAGES:
        result = subprocess.run(
            [sys.executable, "-m", stage], cwd=workdir, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, f"{stage} failed:\n{result.stderr[-3000:]}"
    return workdir


def read(pipeline_dir, path):
    return json.loads((pipeline_dir / path).read_text())


def test_ingest_replays_the_fixture_with_full_coverage(pipeline_dir):
    report = read(pipeline_dir, "reports/ingest.json")
    assert report["coverage"]["packages"] == 50
    assert report["crawl"]["http_requests"] == 0
    assert report["coverage"]["share_with_commits"] > 0.9


def test_validation_rejects_few_rows(pipeline_dir):
    assert read(pipeline_dir, "reports/validation.json")["summary"]["invalid_fraction"] < 0.01


def test_features_cover_every_snapshot(pipeline_dir, params):
    features = pd.read_parquet(pipeline_dir / "data/features/features.parquet")
    expected = {
        *params["snapshots"]["train"],
        *params["snapshots"]["validation"],
        *params["snapshots"]["test"],
        params["snapshots"]["scoring"],
    }
    assert set(features["snapshot"]) == expected
    assert set(params["train"]["feature_columns"]) <= set(features.columns)


def test_labels_exist_for_every_split(pipeline_dir):
    labels = pd.read_parquet(pipeline_dir / "data/labels/labels.parquet")
    assert set(labels["split"]) == {"train", "validation", "test"}
    assert labels["label"].isin([0, 1]).all()


def test_metrics_compare_model_and_heuristic(pipeline_dir):
    metrics = read(pipeline_dir, "reports/metrics.json")
    assert {"test_rows", "test_prevalence"} <= set(metrics)


def test_first_model_is_promoted_and_scores_every_package(pipeline_dir):
    registry = read(pipeline_dir, "reports/registry.json")
    assert registry["promoted"] and registry["champion_version"] == "1"
    scores = pd.read_parquet(pipeline_dir / "data/serving/scores.parquet")
    assert len(scores) == 50 and scores["risk"].between(0, 1).all()


def test_every_stage_left_a_run_record(pipeline_dir):
    runs = [
        json.loads(line)
        for line in (pipeline_dir / "logs/pipeline_runs.jsonl").read_text().splitlines()
    ]
    assert [run["status"] for run in runs] == ["success"] * len(STAGES)


def test_api_serves_the_pipeline_output(pipeline_dir):
    settings = Settings(
        model_server_url="http://unused",
        serving_dir=pipeline_dir / "data/serving",
    )
    services = Services(ServingStore(settings.serving_dir), None, None, RegistryClient(None, 1))
    client = TestClient(create_app(settings, services))
    assert client.get("/packages/requests/risk").json()["package"] == "requests"
    scan = client.post("/projects/scan", json={"requirements": "requests\npandas\n"}).json()
    assert scan["dependencies_total"] >= 5
    assert client.get("/ecosystem/watch?limit=5").status_code == 200
