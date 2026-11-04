import json

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pypirisk_api import external, pipeline
from pypirisk_api.main import Services, create_app
from pypirisk_api.model_client import ModelServerClient, ModelServerError
from pypirisk_api.settings import MissingSettingError, Settings
from pypirisk_api.store import ServingStore

FACTORS = [{"feature": "months_since_last_release", "value": 30.0, "contribution": 1.2}]


def write_serving(directory):
    directory.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(
        {
            "name": ["app", "lib", "core"],
            "snapshot": ["2026-11-01"] * 3,
            "months_since_last_release": [2.0, 20.0, 40.0],
            "releases_in_lookback": [3.0, 0.0, 0.0],
            "commits_in_lookback": [50.0, 1.0, None],
            "bus_factor": [3.0, 1.0, None],
            "dependents": [0.0, 1.0, 2.0],
            "pagerank": [0.5, 1.0, 2.0],
            "repo_url": ["https://github.com/o/app", None, None],
            "risk": [0.05, 0.4, 0.9],
            "grade": ["A", "D", "F"],
            "top_factors": [json.dumps(FACTORS)] * 3,
        }
    )
    scores.to_parquet(directory / "scores.parquet")
    pd.DataFrame(
        {"name": ["app", "lib", "core"], "snapshot": ["2026-11-01"] * 3, "x": [1.0, None, 3.0]}
    ).to_parquet(directory / "features.parquet")
    pd.DataFrame(
        {"source": ["app", "lib"], "target": ["lib", "core"], "snapshot": ["2026-11-01"] * 2}
    ).to_parquet(directory / "edges.parquet")
    (directory / "model.json").write_text(json.dumps({"version": "4", "snapshot": "2026-11-01"}))
    reference = {"x": {"mean": 1.0, "std": 1.0}}
    (directory / "reference.json").write_text(json.dumps(reference))


class FakeModel:
    def __init__(self, fail=False, healthy=True):
        self.fail, self.healthy, self.received = fail, healthy, None

    def predict(self, features):
        if self.fail:
            raise ModelServerError("down")
        self.received = features
        return 0.33, FACTORS

    def is_healthy(self):
        return self.healthy


class FakeRegistry:
    def versions(self, name):
        return [{"version": "4", "aliases": ["champion"], "tags": {}}]


@pytest.fixture
def api(tmp_path):
    write_serving(tmp_path / "serving")
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  a:\n    cmd: x\n    outs: [data/a]\n  b:\n    cmd: y\n    deps: [data/a/file]\n"
    )
    (tmp_path / "runs.jsonl").write_text(
        json.dumps({"stage": "a", "status": "failed", "error": "boom"}) + "\n"
    )
    settings = Settings(
        model_server_url="http://m",
        serving_dir=tmp_path / "serving",
        dvc_yaml_path=tmp_path / "dvc.yaml",
        reports_dir=tmp_path / "reports",
        pipeline_runs_path=tmp_path / "runs.jsonl",
    )
    primary = FakeModel()
    services = Services(ServingStore(settings.serving_dir), primary, FakeModel(), FakeRegistry())
    return TestClient(create_app(settings, services)), settings, primary


def test_batch_risk_uses_canonical_names_and_returns_factors(api):
    client, *_ = api
    body = client.get("/packages/Core/risk").json()
    assert (body["package"], body["risk"], body["grade"], body["source"]) == (
        "core",
        0.9,
        "F",
        "batch",
    )
    assert body["top_factors"] == FACTORS
    assert body["model_version"] == "4"


def test_live_risk_sends_model_inputs_without_nan(api):
    client, _, primary = api
    body = client.get("/packages/lib/risk?live=true").json()
    assert (body["source"], body["risk"], body["grade"]) == ("live", 0.33, "C")
    assert primary.received == {"x": None}


def test_live_risk_returns_502_when_the_model_server_fails(api):
    client, _, primary = api
    primary.fail = True
    assert client.get("/packages/lib/risk?live=true").status_code == 502


def test_unknown_package_is_404(api):
    client, *_ = api
    assert client.get("/packages/flask/risk").status_code == 404


def test_package_detail_includes_graph_neighbours(api):
    client, *_ = api
    body = client.get("/packages/lib").json()
    assert body["dependencies"] == ["core"]
    assert body["dependents_in_universe"] == ["app"]
    assert body["bus_factor"] == 1.0


def test_project_scan_propagates_risk_through_the_graph(api, monkeypatch):
    client, *_ = api
    monkeypatch.setattr("pypirisk_api.main.heuristic_fallback", lambda *a: 14.0)
    body = client.post("/projects/scan", json={"requirements": "app\nunknown-pkg\n"}).json()
    assert body["score"] == pytest.approx(0.9 * 0.25)
    assert body["contributions"][0]["path"] == ["app", "lib", "core"]
    assert body["unscored"] == [{"package": "unknown-pkg", "months_since_last_release": 14.0}]
    assert sorted(map(tuple, body["edges"])) == [("app", "lib"), ("lib", "core")]


def test_project_scan_rejects_files_without_packages(api):
    client, *_ = api
    assert (
        client.post("/projects/scan", json={"requirements": "# nothing\n-r x.txt\n"}).status_code
        == 422
    )


def test_ecosystem_watch_ranks_by_risk_times_centrality(api):
    client, *_ = api
    names = [row["package"] for row in client.get("/ecosystem/watch?limit=2").json()]
    assert names == ["core", "lib"]
    assert client.get("/ecosystem/watch?min_dependents=2").json()[0]["package"] == "core"


def test_pipeline_dag_links_stages_and_attaches_last_runs(api):
    client, *_ = api
    dag = client.get("/pipeline/dag").json()
    assert dag["edges"] == [{"from": "a", "to": "b"}]
    assert dag["stages"][0]["last_run"]["status"] == "failed"
    assert client.get("/pipeline/runs?status=failed").json()[0]["error"] == "boom"


def test_models_lists_registry_versions(api):
    client, *_ = api
    assert client.get("/models").json()["versions"][0]["aliases"] == ["champion"]


def test_health_is_degraded_without_batch_scores(tmp_path):
    settings = Settings(model_server_url="http://m", serving_dir=tmp_path / "empty")
    services = Services(ServingStore(settings.serving_dir), FakeModel(), None, FakeRegistry())
    client = TestClient(create_app(settings, services))
    body = client.get("/health").json()
    assert (body["status"], body["secondary_model_server"]) == ("degraded", "not configured")
    assert client.get("/packages/app/risk").status_code == 503


def test_metrics_include_drift_and_route_templates(api):
    client, *_ = api
    client.get("/packages/app/risk")
    metrics = client.get("/metrics").text
    assert 'pypirisk_feature_drift{feature="x"}' in metrics
    assert 'route="/packages/{name}/risk"' in metrics


def test_store_reloads_when_new_scores_land(api, tmp_path):
    client, settings, *_ = api
    client.get("/health")
    model = json.loads((settings.serving_dir / "model.json").read_text())
    (settings.serving_dir / "model.json").write_text(json.dumps({**model, "version": "5"}))
    import os

    os.utime(settings.serving_dir / "model.json", (1e10, 1e10))
    assert client.get("/health").json()["model_version"] == "5"


def test_settings_require_the_model_server_url(monkeypatch):
    monkeypatch.delenv("MODEL_SERVER_URL", raising=False)
    monkeypatch.setenv("SERVING_DIR", "/data")
    with pytest.raises(MissingSettingError, match="MODEL_SERVER_URL"):
        Settings.from_env()


def test_read_runs_returns_newest_first(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps({"stage": s, "status": "success"}) + "\n" for s in "abc"))
    assert [run["stage"] for run in pipeline.read_runs(path, limit=2)] == ["c", "b"]
    assert pipeline.last_run_per_stage(path)["a"]["status"] == "success"


def model_client(handler):
    client = ModelServerClient("http://m", timeout=1)
    client._client = httpx.Client(base_url="http://m", transport=httpx.MockTransport(handler))
    return client


def test_model_client_parses_risk_and_factors():
    def handler(request):
        assert json.loads(request.content) == {"dataframe_records": [{"x": None}]}
        return httpx.Response(
            200, json={"predictions": [{"risk": 0.2, "top_factors": json.dumps(FACTORS)}]}
        )

    assert model_client(handler).predict({"x": float("nan")}) == (0.2, FACTORS)


def test_model_client_raises_on_errors_and_bad_bodies():
    with pytest.raises(ModelServerError, match="request failed"):
        model_client(lambda r: httpx.Response(500)).predict({})
    with pytest.raises(ModelServerError, match="Unexpected"):
        model_client(lambda r: httpx.Response(200, json={"oops": 1})).predict({})


def test_registry_client_attaches_aliases_from_the_registered_model(monkeypatch):
    def fake_get(url, params, timeout):
        if url.endswith("model-versions/search"):
            body = {
                "model_versions": [
                    {"version": "1", "creation_timestamp": 0},
                    {"version": "2", "creation_timestamp": 0},
                ]
            }
        else:
            body = {"registered_model": {"aliases": [{"alias": "champion", "version": "2"}]}}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(external.httpx, "get", fake_get)
    versions = external.RegistryClient("http://mlflow", 1).versions("m")
    assert [(v["version"], v["aliases"]) for v in versions] == [("2", ["champion"]), ("1", [])]
