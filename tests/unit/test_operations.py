import json
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from pypirisk.models.registry import PREVIOUS_TAG, RollbackError, promote, rollback
from pypirisk.monitoring import drift
from pypirisk.ops.inject_drift import drifted
from pypirisk.propagate.project import parse_requirements, project_risk, resolve
from pypirisk.serving.ray_serve import records_to_frame
from pypirisk.stagelog import runs_path, stage_run


class FakeRegistry:
    def __init__(self, alias_version=None, tags=None):
        self.alias_version = alias_version
        self.tags = {version: dict(t) for version, t in (tags or {}).items()}

    def get_model_version_by_alias(self, name, alias):
        if self.alias_version is None:
            raise MlflowException("no alias")
        return SimpleNamespace(
            version=self.alias_version, tags=self.tags.get(self.alias_version, {})
        )

    def set_registered_model_alias(self, name, alias, version):
        self.alias_version = version

    def set_model_version_tag(self, name, version, key, value):
        self.tags.setdefault(version, {})[key] = value


def test_promotion_records_the_replaced_champion_and_rollback_restores_it():
    registry = FakeRegistry(alias_version="1")
    assert promote(registry, "m", "champion", "2") == "1"
    assert registry.tags["2"][PREVIOUS_TAG] == "1"
    assert rollback(registry, "m", "champion", "bad scores") == ("2", "1")
    assert registry.alias_version == "1"
    assert registry.tags["2"]["rolled_back"] == "bad scores"


def test_rollback_without_an_earlier_champion_is_refused():
    with pytest.raises(RollbackError):
        rollback(FakeRegistry(alias_version="1"), "m", "champion", "x")
    with pytest.raises(RollbackError):
        rollback(FakeRegistry(), "m", "champion", "x")


def test_drift_is_the_mean_shift_in_reference_standard_deviations():
    profile = drift.reference_profile(pd.DataFrame({"x": [1.0, 2.0, 3.0], "const": [5.0] * 3}))
    assert profile["x"] == {"mean": 2.0, "std": 1.0}
    shifted = drift.feature_drift(profile, pd.DataFrame({"x": [4.0, 5.0, 6.0], "const": [9.0] * 3}))
    assert shifted == {"x": 3.0}


def test_drift_ignores_missing_values():
    profile = drift.reference_profile(pd.DataFrame({"x": [1.0, 3.0, np.nan]}))
    assert drift.feature_drift(profile, pd.DataFrame({"x": [2.0, np.nan]}))["x"] == 0.0


def test_injected_drift_zeroes_activity_and_stretches_release_gaps():
    features = pd.DataFrame({"commits_in_lookback": [5.0], "months_since_last_release": [3.0]})
    shifted = drifted(features)
    assert shifted.iloc[0].tolist() == [0.0, 12.0]
    assert features.iloc[0, 0] == 5.0


def test_parse_requirements_skips_comments_options_and_reports_bad_lines():
    parsed = parse_requirements(
        "# deps\nRequests>=2  # http\n-r other.txt\nflask[async]\nrequests\n==bad\n"
    )
    assert parsed.packages == ["requests", "flask"]
    assert parsed.errors and parsed.errors[0].startswith("line 6")


def test_parse_requirements_rejects_huge_files():
    with pytest.raises(ValueError, match="At most"):
        parse_requirements("a\n" * 501)


EDGES = pd.DataFrame(
    {"source": ["app", "lib", "lib", "util"], "target": ["lib", "util", "core", "lib"]}
)


def test_resolve_reports_each_package_at_its_shallowest_depth_with_its_path():
    found = {d.name: d for d in resolve(["app", "util"], EDGES, max_depth=6)}
    assert found["util"].depth == 1
    assert found["core"].depth == 3
    assert found["core"].path == ("app", "lib", "core")


def test_resolve_respects_the_depth_limit():
    assert {d.name for d in resolve(["app"], EDGES, max_depth=2)} == {"app", "lib"}


def test_project_risk_is_the_weakest_link_discounted_by_depth():
    dependencies = resolve(["app"], EDGES, max_depth=6)
    result = project_risk(dependencies, {"app": 0.1, "lib": 0.2, "core": 0.9}, depth_decay=0.5)
    assert result.score == pytest.approx(0.9 * 0.25)
    assert result.expected_abandoned == pytest.approx(1.2)
    assert result.unscored == ["util"]
    assert result.contributions[0]["package"] == "core"


def test_project_risk_does_not_grow_with_the_number_of_safe_dependencies():
    edges = pd.DataFrame(columns=["source", "target"])
    few = project_risk(resolve(["a"], edges, 3), {"a": 0.3}, 0.5).score
    many_names = ["a", *[f"safe{i}" for i in range(100)]]
    many = project_risk(
        resolve(many_names, edges, 3), {"a": 0.3} | {n: 0.05 for n in many_names[1:]}, 0.5
    ).score
    assert few == many


def test_stage_run_records_success_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path / "logs"))
    with stage_run("graph", "fixture"):
        pass
    with pytest.raises(ZeroDivisionError), stage_run("graph", "fixture"):
        raise ZeroDivisionError("boom")
    runs = [json.loads(line) for line in runs_path().read_text().splitlines()]
    assert [run["status"] for run in runs] == ["success", "failed"]
    assert "ZeroDivisionError" in runs[1]["error"]
    assert "Traceback" in runs[1]["traceback"]


def test_ray_serve_accepts_both_mlflow_input_formats():
    records = records_to_frame({"dataframe_records": [{"a": 1}]})
    split = records_to_frame({"dataframe_split": {"columns": ["a"], "data": [[1]]}})
    pd.testing.assert_frame_equal(records, split)
    with pytest.raises(ValueError):
        records_to_frame({"inputs": []})


def test_model_server_waits_until_a_model_is_registered(monkeypatch):
    from pypirisk.serving import model_server

    attempts = []

    def load(uri):
        attempts.append(uri)
        if len(attempts) < 3:
            raise MlflowException("no alias yet")
        return "model"

    monkeypatch.setattr(model_server.mlflow.pyfunc, "load_model", load)
    monkeypatch.setattr(model_server.time, "sleep", lambda seconds: None)
    assert model_server.load_model_when_available("models:/m@champion", 0) == "model"
    assert len(attempts) == 3


def test_frontend_ops_server_serves_health_and_metrics():
    from pypirisk_frontend.ops_server import start_ops_server

    server = start_ops_server(port=0, host="127.0.0.1")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/health") as response:
            assert json.loads(response.read()) == {"status": "ok"}
        with urllib.request.urlopen(f"{base}/metrics") as response:
            assert b"python_info" in response.read()
    finally:
        server.shutdown()


def test_junit_results_are_counted_per_suite(tmp_path):
    from pypirisk.ops.quality import count_results

    junit = tmp_path / "junit.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.unit.test_api" name="a"/>'
        '<testcase classname="tests.unit.test_api" name="b"><failure/></testcase>'
        '<testcase classname="tests.leakage.test_no_future_data" name="c"/>'
        "</testsuite></testsuites>"
    )
    counts = count_results(junit)
    assert counts["unit"] == {"passed": 1, "failed": 1}
    assert counts["leakage"] == {"passed": 1}


def test_ci_stages_write_counts_and_stop_the_pipeline_on_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pypirisk.ops import quality

    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(
        quality, "_run", lambda command: SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    assert quality.lint() == {"ruff_check_exit": 0, "ruff_format_exit": 0}

    monkeypatch.setattr(
        quality, "_run", lambda command: SimpleNamespace(returncode=1, stdout="E501", stderr="")
    )
    with pytest.raises(quality.CheckFailedError, match="ruff found problems"):
        quality.lint()

    def failing_pytest(command):
        quality.JUNIT_PATH.write_text(
            '<testsuites><testsuite><testcase classname="tests.unit.t" name="a">'
            "<failure/></testcase></testsuite></testsuites>"
        )
        return SimpleNamespace(returncode=1, stdout="1 failed", stderr="")

    monkeypatch.setattr(quality, "_run", failing_pytest)
    with pytest.raises(quality.CheckFailedError, match="1 tests failed"):
        quality.test()
    assert json.loads(quality.TEST_REPORT.read_text())["total_failed"] == 1


def test_plot_points_are_thinned_and_named():
    from pypirisk.models.evaluate import curve_points

    points = curve_points(
        np.linspace(0, 1, 1000), np.linspace(1, 0, 1000), "recall", "precision", limit=50
    )
    assert len(points) == 50
    assert points[0] == {"recall": 0.0, "precision": 1.0}
