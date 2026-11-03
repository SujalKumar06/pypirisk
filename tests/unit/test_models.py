import json

import numpy as np
import pandas as pd
import pytest

from pypirisk.grades import grade
from pypirisk.models.estimators import build_candidate, heuristic_score, join_features_and_labels
from pypirisk.models.explain import mean_absolute_shap, shap_values, top_factors
from pypirisk.models.metrics import ranking_metrics, top_fraction_mask
from pypirisk.models.pyfunc import RiskModel
from pypirisk.models.register import decide
from pypirisk.models.train import Split, calibrate, model_signature, select_candidate

LGBM = {
    "n_estimators": 20,
    "learning_rate": 0.1,
    "num_leaves": 4,
    "min_child_samples": 2,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}
LOGREG = {"C": 1.0, "max_iter": 500}


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    features.loc[::17, "y"] = np.nan
    labels = (features["x"] + rng.normal(scale=0.5, size=200) > 0.8).astype(int).to_numpy()
    return features, labels


@pytest.mark.parametrize(
    ("kind", "settings"), [("logistic_regression", LOGREG), ("lightgbm", LGBM)]
)
def test_shap_values_add_up_to_the_models_log_odds(kind, settings, data):
    features, labels = data
    model = build_candidate(kind, settings, seed=0).fit(features, labels)
    shap = shap_values(kind, model, features)
    if kind == "lightgbm":
        expected = (
            model.predict(features, raw_score=True)
            - model.predict(features, pred_contrib=True)[:, -1]
        )
    else:
        expected = model.decision_function(features) - model[-1].intercept_[0]
    np.testing.assert_allclose(shap.sum(axis=1), expected, atol=1e-6)


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_candidate("magic", {}, 0)


def test_top_factors_rank_by_absolute_contribution():
    factors = top_factors(
        pd.Series({"a": 0.1, "b": -2.0, "c": 1.0}),
        pd.Series({"a": 1, "b": np.nan, "c": 3}),
        count=2,
    )
    assert [f["feature"] for f in factors] == ["b", "c"]
    assert factors[0]["value"] is None


def test_mean_absolute_shap_orders_features_by_importance():
    assert list(mean_absolute_shap(pd.DataFrame({"a": [0.1, -0.1], "b": [1.0, -2.0]})).index) == [
        "b",
        "a",
    ]


def test_risk_model_returns_probabilities_and_json_factors(data):
    features, labels = data
    model = build_candidate("lightgbm", LGBM, 0).fit(features, labels)
    calibrated, _ = calibrate(model, Split(features, features, labels), "sigmoid")
    output = RiskModel("lightgbm", model, calibrated, ["x", "y"]).predict(None, features.head(3))
    assert list(output.columns) == ["risk", "top_factors"]
    assert output["risk"].between(0, 1).all()
    assert len(json.loads(output["top_factors"].iloc[0])) == 2


def test_calibration_is_skipped_when_validation_has_one_class(data):
    features, labels = data
    model = build_candidate("logistic_regression", LOGREG, 0).fit(features, labels)
    single = Split(features.head(10), features.head(10), np.zeros(10, dtype=int))
    calibrated, status = calibrate(model, single, "isotonic")
    assert status.startswith("skipped")
    np.testing.assert_allclose(calibrated.predict_proba(features), model.predict_proba(features))


def test_candidate_selection_falls_back_to_the_first_when_scores_are_undefined():
    assert select_candidate({"a": 0.2, "b": 0.5}, ["a", "b"])[0] == "b"
    assert select_candidate({"a": None, "b": None}, ["a", "b"])[0] == "a"


def test_signature_makes_every_feature_optional():
    signature = model_signature(["x", "y"])
    assert all(not column.required for column in signature.inputs.inputs)
    assert signature.outputs.input_names() == ["risk", "top_factors"]


def test_ranking_metrics_are_skipped_for_a_single_class():
    assert ranking_metrics(np.array([0, 0]), np.array([0.1, 0.2]), "test") == {}
    assert ranking_metrics(np.array([0, 1]), np.array([0.1, 0.9]), "test") == {
        "test_roc_auc": 1.0,
        "test_pr_auc": 1.0,
    }


def test_top_fraction_flags_the_highest_scores_and_at_least_one_row():
    scores = np.array([0.9, 0.1, 0.2, 0.3])
    assert top_fraction_mask(scores, 0.25).tolist() == [True, False, False, False]
    assert top_fraction_mask(scores, 0.01).sum() == 1


def test_heuristic_score_is_months_since_last_release():
    assert heuristic_score(pd.DataFrame({"months_since_last_release": [1, 30]})).tolist() == [
        1.0,
        30.0,
    ]


def test_join_fails_for_labelled_rows_without_features():
    features = pd.DataFrame({"name": ["a"], "snapshot": ["s"], "x": [1]})
    labels = pd.DataFrame(
        {"name": ["a", "b"], "snapshot": ["s", "s"], "split": ["train"] * 2, "label": [0, 1]}
    )
    with pytest.raises(ValueError, match="1 labelled rows have no features"):
        join_features_and_labels(features, labels)


SETTINGS = {"min_gain_over_heuristic": 0.0, "max_drop_vs_champion": 0.02}


def test_gate_bootstraps_the_first_model():
    assert decide({}, None, has_champion=False, settings=SETTINGS) == (
        True,
        "bootstrap: no champion yet",
    )


def test_gate_requires_beating_the_heuristic_and_not_losing_to_the_champion():
    assert not decide(
        {"test_model_pr_auc": 0.3, "test_heuristic_pr_auc": 0.4}, None, True, SETTINGS
    )[0]
    assert not decide(
        {"test_model_pr_auc": 0.5, "test_heuristic_pr_auc": 0.4}, 0.6, True, SETTINGS
    )[0]
    assert decide({"test_model_pr_auc": 0.59, "test_heuristic_pr_auc": 0.4}, 0.6, True, SETTINGS)[0]
    assert not decide({}, None, True, SETTINGS)[0]


@pytest.mark.parametrize(
    ("risk", "expected"), [(0.05, "A"), (0.1, "B"), (0.34, "C"), (0.49, "D"), (0.9, "F")]
)
def test_grades_follow_the_bands(risk, expected):
    assert grade(risk, {"A": 0.1, "B": 0.2, "C": 0.35, "D": 0.5}) == expected
