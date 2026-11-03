"""The served model: calibrated risk plus the top reasons, as an MLflow pyfunc.

Both serving paths (MLflow Serve and Ray Serve) and batch scoring load this one model,
so every consumer gets the same probability and the same explanation.
"""

import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from pypirisk.models.explain import shap_values, top_factors


class RiskModel(mlflow.pyfunc.PythonModel):
    """Calibrated abandonment probability with per-prediction explanations."""

    def __init__(
        self,
        kind: str,
        model: ClassifierMixin,
        calibrated: ClassifierMixin,
        feature_columns: list[str],
    ) -> None:
        """Wrap fitted estimators.

        Args:
            kind: Candidate type, used to pick the SHAP method.
            model: The fitted, uncalibrated candidate (for explanations).
            calibrated: The calibrated classifier (for probabilities).
            feature_columns: Input columns in training order.
        """
        self.kind = kind
        self.model = model
        self.calibrated = calibrated
        self.feature_columns = feature_columns

    def risk(self, features: pd.DataFrame) -> np.ndarray:
        """Return the calibrated probability of abandonment for each row.

        Args:
            features: Rows with every column in ``feature_columns``.

        Returns:
            Probabilities between 0 and 1.
        """
        return self.calibrated.predict_proba(self._inputs(features))[:, 1]

    def _inputs(self, features: pd.DataFrame) -> pd.DataFrame:
        return features[self.feature_columns].astype(float)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Score rows for MLflow serving and batch scoring.

        Args:
            context: MLflow context (unused).
            model_input: Rows with every column in ``feature_columns``.
            params: Inference parameters (unused).

        Returns:
            ``risk`` and ``top_factors`` (a JSON list of the three strongest reasons).
        """
        inputs = self._inputs(model_input)
        contributions = shap_values(self.kind, self.model, inputs)
        factors = [
            json.dumps(top_factors(contributions.loc[index], inputs.loc[index]))
            for index in inputs.index
        ]
        return pd.DataFrame({"risk": self.risk(inputs), "top_factors": factors}, index=inputs.index)
