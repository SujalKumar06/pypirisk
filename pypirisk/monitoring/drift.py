"""Feature drift: how far each feature's average has moved, in standard deviations.

The reference is the feature distribution of the latest labelled snapshot (the
population the model was last validated on). For each feature,

    drift = |mean(now) - mean(reference)| / std(reference)

so 1.0 means the average moved by one reference standard deviation.
"""

import pandas as pd


def reference_profile(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Record each feature's mean and standard deviation for later drift checks.

    Args:
        features: The reference population, one column per feature.

    Returns:
        ``{feature: {"mean": ..., "std": ...}}`` (missing values ignored).
    """
    return {
        column: {"mean": float(features[column].mean()), "std": float(features[column].std())}
        for column in features.columns
    }


def feature_drift(profile: dict[str, dict[str, float]], features: pd.DataFrame) -> dict[str, float]:
    """Measure how far each feature's mean has moved from the reference.

    Features with no spread in the reference are skipped: a shift cannot be expressed in
    standard deviations of a constant.

    Args:
        profile: Output of :func:`reference_profile`.
        features: The new sample (e.g. the serving features).

    Returns:
        Drift per feature present in both.
    """
    return {
        column: abs(float(features[column].mean()) - stats["mean"]) / stats["std"]
        for column, stats in profile.items()
        if column in features.columns and stats["std"] > 0
    }
