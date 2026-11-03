"""Letter grades for risk scores, shared by batch scoring and the API."""


def grade(risk: float, bands: dict[str, float]) -> str:
    """Map a probability to a letter grade.

    Args:
        risk: Probability of abandonment.
        bands: Upper bound of each grade, in order (e.g. ``{"A": 0.1, "B": 0.2}``).

    Returns:
        The first grade whose bound exceeds ``risk``, else ``"F"``.
    """
    for letter, upper in bands.items():
        if risk < upper:
            return letter
    return "F"
