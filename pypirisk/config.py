"""Load pipeline parameters from ``params.yaml``."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Read the pipeline parameter file.

    Args:
        path: Location of the YAML parameter file.

    Returns:
        The parsed parameters as nested dictionaries.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain a YAML mapping.
    """
    with Path(path).open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if not isinstance(params, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(params).__name__}")
    return params


def active_profile(params: dict[str, Any]) -> dict[str, Any]:
    """Return the settings of the data profile selected by ``params["profile"]``.

    Args:
        params: The full parameter dictionary.

    Returns:
        The selected entry of ``params["profiles"]``.

    Raises:
        KeyError: If the selected profile is not defined.
    """
    return params["profiles"][params["profile"]]
