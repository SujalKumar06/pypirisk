"""Promotion and rollback in the MLflow Model Registry.

The serving alias (``@champion``) is the deployment switch: the model servers load
whatever version it points to. Promotion moves the alias and records the version it
replaced in a ``previous_champion`` tag on the new version, so a rollback is one alias
move back.
"""

import logging

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

PREVIOUS_TAG = "previous_champion"


class RollbackError(RuntimeError):
    """Raised when there is no earlier champion to roll back to."""


def champion_version(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    """Return the version the alias points to.

    Args:
        client: MLflow client.
        name: Registered model name.
        alias: Serving alias.

    Returns:
        The model version, or ``None`` if the model or alias does not exist yet.
    """
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def promote(client: MlflowClient, name: str, alias: str, version: str) -> str | None:
    """Point the alias at ``version`` and remember which version it replaced.

    Args:
        client: MLflow client.
        name: Registered model name.
        alias: Serving alias.
        version: Version to promote.

    Returns:
        The replaced version, or ``None`` if there was no champion.
    """
    previous = champion_version(client, name, alias)
    client.set_registered_model_alias(name, alias, version)
    if previous is not None and previous.version != version:
        client.set_model_version_tag(name, version, PREVIOUS_TAG, previous.version)
        logger.info("Promoted model", extra={"version": version, "replaced": previous.version})
        return previous.version
    logger.info("Promoted model", extra={"version": version, "replaced": None})
    return None


def rollback(client: MlflowClient, name: str, alias: str, reason: str) -> tuple[str, str]:
    """Move the alias back to the champion before the current one.

    Args:
        client: MLflow client.
        name: Registered model name.
        alias: Serving alias.
        reason: Why the rollback happened; stored as a tag on the demoted version.

    Returns:
        The demoted version and the restored version.

    Raises:
        RollbackError: If there is no current champion or no earlier one recorded.
    """
    current = champion_version(client, name, alias)
    if current is None:
        raise RollbackError(f"{name}@{alias} does not exist")
    previous = current.tags.get(PREVIOUS_TAG)
    if not previous:
        raise RollbackError(f"{name} version {current.version} has no previous champion")
    client.set_registered_model_alias(name, alias, previous)
    client.set_model_version_tag(name, current.version, "rolled_back", reason[:500])
    logger.warning(
        "Rolled back model",
        extra={"demoted": current.version, "restored": previous, "reason": reason},
    )
    return current.version, previous
