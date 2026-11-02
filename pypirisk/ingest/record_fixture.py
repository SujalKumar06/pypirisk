"""Record the offline fixture: ``python -m pypirisk.ingest.record_fixture``.

Runs the crawl for ``fixture_packages`` with the network enabled, filling
``data/fixture/cache``. Afterwards the ``fixture`` profile replays that cache offline,
which is how ``dvc repro`` runs on a fresh clone without calling any API.
"""

import logging

from pypirisk.config import active_profile
from pypirisk.ingest.crawl import build_settings, crawl, seed_packages
from pypirisk.rayutil import local_ray
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.ingest.record_fixture")


def record(params: dict) -> None:
    """Crawl the fixture packages online into the fixture cache.

    Args:
        params: The full parameter dictionary.
    """
    params = {**params, "profile": "fixture"}
    profile = {**active_profile(params), "offline": False}
    settings = build_settings(params, profile)
    with local_ray(params["crawl"]["ray_num_cpus"], params["crawl"]["ray_object_store_mb"]):
        _, _, stats = crawl(seed_packages(params, profile, settings), settings, profile)
    logger.info("Fixture recorded", extra=stats)


if __name__ == "__main__":
    run_stage("record_fixture", record)
