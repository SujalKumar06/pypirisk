"""Ingest stage: ``python -m pypirisk.ingest``.

Crawls (or, for the offline fixture profile, replays) every source and writes the five
tables to ``data/raw`` as Parquet, plus a coverage and throughput report.
"""

import logging
from pathlib import Path
from typing import Any

from pypirisk.config import active_profile
from pypirisk.ingest.cache import ResponseCache
from pypirisk.ingest.crawl import build_settings, crawl, seed_packages
from pypirisk.ingest.tables import build_tables, write_event_tables
from pypirisk.io import write_json
from pypirisk.rayutil import local_ray
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.ingest")


def coverage(tables: dict) -> dict[str, Any]:
    """Summarise how much of the universe each source covers.

    Args:
        tables: The ingest tables.

    Returns:
        Package count and the share with a repository, commits and issues.
    """
    packages = tables["packages"]
    total = max(len(packages), 1)
    return {
        "packages": len(packages),
        "seed_packages": int(packages["is_seed"].sum()),
        "share_with_repository": round(packages["repo_url"].notna().sum() / total, 4),
        "share_with_commits": round(packages["has_commit_data"].sum() / total, 4),
        "share_with_issues": round(packages["has_issue_data"].sum() / total, 4),
        "share_monorepo": round(packages["monorepo"].sum() / total, 4),
        "share_archived": round(packages["archived"].sum() / total, 4),
    }


def ingest(params: dict[str, Any]) -> None:
    """Crawl the active profile's universe and write tables, report and metrics.

    Args:
        params: The full parameter dictionary.
    """
    profile = active_profile(params)
    settings = build_settings(params, profile)
    seeds = seed_packages(params, profile, settings)
    logger.info("Starting crawl", extra={"profile": params["profile"], "seeds": len(seeds)})
    with local_ray(params["crawl"]["ray_num_cpus"], params["crawl"]["ray_object_store_mb"]):
        packages, repositories, stats = crawl(seeds, settings, profile)

    tables = build_tables(packages, repositories, seeds)
    output_dir = Path(params["ingest"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
    cache = ResponseCache(profile["cache_dir"], offline=True)
    event_rows = write_event_tables(repositories, cache, output_dir)

    report = {
        "profile": params["profile"],
        "crawl": stats,
        "coverage": coverage(tables),
        "rows": {**{name: len(table) for name, table in tables.items()}, **event_rows},
    }
    write_json(Path(params["ingest"]["report_path"]), report)
    logger.info("Ingest finished", extra={"rows": report["rows"], **report["coverage"]})


if __name__ == "__main__":
    run_stage("ingest", ingest)
