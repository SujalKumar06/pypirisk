"""Pipeline status for the Pipeline Console: DAG, run history and stage reports."""

import json
from collections import deque
from pathlib import Path
from typing import Any

import yaml


def _paths(entries: list) -> list[str]:
    paths = []
    for entry in entries or []:
        paths.extend(entry.keys() if isinstance(entry, dict) else [entry])
    return paths


def _produces(output: str, dependency: str) -> bool:
    return dependency == output or dependency.startswith(output.rstrip("/") + "/")


def read_dag(dvc_yaml: Path) -> dict[str, Any]:
    """Derive the stage graph from ``dvc.yaml``, as ``dvc dag`` does.

    A stage depends on another when one of its dependencies is (or is inside) one of the
    other's outputs or metrics files.

    Args:
        dvc_yaml: Path to ``dvc.yaml``.

    Returns:
        ``stages`` (name, command, outputs) in file order and ``edges`` (from, to).
    """
    stages = yaml.safe_load(Path(dvc_yaml).read_text())["stages"]
    outputs = {
        name: _paths(stage.get("outs")) + _paths(stage.get("metrics"))
        for name, stage in stages.items()
    }
    edges = [
        {"from": producer, "to": consumer}
        for consumer, stage in stages.items()
        for producer, produced in outputs.items()
        if producer != consumer
        and any(_produces(out, dep) for out in produced for dep in _paths(stage.get("deps")))
    ]
    nodes = [
        {"name": name, "cmd": stage["cmd"], "outputs": outputs[name]}
        for name, stage in stages.items()
    ]
    return {"stages": nodes, "edges": edges}


def read_runs(path: Path, limit: int, status: str | None = None) -> list[dict[str, Any]]:
    """Return the most recent stage runs, newest first.

    Args:
        path: The ``pipeline_runs.jsonl`` file.
        limit: Maximum runs to return.
        status: Only runs with this status (``success`` or ``failed``).

    Returns:
        Run records.
    """
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as handle:
        runs = [json.loads(line) for line in handle if line.strip()]
    if status:
        runs = [run for run in runs if run.get("status") == status]
    return list(reversed(runs[-limit:]))


def last_run_per_stage(path: Path) -> dict[str, dict[str, Any]]:
    """Return the latest run of each stage.

    Args:
        path: The ``pipeline_runs.jsonl`` file.

    Returns:
        Stage name mapped to its latest run record.
    """
    latest: dict[str, dict[str, Any]] = {}
    if Path(path).exists():
        with Path(path).open(encoding="utf-8") as handle:
            for line in deque(handle, maxlen=5000):
                if line.strip():
                    run = json.loads(line)
                    latest[run["stage"]] = run
    return latest


def read_reports(directory: Path) -> dict[str, Any]:
    """Load the stage reports the console shows.

    Args:
        directory: The ``reports`` directory.

    Returns:
        ``ingest``, ``validation``, ``metrics`` and ``registry`` documents (``None`` when
        a stage has not produced one yet).
    """
    names = {
        "ingest": "ingest.json",
        "validation": "validation.json",
        "metrics": "metrics.json",
        "registry": "registry.json",
    }
    reports = {}
    for key, file_name in names.items():
        path = Path(directory) / file_name
        reports[key] = json.loads(path.read_text()) if path.exists() else None
    return reports
