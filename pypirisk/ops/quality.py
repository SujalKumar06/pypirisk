"""CI checks as DVC stages: ``python -m pypirisk.ops.quality {lint,test}``.

The DVC DAG is the CI pipeline. Its first two stages run ruff and pytest; ``ingest``
depends on the test report, so ``dvc repro`` stops before touching data when a check
fails. Both reports hold counts only, so an unchanged result does not invalidate the
stages after it.
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from pypirisk.io import write_json
from pypirisk.logs import configure_logging
from pypirisk.stagelog import stage_run

LINT_REPORT = Path("reports/lint.json")
TEST_REPORT = Path("reports/tests.json")
JUNIT_PATH = Path("reports/junit.xml")


class CheckFailedError(RuntimeError):
    """Raised when lint or tests fail, which stops ``dvc repro``."""


def count_results(junit: Path) -> dict[str, Counter]:
    """Count passed, failed and skipped tests per suite (``tests/<suite>/...``).

    Args:
        junit: A pytest ``--junitxml`` file.

    Returns:
        Counters keyed by suite name (``unit``, ``leakage``, ``integration``, ``e2e``).
    """
    counts: dict[str, Counter] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        suite = case.get("classname", "").split(".")[1:2] or ["other"]
        counts.setdefault(suite[0], Counter())[outcome] += 1
    return counts


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def lint() -> dict[str, int]:
    """Run ``ruff check`` and ``ruff format --check`` and write ``reports/lint.json``.

    Returns:
        ``ruff_check_exit`` and ``ruff_format_exit`` (0 means clean).

    Raises:
        CheckFailedError: If either command reports problems.
    """
    check = _run(["ruff", "check", "."])
    formatting = _run(["ruff", "format", "--check", "."])
    report = {"ruff_check_exit": check.returncode, "ruff_format_exit": formatting.returncode}
    write_json(LINT_REPORT, report)
    if check.returncode or formatting.returncode:
        raise CheckFailedError(
            f"ruff found problems:\n{check.stdout[-3000:]}{formatting.stdout[-2000:]}"
        )
    return report


def test() -> dict[str, Any]:
    """Run the unit, leakage and integration tests and write ``reports/tests.json``.

    Returns:
        Passed, failed and skipped counts per suite and in total.

    Raises:
        CheckFailedError: If any test fails or pytest cannot run.
    """
    result = _run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={JUNIT_PATH}"]
    )
    if not JUNIT_PATH.exists():
        raise CheckFailedError(
            f"pytest did not run:\n{result.stdout[-3000:]}{result.stderr[-2000:]}"
        )
    counts = count_results(JUNIT_PATH)
    report = {suite: dict(sorted(result.items())) for suite, result in sorted(counts.items())}
    report["total_failed"] = sum(result["failed"] for result in counts.values())
    report["total_passed"] = sum(result["passed"] for result in counts.values())
    write_json(TEST_REPORT, report)
    if result.returncode or report["total_failed"]:
        raise CheckFailedError(f"{report['total_failed']} tests failed:\n{result.stdout[-3000:]}")
    return report


def main() -> None:
    """Run the requested check inside a recorded stage run."""
    parser = argparse.ArgumentParser(description="Run a CI check as a DVC stage.")
    parser.add_argument("check", choices=["lint", "test"])
    args = parser.parse_args()
    configure_logging()
    with stage_run(args.check, profile="ci"):
        lint() if args.check == "lint" else test()


if __name__ == "__main__":
    main()
