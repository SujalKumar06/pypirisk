from pathlib import Path

import pandas as pd
import pytest

from pypirisk.config import load_params
from pypirisk.labels.build import LabelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def ts(value):
    return pd.Timestamp(value, tz="UTC")


def packages_frame(*rows):
    """Build a packages table; each row is a dict overriding the defaults."""
    defaults = {
        "repo_url": None,
        "archived": False,
        "archived_at": pd.NaT,
        "has_commit_data": True,
        "has_issue_data": True,
        "issues_complete_since": ts("2017-01-01"),
        "monorepo": False,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    for column in ("archived_at", "issues_complete_since"):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
    return frame


def releases_frame(rows):
    """rows: [(name, version, date)]"""
    frame = pd.DataFrame(rows, columns=["name", "version", "uploaded_at"])
    frame["uploaded_at"] = pd.to_datetime(frame["uploaded_at"], utc=True, format="ISO8601")
    return frame


def commits_frame(rows):
    """rows: [(repo_url, date, author, is_bot)]"""
    frame = pd.DataFrame(rows, columns=["repo_url", "authored_at", "author_id", "is_bot"])
    frame["authored_at"] = pd.to_datetime(frame["authored_at"], utc=True, format="ISO8601")
    return frame


def issues_frame(rows):
    """rows: [(repo_url, created, closed, is_pr, is_bot)]"""
    frame = pd.DataFrame(rows, columns=["repo_url", "created_at", "closed_at", "is_pr", "is_bot"])
    for column in ("created_at", "closed_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
    return frame


@pytest.fixture
def params():
    return load_params(REPO_ROOT / "params.yaml")


@pytest.fixture
def label_spec():
    return LabelSpec(
        lookback_months=12,
        outcome_months=18,
        max_commits_in_outcome=2,
        min_releases_for_eligibility=1,
    )
