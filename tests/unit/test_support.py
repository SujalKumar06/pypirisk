import json
import logging

import pandas as pd
import pytest

from pypirisk.config import active_profile, load_params
from pypirisk.io import read_json, read_table, write_json
from pypirisk.logs import JsonFormatter, TextFormatter
from pypirisk.timeutils import add_months, parse_timestamp
from pypirisk.tracking import file_md5, lineage_tags


def make_log_record():
    record = logging.makeLogRecord({"name": "test", "levelname": "INFO", "msg": "hello"})
    record.package = "requests"
    return record


def test_json_formatter_includes_extra_fields():
    payload = json.loads(JsonFormatter().format(make_log_record()))
    assert payload["message"] == "hello"
    assert payload["package"] == "requests"


def test_text_formatter_appends_extra_fields():
    assert "hello" in TextFormatter().format(make_log_record())
    assert "package=requests" in TextFormatter().format(make_log_record())


def test_json_file_round_trips(tmp_path):
    write_json(tmp_path / "doc.json", {"k": "v"})
    assert read_json(tmp_path / "doc.json") == {"k": "v"}


def test_read_table_accepts_a_file_or_a_directory_with_plain_dtypes(tmp_path):
    frame = pd.DataFrame({"t": pd.to_datetime(["2024-01-01"], utc=True), "n": [1]})
    frame.to_parquet(tmp_path / "single.parquet")
    (tmp_path / "multi").mkdir()
    frame.to_parquet(tmp_path / "multi" / "part-0.parquet")
    for name in ("single", "multi"):
        table = read_table(tmp_path, name)
        assert isinstance(table["t"].dtype, pd.DatetimeTZDtype)
        assert str(table["t"].dt.tz) == "UTC"


def test_load_params_rejects_non_mapping(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text("- a list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_params(path)


def test_active_profile_follows_the_profile_key():
    params = {"profile": "pilot", "profiles": {"pilot": {"offline": False}}}
    assert active_profile(params) == {"offline": False}


def test_fixture_profile_replays_the_cache_offline(params):
    assert params["profiles"]["fixture"]["offline"] is True


def test_add_months_clamps_to_month_end():
    assert add_months(parse_timestamp("2021-01-31"), 1) == parse_timestamp("2021-02-28")
    assert add_months(parse_timestamp("2021-01-15"), -13) == parse_timestamp("2019-12-15")


def test_file_md5_matches_known_digest(tmp_path):
    path = tmp_path / "data.txt"
    path.write_bytes(b"abc")
    assert file_md5(path) == "900150983cd24fb0d6963f7d28e17f72"


def test_lineage_tags_include_commit_dirty_flag_and_data_hashes(tmp_path):
    path = tmp_path / "features.parquet"
    path.write_bytes(b"abc")
    tags = lineage_tags({"features": path})
    assert set(tags) == {"git_commit", "git_dirty", "data_md5.features"}
