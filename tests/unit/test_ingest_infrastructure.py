import pytest

from pypirisk.ingest.cache import CacheMissError, ResponseCache, cache_key
from pypirisk.ingest.http import HttpClient
from pypirisk.ingest.ratelimit import RateLimiter, TokenBucket


def test_cache_key_is_filesystem_safe_and_lowercase():
    assert (
        cache_key("https://GitHub.com/psf/requests", "2.0") == "https_github.com_psf_requests__2.0"
    )


def test_cache_round_trips_values(tmp_path):
    cache = ResponseCache(tmp_path, offline=False)
    cache.put("pypi", "requests", {"releases": [1, 2]})
    assert cache.get("pypi", "requests") == {"releases": [1, 2]}
    assert cache.get("pypi", "missing") is None


def test_fetch_calls_the_loader_once_then_reads_the_cache(tmp_path):
    cache = ResponseCache(tmp_path, offline=False)
    calls = []

    def loader():
        calls.append(1)
        return {"value": 1}

    assert cache.fetch("src", "key", loader) == {"value": 1}
    assert cache.fetch("src", "key", loader) == {"value": 1}
    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_offline_cache_raises_instead_of_calling_the_network(tmp_path):
    cache = ResponseCache(tmp_path, offline=True)
    with pytest.raises(CacheMissError):
        cache.fetch("src", "key", lambda: pytest.fail("loader must not run offline"))


def test_token_bucket_allows_a_burst_then_spaces_requests():
    bucket = TokenBucket(rate=2.0, capacity=1.0, tokens=1.0, updated=0.0)
    assert bucket.reserve(now=0.0) == 0.0
    assert bucket.reserve(now=0.0) == pytest.approx(0.5)
    assert bucket.reserve(now=0.0) == pytest.approx(1.0)


def test_token_bucket_refills_over_time():
    bucket = TokenBucket(rate=1.0, capacity=1.0, tokens=0.0, updated=0.0)
    assert bucket.reserve(now=5.0) == 0.0


def test_rate_limiter_does_not_limit_unknown_hosts():
    assert RateLimiter({"pypi": 1.0}).reserve("elsewhere") == 0.0


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self.body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.body


def test_http_client_returns_none_for_404_and_waits_for_a_slot(monkeypatch):
    waits = []
    client = HttpClient(timeout=1, reserve_slot=lambda host: waits.append(host) or 0.0)
    monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse(404))
    assert client.get_json("pypi", "https://example.org") is None
    assert waits == ["pypi"]
    assert client.requests_sent == 1


def test_http_client_decodes_json(monkeypatch):
    client = HttpClient(timeout=1)
    monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse(200, {"ok": True}))
    assert client.get_json("pypi", "https://example.org") == {"ok": True}
