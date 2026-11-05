import httpx
import pytest

from pypirisk_frontend.api_client import ApiClient, ApiError, is_valid_package_name
from pypirisk_frontend.display import describe_factor, dot_graph, number


def client_with(handler):
    client = ApiClient("http://api:8000", timeout=1)
    client._client = httpx.Client(
        base_url="http://api:8000", transport=httpx.MockTransport(handler)
    )
    return client


@pytest.mark.parametrize("name", ["requests", "typing_extensions", "zope.interface", "a"])
def test_valid_package_names_are_accepted(name):
    assert is_valid_package_name(name)


@pytest.mark.parametrize("name", ["", "-leading", "trailing-", "has space", "../etc/passwd"])
def test_invalid_package_names_are_rejected(name):
    assert not is_valid_package_name(name)


def test_risk_passes_live_and_backend_query_parameters():
    def handler(request):
        assert request.url.path == "/packages/toml/risk"
        assert request.url.params["live"] == "true"
        assert request.url.params["backend"] == "ray"
        return httpx.Response(200, json={"risk": 0.5})

    assert client_with(handler).risk("toml", live=True, backend="ray") == {"risk": 0.5}


def test_api_error_detail_is_shown_to_the_user():
    client = client_with(lambda r: httpx.Response(404, json={"detail": "'x' is not scored yet"}))
    with pytest.raises(ApiError, match="not scored yet"):
        client.risk("x")


def test_validation_errors_are_joined_into_one_message():
    client = client_with(
        lambda r: httpx.Response(422, json={"detail": [{"msg": "too long"}, {"msg": "bad"}]})
    )
    with pytest.raises(ApiError, match="too long; bad"):
        client.scan("x")


def test_unreachable_api_gives_a_friendly_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="unavailable"):
        client_with(handler).watch(10, 0)


def test_factor_is_described_in_plain_language():
    text = describe_factor(
        {"feature": "months_since_last_release", "value": 26.13, "contribution": 0.8}
    )
    assert text == "months since the last release = 26.1 raises the risk"
    assert "unknown" in describe_factor({"feature": "x", "value": None, "contribution": -1})


def test_dot_graph_colours_nodes_by_grade():
    dot = dot_graph([("app", "lib")], {"lib": "F"}, highlight={"app"})
    assert '"lib" [fillcolor="#c62828"' in dot
    assert '"app" -> "lib";' in dot
    assert "penwidth=3" in dot


def test_number_shows_a_dash_for_unknown_values():
    assert number(None, ".1f") == "–"
    assert number(2.345, ".1f") == "2.3"


def test_none_query_parameters_are_not_sent():
    def handler(request):
        assert "status" not in request.url.params
        return httpx.Response(200, json=[])

    assert client_with(handler).get("runs", "/pipeline/runs", limit=5, status=None) == []
