"""Rate-limited JSON-over-HTTP client used by the crawlers."""

import time
from collections.abc import Callable
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "pypirisk-crawler (+https://github.com/SujalKumar06/pypirisk)"


def no_wait(_: str) -> float:
    """Rate-limit hook that never waits; used outside Ray and in tests."""
    return 0.0


class HttpClient:
    """GET JSON documents, waiting for a rate-limit slot before each request.

    Transient failures (HTTP 429 and 5xx, connection errors) are retried with
    exponential back-off, honouring ``Retry-After``.
    """

    def __init__(
        self,
        timeout: float,
        reserve_slot: Callable[[str], float] = no_wait,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Create a client.

        Args:
            timeout: Per-request timeout in seconds.
            reserve_slot: Returns how long to wait before calling a host.
            headers: Extra headers, e.g. an ``Authorization`` header.
        """
        self.timeout = timeout
        self.reserve_slot = reserve_slot
        self.requests_sent = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, **(headers or {})})
        retry = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _throttle(self, host: str) -> None:
        wait = self.reserve_slot(host)
        if wait > 0:
            time.sleep(wait)
        self.requests_sent += 1

    def get_json(self, host: str, url: str, params: dict | None = None) -> Any | None:
        """GET a JSON document.

        Args:
            host: Rate-limit bucket name, e.g. ``"pypi"``.
            url: Full URL.
            params: Query parameters.

        Returns:
            The decoded body, or ``None`` if the server answers 404.

        Raises:
            requests.RequestException: On any other failure after retries.
        """
        self._throttle(host)
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def post_json(self, host: str, url: str, body: dict) -> Any:
        """POST a JSON body and decode the JSON answer.

        Args:
            host: Rate-limit bucket name.
            url: Full URL.
            body: JSON body.

        Returns:
            The decoded response.

        Raises:
            requests.RequestException: On failure after retries.
        """
        self._throttle(host)
        response = self.session.post(url, json=body, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
