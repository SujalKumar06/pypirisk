"""Token-bucket rate limiting shared by every crawler task.

The crawl runs as many Ray tasks at once, but each API host has one rate limit. A
single :class:`RateLimiter` instance runs as a Ray actor, and every task asks it for a
slot before calling a host, so the limit holds across all tasks.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """A token bucket that hands out reservations.

    Attributes:
        rate: Tokens added per second (the sustained request rate).
        capacity: Largest burst allowed.
        tokens: Tokens available now; negative values are reservations already made.
        updated: Time of the last refill, in seconds.
    """

    rate: float
    capacity: float = 1.0
    tokens: float = field(default=1.0)
    updated: float = field(default_factory=time.monotonic)

    def reserve(self, now: float) -> float:
        """Reserve one token.

        Args:
            now: Current time in seconds (``time.monotonic()``).

        Returns:
            Seconds the caller must wait before using the token; 0 if one is free now.
        """
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        self.tokens -= 1
        return 0.0 if self.tokens >= 0 else -self.tokens / self.rate


class RateLimiter:
    """One token bucket per API host; run it as a Ray actor with ``ray.remote``."""

    def __init__(self, rates: dict[str, float]) -> None:
        """Create a limiter.

        Args:
            rates: Requests per second for each host name.
        """
        self._buckets = {host: TokenBucket(rate=rate) for host, rate in rates.items()}

    def reserve(self, host: str) -> float:
        """Reserve a request slot on ``host``.

        Args:
            host: A host name from ``rates``; unknown hosts are not limited.

        Returns:
            Seconds to wait before sending the request.
        """
        bucket = self._buckets.get(host)
        return bucket.reserve(time.monotonic()) if bucket else 0.0
