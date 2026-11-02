"""On-disk cache of trimmed API responses, so crawls are resumable and replayable.

Every network result is stored as gzip-compressed JSON under
``<root>/<source>/<key>.json.gz``. A re-run reads the cache first, so it costs no rate
limit. In offline mode (the fixture profile) a cache miss is an error instead of a request.
"""

import gzip
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^a-z0-9._-]+")


class CacheMissError(LookupError):
    """Raised in offline mode when a response is not in the cache."""


def cache_key(*parts: str) -> str:
    """Build a filesystem-safe cache key from its parts.

    Args:
        *parts: Key components such as a package name and a version.

    Returns:
        A lowercase key using only letters, digits, ``.``, ``_`` and ``-``.
    """
    return "__".join(_UNSAFE.sub("_", part.lower()) for part in parts)


class ResponseCache:
    """Read-through cache for trimmed responses."""

    def __init__(self, root: Path, offline: bool) -> None:
        """Create a cache.

        Args:
            root: Directory that holds one sub-directory per source.
            offline: Raise :class:`CacheMissError` instead of calling the loader.
        """
        self.root = Path(root)
        self.offline = offline
        self.hits = 0
        self.misses = 0

    def path(self, source: str, key: str) -> Path:
        """Return the file that stores ``key`` for ``source``."""
        return self.root / source / f"{key}.json.gz"

    def get(self, source: str, key: str) -> Any | None:
        """Return the cached value, or ``None`` if absent.

        Args:
            source: Data source name, e.g. ``"pypi"``.
            key: Key from :func:`cache_key`.

        Returns:
            The cached JSON value, or ``None``.
        """
        path = self.path(source, key)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    def put(self, source: str, key: str, value: Any) -> None:
        """Store a value atomically, so a crash never leaves a half-written entry.

        Args:
            source: Data source name.
            key: Key from :func:`cache_key`.
            value: JSON-serialisable value.
        """
        path = self.path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        os.replace(tmp, path)

    def fetch(self, source: str, key: str, loader: Callable[[], Any]) -> Any:
        """Return the cached value, calling ``loader`` and caching its result on a miss.

        Args:
            source: Data source name.
            key: Key from :func:`cache_key`.
            loader: Produces the value from the network.

        Returns:
            The cached or freshly loaded value.

        Raises:
            CacheMissError: In offline mode when the value is not cached.
        """
        cached = self.get(source, key)
        if cached is not None:
            self.hits += 1
            return cached
        if self.offline:
            raise CacheMissError(f"{source}/{key} is not cached and the profile is offline")
        self.misses += 1
        value = loader()
        self.put(source, key, value)
        return value
