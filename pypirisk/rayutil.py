"""Start Ray with the resource limits from ``params.yaml``."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import ray


@contextmanager
def local_ray(num_cpus: int, object_store_mb: int) -> Iterator[None]:
    """Run a local Ray instance for the duration of a ``with`` block.

    The object store is capped because the target machine has little free memory.
    ``RAY_TMPDIR`` moves Ray's session files off a nearly full system disk.

    Args:
        num_cpus: CPUs Ray may schedule tasks on.
        object_store_mb: Shared-memory object store size in MB.

    Yields:
        Nothing; Ray is shut down when the block exits.
    """
    ray.init(
        num_cpus=num_cpus,
        object_store_memory=object_store_mb * 1024 * 1024,
        include_dashboard=False,
        logging_level=logging.WARNING,
        _temp_dir=os.environ.get("RAY_TMPDIR") or None,
    )
    try:
        yield
    finally:
        ray.shutdown()
